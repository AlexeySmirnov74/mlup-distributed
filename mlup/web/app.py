import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, wraps
from typing import Type, Optional, Dict

from fastapi import FastAPI, Request as FastAPIRequest, Response as FastAPIResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel as PydanticBaseModel, ValidationError
import uvicorn

from mlup.config import LOGGING_CONFIG
from mlup.constants import ITEM_ID_COL_NAME, WebAppArchitecture, PREDICT_ID_HEADER
from mlup.errors import (
    JsonValidationError,
    PredictError,
    WebAppLoadError,
    PredictWaitResultError,
    PredictTransformDataError,
    PredictValidationInnerDataError,
)
from mlup.ml.model import MLupModel
from mlup.utils.crypto import generate_unique_id
from mlup.utils.interspection import get_class_by_path
from mlup.utils.logging import configure_logging_formatter
from mlup.utils.loop import run_async
from mlup.web.api_docs import openapi_schema
from mlup.web.api_errors import (
    ApiErrorResponse,
    api_exception_handler,
    ApiRequestError,
    predict_errors_handler,
    api_request_error_handler,
)
from mlup.web.api_validators import create_pydantic_predict_model, MlModelPredictRequest
from mlup.web.architecture.base import BaseWebAppArchitecture

logger = logging.getLogger("mlup")


def _requests_throttling(func):
    @wraps(func)
    async def wrap(self, *args, **kwargs):
        if not self.conf.throttling_max_requests:
            return await func(self, *args, **kwargs)

        if self._throttling_max_requests_current_count < self.conf.throttling_max_requests:
            self._throttling_max_requests_current_count += 1
            try:
                return await func(self, *args, **kwargs)
            finally:
                self._throttling_max_requests_current_count -= 1
        else:
            raise ApiRequestError(
                "Max requests in app. Please try again later.",
                status_code=429,
                type="throttling_error",
            )

    return wrap


def _set_predict_id_to_response_headers(func):
    @wraps(func)
    async def wrap(*args, response: FastAPIResponse, **kwargs):
        predict_id = generate_unique_id()
        response.headers[PREDICT_ID_HEADER] = predict_id
        try:
            return await func(*args, response=response, **kwargs)
        except Exception as e:
            # attach predict_id to any exception (MLup convention)
            e.predict_id = predict_id
            raise

    return wrap


@dataclass
class WebAppConfig:
    """
    WebAppConfig config class. This class have settings for web app.
    """

    # WebApp web outer interface settings
    host: str = "0.0.0.0"
    port: int = 8009
    web_app_version: str = "1.0.0.0"

    # Can use only single from two params
    column_validation: bool = False
    custom_column_pydantic_model: Optional[Type[PydanticBaseModel]] = None

    # WebApp architecture settings
    mode: WebAppArchitecture = WebAppArchitecture.directly_to_predict

    # Max queue size for waiting batching.
    max_queue_size: int = 100

    # Max time member predict result data. Need for clean if client not returned for predict result.
    ttl_predicted_data: int = 60

    deploy_type: str = "test"
    web_description: str = ""
    model_directory_swagger: str = "/model1"
    redis_url: str = "redis://localhost:6379"
    get_first_model_argument: bool = True
    get_formatted_result: bool = True
    redis_client: str = ""
    is_use_redis: bool = False
    redis_queue_name: str = "predict_queue"

    # Max time wait results for clients, in single request.
    ttl_client_wait: float = 30.0

    # Min batch len for start predict.
    min_batch_len: int = 10

    # Max time for pending before run batching.
    batch_worker_timeout: float = 1.0

    # Added get-predict api method and add return predict_id to predict response.
    is_long_predict: bool = False

    # WebApp work settings
    show_docs: bool = True
    debug: bool = False

    # Max count simultaneous requests to web app
    throttling_max_requests: Optional[int] = None

    # Max count objects to predict in single request
    throttling_max_request_len: Optional[int] = None

    # Validate request body using pydantic schema generated from model signature.
    # If False, raw JSON is passed to architecture as-is (useful for demos / raw DS functions).
    validate_predict_payload: bool = True

    # Wait time for graceful shutdown web app
    timeout_for_shutdown_daemon: float = 3.0

    # Uvicorn settings
    uvicorn_kwargs: Dict = field(default_factory=dict, repr=False)

    # Column name for unique item_id
    item_id_col_name: str = ITEM_ID_COL_NAME

    def wb_dict(self):
        res = {}
        for n, f in WebAppConfig.__dataclass_fields__.items():
            if f.repr is True:
                v = getattr(self, n)
                if isinstance(v, Enum):
                    v = v.value
                res[n] = v
        return res

    def wb_str(self, need_spaces: bool = False):
        res = []
        space = "    " if need_spaces else ""
        for n, f in WebAppConfig.__dataclass_fields__.items():
            if f.repr is True:
                v = getattr(self, n)
                if isinstance(v, Enum):
                    v = v.value
                res.append(space + f"{n}={v}")
        return "\n".join(res)


@dataclass(repr=True)
class MLupWebApp:
    """This is main UP web app class."""

    ml: MLupModel = field(repr=False)
    conf: WebAppConfig = field(default_factory=WebAppConfig, repr=False)

    # WebApp inner settings
    _fast_api: FastAPI = field(init=False, default=None, repr=False)
    _throttling_max_requests_current_count: int = field(init=False, default=0, repr=False)
    _predict_inner_pydantic_model: Type[MlModelPredictRequest] = field(init=False, repr=False)
    _daemon_thread: threading.Thread = field(init=False, default=None, repr=False)
    _uvicorn_server: uvicorn.Server = field(init=False, repr=False)
    _architecture_obj: BaseWebAppArchitecture = field(init=False, repr=False)

    def __del__(self):
        self.stop()

    def __getstate__(self):
        """Before pickle object"""
        logger.info(f"Running binarization {self}.")
        attributes = self.__dict__.copy()
        attributes.pop("_fast_api", None)
        attributes.pop("_predict_inner_pydantic_model", None)
        attributes.pop("_daemon_thread", None)
        attributes.pop("_uvicorn_server", None)
        attributes.pop("_architecture_obj", None)
        return attributes

    def __setstate__(self, state):
        """After unpickle object"""
        logger.info(f"Running an {self} load from binary data.")
        self.__dict__ = state
        self.load()

    @property
    def loaded(self) -> bool:
        return self._fast_api is not None

    @property
    def app(self) -> FastAPI:
        if not self.loaded:
            raise WebAppLoadError("web nor creating. Please call web.load().")
        return self._fast_api

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Startup web app code
        await self._architecture_obj.run()
        yield
        # Shutdown web app code
        await self._architecture_obj.stop()
        configure_logging_formatter("default")

    def _request_len_throttling(self, data_for_predict: Dict):
        """
        Throttle by number of items/rows in a single request.

        NOTE: In the original snippet there was a bug:
              it raised for any non-empty payload.
        """
        x_for_predict = self.ml.get_X_from_predict_data(data_for_predict, remove=False)

        # If model expects batch-like input, x_for_predict may be list/array-like.
        if x_for_predict is None:
            return

        try:
            req_len = len(x_for_predict)
        except Exception:
            return

        if req_len > int(self.conf.throttling_max_request_len):
            raise ApiRequestError(
                "The query exceeded the limit on the number of rows for the predict. Please downsize your request.",
                status_code=429,
                type="throttling_error",
            )

    def _create_app(self):
        fast_api_kwargs = {}
        if self.conf.show_docs is False:
            fast_api_kwargs["docs_url"] = None
            fast_api_kwargs["redoc_url"] = None

        self._fast_api = FastAPI(
            debug=self.conf.debug,
            title=f"model: {self.ml.conf.name} v{self.ml.conf.version}.",
            description=(
                f"Web application for use {self.ml.conf.name} v{self.ml.conf.version} in web.<p>"
                + self.conf.web_description
                + "</p>"
            ),
            version=self.conf.web_app_version,
            lifespan=self._lifespan,
            exception_handlers={
                RequestValidationError: api_exception_handler,
                ValidationError: api_exception_handler,
                ApiRequestError: api_request_error_handler,
                # Predict errors
                PredictError: predict_errors_handler,
                PredictWaitResultError: predict_errors_handler,
                PredictTransformDataError: predict_errors_handler,
                PredictValidationInnerDataError: predict_errors_handler,
            },
            responses={
                422: {
                    "description": "Error with validation input data",
                    "model": ApiErrorResponse,
                },
                429: {
                    "description": "Throttling input request error",
                    "model": ApiErrorResponse,
                },
                500: {
                    "description": "Error with predict process exception",
                    "model": ApiErrorResponse,
                },
            },
            **fast_api_kwargs,
        )

        # Set web api points
        self.app.add_api_route(self.conf.model_directory_swagger + "/health", self.http_health, methods=["GET"])

        if self.conf.debug:
            self.app.add_api_route(
                self.conf.model_directory_swagger + "/info", self.debug_info, methods=["GET"], name="info"
            )
        else:
            self.app.add_api_route(self.conf.model_directory_swagger + "/info", self.info, methods=["GET"], name="info")

        self.app.add_api_route(self.conf.model_directory_swagger + "/predict", self.predict, methods=["POST"])

        self._architecture_obj = get_class_by_path(self.conf.mode)(
            fast_api=self._fast_api,
            ml=self.ml,
            item_id_col_name=self.conf.item_id_col_name,
            # Worker settings
            max_queue_size=self.conf.max_queue_size,
            ttl_predicted_data=self.conf.ttl_predicted_data,
            is_use_redis=self.conf.is_use_redis,
            web_description=self.conf.web_description,
            deploy_type=self.conf.deploy_type,
            model_directory_swagger=self.conf.model_directory_swagger,
            redis_url=self.conf.redis_url,
            get_first_model_argument=self.conf.get_first_model_argument,
            get_formatted_result=self.conf.get_formatted_result,
            redis_client=self.conf.redis_client,
            redis_queue_name=self.conf.redis_queue_name,
            # Batching settings
            min_batch_len=self.conf.min_batch_len,
            batch_worker_timeout=self.conf.batch_worker_timeout,
            # Wait result from worker settings
            is_long_predict=self.conf.is_long_predict,
            ttl_client_wait=self.conf.ttl_client_wait,
        )

        self._predict_inner_pydantic_model = create_pydantic_predict_model(
            self.ml,
            self.conf.column_validation,
            custom_column_pydantic_model=self.conf.custom_column_pydantic_model,
        )

    def _run(self, in_thread: bool = False):
        if "loop" not in self.conf.uvicorn_kwargs:
            self.conf.uvicorn_kwargs["loop"] = "none" if in_thread else "auto"
        if "timeout_graceful_shutdown" not in self.conf.uvicorn_kwargs:
            self.conf.uvicorn_kwargs["timeout_graceful_shutdown"] = self.conf.timeout_for_shutdown_daemon
        if "log_config" not in self.conf.uvicorn_kwargs:
            self.conf.uvicorn_kwargs["log_config"] = LOGGING_CONFIG

        configure_logging_formatter("web")

        logger.info(f"MLup application will be launched at: http://{self.conf.host}:{self.conf.port}")
        if self.conf.show_docs:
            logger.info(f"You can open your application's API documentation at http://{self.conf.host}:{self.conf.port}/docs")

        self._uvicorn_server = uvicorn.Server(
            uvicorn.Config(self.app, **self.conf.uvicorn_kwargs),
        )
        self._uvicorn_server.run()

    def load_web_app_settings(self):
        """Load web app settings"""
        self.conf.uvicorn_kwargs["host"] = self.conf.host
        self.conf.uvicorn_kwargs["port"] = self.conf.port
        self.conf.uvicorn_kwargs["timeout_graceful_shutdown"] = int(self.conf.timeout_for_shutdown_daemon)

    def load(self):
        """Create and full load web app"""
        logger.debug("Run load Web application")
        if not self.ml.loaded:
            raise WebAppLoadError("Model not loaded to memory. Analyze impossible. Please call ml.load().")

        logger.debug(f"Load Web application with settings:\n{self.conf.wb_str(need_spaces=True)}")

        if self.conf.throttling_max_requests is not None and self.conf.throttling_max_requests < 1:
            raise WebAppLoadError(
                "The param throttling_max_requests must be greater than 0. "
                f"Now it is {self.conf.throttling_max_requests}."
            )

        if self.conf.throttling_max_request_len is not None and self.conf.throttling_max_request_len < 1:
            raise WebAppLoadError(
                "The param throttling_max_request_len must be greater than 0. "
                f"Now it is {self.conf.throttling_max_request_len}."
            )

        if self.conf.column_validation and not self.ml.conf.columns:
            raise WebAppLoadError(
                "The param column_validation=True must use only, when there is ml.columns. "
                f"Now ml.columns is {self.ml.conf.columns}."
            )

        if self.conf.column_validation and self.conf.custom_column_pydantic_model:
            raise WebAppLoadError(
                "Only one of the two parameters can be used: column_validation, custom_column_pydantic_model. "
                f"Now set column_validation={self.conf.column_validation}, "
                f"custom_column_pydantic_model={self.conf.custom_column_pydantic_model}."
            )

        self.load_web_app_settings()
        self._create_app()
        self._architecture_obj.load()

        self.app.openapi = partial(openapi_schema, app=self.app, ml=self.ml, model_dir_swagger=self.conf.model_directory_swagger)

    def stop(self, shutdown_timeout: Optional[float] = None):
        """
        Stop web app, if web app was runned in daemon mode.
        """
        if self._daemon_thread and self._daemon_thread.is_alive():
            # Shutdown uvicorn not in main thread
            self._uvicorn_server.should_exit = True

            run_async(
                asyncio.wait_for,
                self._uvicorn_server.shutdown(),
                shutdown_timeout or self.conf.timeout_for_shutdown_daemon,
            )
            self._daemon_thread.join(shutdown_timeout or self.conf.timeout_for_shutdown_daemon)

    def run(self, daemon: bool = False):
        """
        Run web app with ML model.
        """
        if not self.ml.loaded:
            raise WebAppLoadError("ML Model not loaded to memory. For run web app, please call ml.load(), or ml.load().")

        if not self.loaded:
            raise WebAppLoadError("For run web app, please call web.load(), or web.load().")

        if daemon is not True:
            self._run(in_thread=False)
            return

        if self._daemon_thread and self._daemon_thread.is_alive():
            logger.error(f"WebApp is already running. Thread name {self._daemon_thread.name}")
            raise WebAppLoadError(f"WebApp is already running. Thread name {self._daemon_thread.name}")

        self._daemon_thread = threading.Thread(
            target=self._run,
            kwargs={"in_thread": True},
            daemon=False,
            name="MLupWebAppDaemonThread",
        )
        self._daemon_thread.start()

        logger.info("Waiting start uvicorn proc with web app 30.0 seconds.")
        time.sleep(0.1)
        time_run = time.monotonic()
        while not self._uvicorn_server and time.monotonic() - time_run < 30.0:
            time.sleep(0.1)
        while self._uvicorn_server.started is False and time.monotonic() - time_run < 30.0:
            time.sleep(0.1)

    async def http_health(self):
        return {"status": 200}

    async def debug_info(self):
        info = await self.info()
        info.update(
            {
                "model_config": self.ml.conf.ml_dict(),
                "web_app_config": self.conf.wb_dict(),
            }
        )
        return info

    async def info(self):
        return {
            "model_info": {
                "name": self.ml.conf.name,
                "version": self.ml.conf.version,
                "deploy_type": self.conf.deploy_type,
                "description": self.conf.web_description,
            },
            "web_app_info": {
                "version": self.conf.web_app_version,
            },
        }

    @_set_predict_id_to_response_headers
    @_requests_throttling
    async def predict(self, request: FastAPIRequest, response: FastAPIResponse):
        predict_id = response.headers[PREDICT_ID_HEADER]

        # --- Read JSON body ---
        try:
            request_data = await request.json()
        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON: %s", e)
            raise JsonValidationError("Json is not valid", predict_id=predict_id)

        # --- Optional validation (config-driven) ---
        if self.conf.validate_predict_payload:
            try:
                # Request must be a dict for **kwargs; if it isn't, TypeError will be raised
                predict_request_body = self._predict_inner_pydantic_model(**request_data)
                data_for_predict = predict_request_body.dict()
            except TypeError as e:
                raise PredictValidationInnerDataError(
                    msg=f"Predict payload must be a JSON object: {e}",
                    predict_id=predict_id,
                )
            except (ValidationError, RequestValidationError) as e:
                raise PredictValidationInnerDataError(
                    msg=f"Predict payload validation error: {e}",
                    predict_id=predict_id,
                )
        else:
            # No schema validation (raw JSON goes to architecture)
            data_for_predict = request_data

        # Throttling by request length (items/rows)
        if self.conf.throttling_max_request_len:
            self._request_len_throttling(data_for_predict)

        # Predict in web app architecture object
        predict_result = await self._architecture_obj.predict(data_for_predict, predict_id=predict_id)

        # Result (keep MLup-compatible response format)
        return {"predict_result": predict_result}
