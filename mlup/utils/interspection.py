import importlib
import inspect
import logging
from enum import Enum
from typing import Callable, Any, Union, Optional, Type, Generic, Sequence, Dict, Tuple

try:
    from typing import get_args as typing_get_args, get_origin as typing_get_origin
# For Python <= 3.7
except ImportError:
    typing_get_args = lambda t: getattr(t, '__args__', ()) if t is not Generic else Generic  # noqa: E731
    typing_get_origin = lambda t: getattr(t, '__origin__', None)  # noqa: E731

from mlup.constants import IS_X, THERE_IS_ARGS, DEFAULT_X_ARG_NAME, BinarizationType, LoadedFile
from mlup.utils.profiling import TimeProfiler


logger = logging.getLogger('mlup')


SUPPORTED_PRIMITIVE_TYPES = {
    int: 'int',
    float: 'float',
    bool: 'bool',
    str: 'str',
}


def _is_optional(field: Any) -> bool:
    return typing_get_origin(field) is Union and type(None) in typing_get_args(field)


def _is_sequence(field: Any) -> bool:
    _type_for_check = typing_get_origin(field)
    _collection_types = (list, set, tuple)
    try:
        return ((field is not None and field in _collection_types)
                or (_type_for_check is not None and issubclass(_type_for_check, Sequence)))
    except TypeError:
        # If _type_for_check not in class object& For example Optional[int].
        return False


def parse_attributes_from_generic_type(
    param: inspect.Parameter,
) -> Tuple[Dict[str, Any], bool]:
    """
    Search and return primitive type from single level of Generic.
    If ont found supported types, return default type = str.

    :param inspect.Parameter param: Parameter that needs to be parsed.

    :return: Attributes from parsed Generic and result parsing.
        If bool is True, then parsing was success, else parsing was failure.
        {"type": "int", "required": False, "collection_type": None}, True
        Key "type" is optional.
        Key "collection_type" is optional.
    :rtype: Dict[str, Any], bool

    """
    result = {
        'required': True,
    }
    _types_for_analyze = typing_get_args(param.annotation)

    logger.debug(f"Analyze argument '{param.name}', attempt to pick up determine primitive type.")

    if _is_optional(param.annotation):
        result['required'] = False
    if _is_sequence(param.annotation):
        result['collection_type'] = 'List'
    if len(_types_for_analyze) > 0 and _is_sequence(_types_for_analyze[0]):
        result['collection_type'] = 'List'
        _types_for_analyze = typing_get_args(_types_for_analyze[0])

    for p in _types_for_analyze:
        if p in SUPPORTED_PRIMITIVE_TYPES:
            result['type'] = SUPPORTED_PRIMITIVE_TYPES[p]
            break

    _parse_error = False

    if 'type' not in result:
        logger.warning(f"Cannot determine primitive type for '{param.name}'.")
        _parse_error = True

    logger.debug(f"For argument '{param.name}' parsing result '{result}'")

    return result, _parse_error


def get_class_by_path(path_to_class: Union[str, Enum]) -> Any:
    """
    Get class by path to class. Use importlib.import_module.

    :param Union[str, Enum] path_to_class: Path to class for import and return.

    """
    if isinstance(path_to_class, Enum):
        path_to_class = path_to_class.value

    # Try get class by path
    try:
        return import_obj(path_to_class)
    except (ModuleNotFoundError, AttributeError) as e:
        logger.warning(f'Object {path_to_class} import error. {e}')
        raise


def import_obj(path_to_import_obj: str) -> Any:
    """
    Import any python object.

    :param str path_to_import_obj: Path for import object.

    :return: Imported object.

    """
    mod = importlib.import_module(path_to_import_obj.rsplit('.', 1)[0])
    return getattr(mod, path_to_import_obj.rsplit('.', 1)[1])


def analyze_method_params(func: Callable, auto_detect_predict_params: bool = True, ignore_self: bool = True):
    """
    Parse method and create dict with method parameters.

    :param Callable func: Func for parse arguments.
    :param bool auto_detect_predict_params: Auto detect predict params and set List to first param.
    :param bool ignore_self: Ignore first param with name self and cls.

    :return: Dict with parsing func arguments.
    :rtype: Dict[str, Dict[str, str]]

    def example(a, b = 100, *, c: float = 123):
        pass

    method_params = analyze_method_params(method)
    method_params
    [{'name': 'a', 'required': True},
     {'name': 'b', 'required': False, 'default': 100},
     {'name': 'c', 'type': float, 'required': False, 'default': 123}]

    """
    sign = inspect.signature(func)
    arg_spec = inspect.getfullargspec(func)
    result = []
    is_there_args = False
    #logger.info(f'Analyzing arguments in {func}.')
    #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    return result


def auto_search_binarization_type(loaded_file: LoadedFile) -> Optional[Type[BinarizationType]]:
    """
    Search binarizer by model raw binary data and model name.
    Search in only mlup binarizers from mlup.constants.BinarizationType.

    :param LoadedFile loaded_file: Model raw data information.

    :return: Found binarizer class if found or None if not found.
    :rtype: Optional[Type[BinarizationType]]

    """
    logger.info('Run auto search binarizer.')
    probabilities = []
    with TimeProfiler('Time to auto search binarizer:', log_level='info'):
        for binarizer_path in BinarizationType:
            try:
                binarization_class = get_class_by_path(binarizer_path.value)
            except (ModuleNotFoundError, AttributeError):
                continue

            probability = binarization_class.is_this_type(loaded_file=loaded_file)
            probabilities.append((probability, binarizer_path))
            logger.debug(f'Binarizer "{binarizer_path}" have probability {probability}.')

        high_prob = sorted(probabilities, key=lambda x: x[0])[-1]
        logger.debug(f'Hypothesis binarizer "{high_prob[1]}" with probaility {high_prob[0]}.')
        if high_prob[0] > 0.0:
            logger.info(f'Found binarizer "{high_prob[1]}" with probaility {high_prob[0]}.')
            return high_prob[1]

    logger.info('Binarizer not found.')
    return None
