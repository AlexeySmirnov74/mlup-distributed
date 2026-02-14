import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from mlup.config import set_logging_settings, LOGGING_CONFIG

# IMPORTANT: set run mode BEFORE importing common_app
os.environ["MLUP_RUN_MODE"] = "worker"

from common_app import build_up  # noqa: E402


set_logging_settings(LOGGING_CONFIG, level=logging.INFO)
logger = logging.getLogger("mlup")

LOCAL_WORKER_PROCESSES = int(os.environ.get("LOCAL_WORKER_PROCESSES", 4))
IS_CHILD = os.environ.get("WORKER_CHILD", "0") == "1"

# When running inside Docker, do NOT spawn local subprocesses.
# You scale via: docker compose up --scale worker=4
IN_DOCKER = os.environ.get("IN_DOCKER", "0") == "1" or os.path.exists("/.dockerenv")


async def run_worker_forever() -> None:
    up = build_up()
    up.ml.load()
    up.web.load()

    # Architecture instance from queue_module
    arch = up.web._architecture_obj

    # Optional file logging for Windows local demo
    if os.name == "nt":
        os.makedirs("logs", exist_ok=True)
        pid = os.getpid()
        fh = logging.FileHandler(f"logs/worker_{pid}.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s:[%(asctime)s] - %(message)s")
        fh.setFormatter(formatter)
        logging.getLogger("mlup").addHandler(fh)

    # Start only Redis + queue worker loop (no uvicorn)
    await arch.run()

    logger.info("Queue worker is running (no HTTP server). Waiting forever...")
    await asyncio.Event().wait()


def spawn_children(count: int) -> None:
    """
    Spawn N worker processes on the same machine.
    Useful for Windows demo (one terminal -> N workers).
    Do NOT use inside Docker (scale by docker compose).
    """
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())

    for i in range(count):
        env = os.environ.copy()
        env["WORKER_CHILD"] = "1"
        # Nice worker identity for logs/redis (optional)
        # env["MLUP_NODE_ID"] = f"{env.get('MLUP_NODE_ID', 'node')}-w{i+1}"

        logger.info("Spawning worker process %d/%d ...", i + 1, count)
        subprocess.Popen([python_exe, script_path], env=env)


if __name__ == "__main__":
    logger.info("Starting workers (MLUP_RUN_MODE=worker). Queue workers=%s", LOCAL_WORKER_PROCESSES)

    # Docker / Linux: one worker process per container (no local subprocess fan-out)
    if IN_DOCKER:
        asyncio.run(run_worker_forever())
        raise SystemExit(0)

    # Windows local demo: parent spawns N children and exits
    if not IS_CHILD:
        spawn_children(LOCAL_WORKER_PROCESSES)
        sys.exit(0)

    asyncio.run(run_worker_forever())
