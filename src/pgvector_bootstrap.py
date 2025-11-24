"""Utility helpers to ensure a local pgvector instance is available.

This module mirrors the logic used in the JavaScript dev runner so that
starting the FastAPI/uvicorn app directly (e.g. via ``uv run``) can still
bootstrap the required PostgreSQL container automatically.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional


LOGGER = logging.getLogger("pgvector_bootstrap")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _is_local_host(host: str) -> bool:
    normalized = (host or "").strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"} or normalized.startswith("0.0.0.0")


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=True,
    )


def _run_shell(command: str) -> subprocess.CompletedProcess[str]:
    shell = os.environ.get("SHELL", "/bin/bash")
    return subprocess.run(
        [shell, "-lc", command],
        check=True,
        text=True,
        capture_output=True,
    )


def _docker_exec(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return _run(["docker", *args])
    except subprocess.CalledProcessError as err:
        message = f"docker {' '.join(shlex.quote(arg) for arg in args)} failed: {err.stderr or err.stdout or err}"
        raise RuntimeError(message) from err


def _docker_ready() -> bool:
    try:
        _docker_exec(["info"])
        return True
    except RuntimeError as err:
        text = str(err).lower()
        if "cannot connect" in text or "is the docker daemon running" in text:
            return False
        raise


def _detect_docker_desktop_start_cmd() -> Optional[str]:
    if sys.platform == "darwin" and os.path.exists("/Applications/Docker.app"):
        return "open -ga Docker"
    return None


def _docker_start_strategies() -> List[tuple[str, Callable[[], None]]]:
    strategies: List[tuple[str, Callable[[], None]]] = []
    start_cmd = os.environ.get("RAG_AUTODEV_DOCKER_START_CMD")
    if start_cmd:
        strategies.append((start_cmd, lambda: _run_shell(start_cmd)))
        return strategies

    if _env_flag("RAG_AUTODEV_AUTO_COLIMA", True) and shutil.which("colima"):
        strategies.append(("colima start", lambda: _run(["colima", "start"], check=True)))

    desktop_cmd = _detect_docker_desktop_start_cmd()
    if desktop_cmd:
        strategies.append((desktop_cmd, lambda: _run_shell(desktop_cmd)))

    return strategies


def _ensure_docker_daemon() -> None:
    if _docker_ready():
        return

    wait_attempts = int(os.environ.get("RAG_AUTODEV_DOCKER_WAIT_ATTEMPTS", "30"))
    wait_delay = float(os.environ.get("RAG_AUTODEV_DOCKER_WAIT_DELAY_MS", "2000")) / 1000.0

    strategies = _docker_start_strategies()
    if not strategies:
        raise RuntimeError(
            "Docker daemon is not running. Start Docker Desktop/Colima manually, set RAG_AUTODEV_DOCKER_START_CMD, or export RAG_DB_AUTO_START=0."
        )

    started_label: Optional[str] = None
    last_err: Optional[Exception] = None
    for label, runner in strategies:
        try:
            LOGGER.info("🐋 Attempting to start Docker daemon via: %s", label)
            runner()
            started_label = label
            break
        except Exception as err:  # pragma: no cover - depends on local setup
            LOGGER.warning("⚠️  Failed to start Docker via %s: %s", label, getattr(err, "stderr", None) or err)
            last_err = err

    if not started_label:
        raise RuntimeError(
            "Unable to automatically start Docker. Start it manually or configure RAG_AUTODEV_DOCKER_START_CMD." +
            (f" Last error: {last_err}" if last_err else "")
        )

    for attempt in range(1, wait_attempts + 1):
        if _docker_ready():
            LOGGER.info("✅ Docker daemon became ready (attempt %s)", attempt)
            return
        time.sleep(wait_delay)

    raise RuntimeError(f"Docker daemon did not become ready after running {started_label}")


def _ensure_docker_cli() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI not found in PATH. Install Docker Desktop or Colima.")


def _inspect_container(name: str) -> Optional[Dict[str, Any]]:
    try:
        result = _docker_exec(["container", "inspect", name])
        parsed = json.loads(result.stdout)
        if isinstance(parsed, list) and parsed:
            return parsed[0]
        return None
    except RuntimeError:
        return None


def _ensure_container(host_port: int) -> None:
    container = os.environ.get("RAG_DB_CONTAINER", "postgres-pgvector")
    image = os.environ.get("RAG_DB_IMAGE", "ankane/pgvector:latest")
    db_user = os.environ.get("RAG_DB_USER", "rag")
    db_pass = os.environ.get("RAG_DB_PASSWORD", "ragpass")
    db_name = os.environ.get("RAG_DB_NAME", "ragdb")

    info = _inspect_container(container)
    desired = str(host_port)
    if info is not None:
        mapped = info.get("NetworkSettings", {}).get("Ports", {}).get("5432/tcp")
        mapped_port = mapped[0]["HostPort"] if isinstance(mapped, list) and mapped else None
        if mapped_port != desired:
            LOGGER.info(
                "♻️  Recreating pgvector container %s to use host port %s (was %s)",
                container,
                desired,
                mapped_port or "unmapped",
            )
            _docker_exec(["container", "rm", "-f", container])
            info = None

    if info is None:
        LOGGER.info("🐘 Creating pgvector container %s", container)
        _docker_exec(
            [
                "run",
                "-d",
                "--name",
                container,
                "-e",
                f"POSTGRES_USER={db_user}",
                "-e",
                f"POSTGRES_PASSWORD={db_pass}",
                "-e",
                f"POSTGRES_DB={db_name}",
                "-p",
                f"{host_port}:5432",
                image,
            ]
        )
    else:
        state = info.get("State", {}).get("Status") or "unknown"
        if state != "running":
            LOGGER.info("🔄 Starting pgvector container %s (status=%s)", container, state)
            _docker_exec(["container", "start", container])
        else:
            LOGGER.info("✅ pgvector container %s already running", container)


def ensure_pgvector_available() -> None:
    """Ensure pgvector is reachable, starting Docker if necessary."""

    if not _env_flag("RAG_DB_AUTO_START", True):
        LOGGER.info("⚠️  Skipping pgvector auto-start (RAG_DB_AUTO_START=0)")
        return

    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    env_port = os.environ.get("POSTGRES_PORT")
    port = int(env_port or os.environ.get("RAG_DB_DEFAULT_PORT", "5434"))
    should_export_port = env_port is None

    if not _is_local_host(host):
        LOGGER.debug("pgvector auto-start skipped because host %s is not local", host)
        return

    def _remember_port() -> None:
        if should_export_port:
            os.environ["POSTGRES_PORT"] = str(port)
            LOGGER.info("📌 POSTGRES_PORT not set; defaulting to %s for local pgvector", port)

    if _is_port_open(host, port):
        _remember_port()
        LOGGER.debug("pgvector already reachable at %s:%s", host, port)
        return

    LOGGER.info("🚚 pgvector not reachable at %s:%s — bootstrapping Docker", host, port)
    _ensure_docker_cli()
    _ensure_docker_daemon()
    _ensure_container(port)

    wait_attempts = int(os.environ.get("RAG_DB_WAIT_ATTEMPTS", "30"))
    wait_delay = float(os.environ.get("RAG_DB_WAIT_DELAY_MS", "1000")) / 1000.0
    for attempt in range(1, wait_attempts + 1):
        if _is_port_open(host, port):
            _remember_port()
            LOGGER.info("✅ pgvector ready on %s:%s", host, port)
            break
        time.sleep(wait_delay)
    else:
        raise RuntimeError(f"pgvector did not become ready on {host}:{port}")

    init_sql = os.environ.get("RAG_DB_INIT_SQL", "CREATE EXTENSION IF NOT EXISTS vector;")
    if not init_sql.strip():
        return

    attempts = int(os.environ.get("RAG_DB_INIT_ATTEMPTS", "5"))
    delay = float(os.environ.get("RAG_DB_INIT_DELAY_MS", "1000")) / 1000.0
    container = os.environ.get("RAG_DB_CONTAINER", "postgres-pgvector")
    user = os.environ.get("RAG_DB_USER", "rag")
    database = os.environ.get("RAG_DB_NAME", "ragdb")
    for attempt in range(1, attempts + 1):
        try:
            _docker_exec(
                [
                    "exec",
                    container,
                    "psql",
                    "-U",
                    user,
                    "-d",
                    database,
                    "-h",
                    "localhost",
                    "-c",
                    init_sql,
                ]
            )
            LOGGER.info("🧩 Ensured pgvector extension via init SQL")
            break
        except RuntimeError as err:
            if attempt == attempts:
                LOGGER.warning("⚠️  Failed to run init SQL '%s': %s", init_sql, err)
            else:
                time.sleep(delay)
