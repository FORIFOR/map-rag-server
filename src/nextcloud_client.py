from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import httpx


class NextcloudClientError(RuntimeError):
    """Base error for Nextcloud client operations."""


class NextcloudConfigError(NextcloudClientError):
    """Raised when mandatory environment variables are missing."""


class NextcloudRequestError(NextcloudClientError):
    """Raised when Nextcloud WebDAV operations fail."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_nextcloud_path(path: str) -> str:
    """Normalize Nextcloud paths to always start with a single slash."""
    if not path:
        return "/"
    segments = [segment for segment in path.strip().split("/") if segment]
    if not segments:
        return "/"
    return "/" + "/".join(segments)


def _encode_path(path: str) -> str:
    segments = [segment for segment in path.strip("/").split("/") if segment]
    return "/".join(quote(segment, safe="") for segment in segments)


@dataclass(frozen=True)
class NextcloudSettings:
    base_url: str
    username: str
    password: str
    timeout: float = 60.0
    verify_tls: bool = True

    @classmethod
    def from_env(cls) -> "NextcloudSettings":
        base = os.getenv("NEXTCLOUD_WEBDAV_BASE_URL")
        username = os.getenv("NEXTCLOUD_USERNAME") or os.getenv("NEXTCLOUD_WEBDAV_USERNAME")
        password = os.getenv("NEXTCLOUD_APP_PASSWORD") or os.getenv("NEXTCLOUD_WEBDAV_PASSWORD")
        if not base or not username or not password:
            raise NextcloudConfigError(
                "NEXTCLOUD_WEBDAV_BASE_URL/NEXTCLOUD_USERNAME/NEXTCLOUD_APP_PASSWORD が未設定です"
            )
        timeout_raw = os.getenv("NEXTCLOUD_TIMEOUT_SEC")
        try:
            timeout = max(5.0, float(timeout_raw)) if timeout_raw else 60.0
        except (TypeError, ValueError):
            timeout = 60.0
        verify = _env_bool("NEXTCLOUD_VERIFY_TLS", True)
        return cls(base.rstrip("/"), username.strip(), password.strip(), timeout, verify)


class NextcloudClient:
    def __init__(self, settings: NextcloudSettings):
        self.settings = settings
        self._auth = (settings.username, settings.password)

    @classmethod
    def from_env(cls) -> "NextcloudClient":
        return cls(NextcloudSettings.from_env())

    def build_url(self, path: str) -> str:
        normalized = normalize_nextcloud_path(path)
        encoded = _encode_path(normalized)
        if not encoded:
            return f"{self.settings.base_url}/"
        return f"{self.settings.base_url}/{encoded}"

    async def download_file(self, path: str) -> bytes:
        url = self.build_url(path)
        async with httpx.AsyncClient(
            auth=self._auth,
            timeout=self.settings.timeout,
            verify=self.settings.verify_tls,
        ) as client:
            try:
                response = await client.get(url)
            except httpx.RequestError as exc:
                raise NextcloudRequestError(-1, f"Nextcloud 接続に失敗しました: {exc}") from exc

        if response.status_code == 404:
            raise NextcloudRequestError(404, f"Nextcloud にファイルがありません: {path}")
        if response.status_code >= 400:
            text = response.text.strip()
            message = text or response.reason_phrase or f"HTTP {response.status_code}"
            raise NextcloudRequestError(response.status_code, message)
        return response.content


_CLIENT: Optional[NextcloudClient] = None


def get_nextcloud_client() -> Optional[NextcloudClient]:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        _CLIENT = NextcloudClient.from_env()
    except NextcloudConfigError:
        _CLIENT = None
    return _CLIENT
