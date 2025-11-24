from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

FILE_STORAGE_ROOT = Path(os.environ.get("FILE_STORAGE_ROOT", "data/files")).resolve()
FILE_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = Path(
    os.environ.get("FILE_REGISTRY_PATH", FILE_STORAGE_ROOT / "registry.json")
).resolve()
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

_registry_lock = threading.Lock()


def _load_registry() -> Dict[str, Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _save_registry(data: Dict[str, Dict[str, Any]]) -> None:
    tmp_path = REGISTRY_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(REGISTRY_PATH)


def _ensure_within_root(path: Path) -> Path:
    try:
        path.relative_to(FILE_STORAGE_ROOT)
        return path
    except ValueError:
        raise ValueError("Invalid storage path")


def store_bytes(payload: bytes, *, suffix: Optional[str] = None) -> Path:
    file_id = uuid4().hex
    subdir = Path(file_id[:2]) / file_id[2:4]
    target_dir = FILE_STORAGE_ROOT / "objects" / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = file_id + (suffix or "")
    destination = target_dir / filename
    destination.write_bytes(payload)
    return _ensure_within_root(destination)


def create_metadata(
    *,
    tenant: str,
    user_id: str,
    scope: str,
    folder_path: str,
    notebook_id: Optional[str],
    original_name: str,
    mime_type: str,
    size_bytes: int,
    storage_path: Path,
) -> Dict[str, Any]:
    file_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    folder = folder_path if folder_path.startswith("/") else f"/{folder_path}"
    folder = folder.replace("//", "/") or "/"
    return {
        "id": file_id,
        "tenant": tenant,
        "user_id": user_id,
        "scope": scope,
        "folder_path": folder,
        "notebook_id": notebook_id or "",
        "original_name": original_name,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "storage_path": str(storage_path),
        "created_at": now,
        "updated_at": now,
    }


def register_file(metadata: Dict[str, Any]) -> Dict[str, Any]:
    with _registry_lock:
        registry = _load_registry()
        registry[metadata["id"]] = metadata
        _save_registry(registry)
    return metadata


def get_file(file_id: str) -> Optional[Dict[str, Any]]:
    registry = _load_registry()
    record = registry.get(file_id)
    if not record:
        return None
    return dict(record)


def list_files(*, tenant: Optional[str] = None, user_id: Optional[str] = None, folder_path: Optional[str] = None) -> List[Dict[str, Any]]:
    registry = _load_registry()
    records = list(registry.values())
    if tenant:
        records = [item for item in records if item.get("tenant") == tenant]
    if user_id:
        records = [item for item in records if item.get("user_id") == user_id]
    if folder_path and folder_path.strip() not in {"", "/"}:
        folder = folder_path if folder_path.startswith("/") else f"/{folder_path}"
        folder = folder.replace("//", "/") or "/"
        records = [item for item in records if item.get("folder_path") == folder]
    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return records


def list_folders(*, tenant: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    registry = _load_registry()
    scope_map: Dict[str, set[str]] = {}
    counts: Dict[str, int] = {}
    for item in registry.values():
        if tenant and item.get("tenant") != tenant:
            continue
        if user_id and item.get("user_id") != user_id:
            continue
        folder = item.get("folder_path") or "/"
        if not folder.startswith("/"):
            folder = f"/{folder}"
        folder = folder.replace("//", "/") or "/"
        scope = (item.get("scope") or "personal").lower()
        scope_map.setdefault(folder, set()).add(scope)
        counts[folder] = counts.get(folder, 0) + 1

    if not scope_map:
        scope_map["/"] = {"personal"}
        counts["/"] = 0

    result = []
    for path in sorted(scope_map.keys()):
        scopes = scope_map[path]
        scope_value = "mixed" if len(scopes) > 1 else next(iter(scopes))
        result.append({"path": path, "scope": scope_value, "count": counts.get(path, 0)})
    return result


def update_file(file_id: str, **changes: Any) -> Optional[Dict[str, Any]]:
    with _registry_lock:
        registry = _load_registry()
        record = registry.get(file_id)
        if not record:
            return None
        now = datetime.now(timezone.utc).isoformat()
        record.update(changes)
        record["updated_at"] = now
        _save_registry(registry)
        return dict(record)


def delete_file(file_id: str) -> bool:
    with _registry_lock:
        registry = _load_registry()
        record = registry.pop(file_id, None)
        if record:
            _save_registry(registry)
    if not record:
        return False
    try:
        path = _ensure_within_root(Path(record.get("storage_path", "")))
        if path.exists():
            path.unlink()
            parent = path.parent
            while parent != FILE_STORAGE_ROOT and parent != parent.parent:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
                parent = parent.parent
    except Exception:
        pass
    return True
