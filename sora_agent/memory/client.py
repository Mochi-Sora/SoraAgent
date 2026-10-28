"""Minimal REST client for the CortexDB sidecar.

The sidecar is the only process that touches the active database. This client
speaks JSON over loopback so the memory subsystem needs no gRPC/protobuf
runtime dependency.
"""

from __future__ import annotations

import http.client
import json
import threading
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlencode, urlsplit


class CortexDBError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"CortexDB HTTP {status}: {message}")
        self.status = status
        self.message = message


class CortexDBUnavailable(CortexDBError):
    def __init__(self, message: str) -> None:
        super().__init__(0, message)


class MemoryBackend(Protocol):
    def health(self) -> bool: ...
    def save_memory(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]: ...
    def list_memories(self, limit: int = 500) -> List[Dict[str, Any]]: ...
    def update_memory(self, memory_id: str, **fields: Any) -> Dict[str, Any]: ...
    def delete_memory(self, memory_id: str) -> None: ...
    def recall(self, query: str, top_k: int = 8, **options: Any) -> Dict[str, Any]: ...
    def rules_apply(self, **payload: Any) -> Dict[str, Any]: ...
    def sweep_expired(self) -> Dict[str, Any]: ...


class CortexDBClient:
    def __init__(
        self,
        http_addr: str = "127.0.0.1:47822",
        token: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        parsed = urlsplit(http_addr if "://" in http_addr else "http://" + http_addr)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 47822
        self.token = token or ""
        self.timeout = float(timeout)
        self._lock = threading.Lock()

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if params:
            query = urlencode({k: v for k, v in params.items() if v is not None})
            if query:
                path = path + ("&" if "?" in path else "?") + query
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        with self._lock:
            conn = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout
            )
            try:
                conn.request(method, path, body=body, headers=headers)
                response = conn.getresponse()
                raw = response.read()
            except OSError as exc:
                raise CortexDBUnavailable(str(exc)) from exc
            finally:
                conn.close()
        text = raw.decode("utf-8", errors="replace")
        if response.status < 200 or response.status >= 300:
            message = text
            try:
                decoded = json.loads(text)
                if isinstance(decoded, dict):
                    message = str(decoded.get("error") or decoded.get("message") or text)
            except ValueError:
                pass
            raise CortexDBError(response.status, message)
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError as exc:
            raise CortexDBError(response.status, "invalid JSON response") from exc

    def health(self) -> bool:
        try:
            payload = self._request("GET", "/v1/health")
        except CortexDBError:
            return False
        return bool(payload.get("ok", True)) if isinstance(payload, dict) else True

    def info(self) -> Dict[str, Any]:
        payload = self._request("GET", "/v1/info")
        return payload if isinstance(payload, dict) else {}

    def save_memory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request("POST", "/v1/memory", payload)
        return response if isinstance(response, dict) else {}

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        response = self._request(
            "GET", "/v1/memory", params={"memory_id": memory_id}
        )
        if isinstance(response, dict):
            memory = response.get("memory")
            return memory if isinstance(memory, dict) else None
        return None

    def list_memories(self, limit: int = 500) -> List[Dict[str, Any]]:
        memories: List[Dict[str, Any]] = []
        cursor = ""
        page_limit = max(1, min(int(limit), 500))
        for _ in range(1000):
            payload: Dict[str, Any] = {"limit": page_limit}
            if cursor:
                payload["cursor"] = cursor
            response = self._request(
                "POST", "/v1/tools/memory_list_all", payload
            )
            if not isinstance(response, dict):
                break
            page = response.get("memories")
            if isinstance(page, list):
                memories.extend(item for item in page if isinstance(item, dict))
            if not response.get("truncated"):
                break
            cursor = str(response.get("next_cursor") or "")
            if not cursor:
                break
            if len(memories) >= limit:
                break
        return memories[:limit]

    def update_memory(self, memory_id: str, **fields: Any) -> Dict[str, Any]:
        payload = dict(fields)
        payload["memory_id"] = memory_id
        response = self._request("POST", "/v1/tools/memory_update", payload)
        return response if isinstance(response, dict) else {}

    def delete_memory(self, memory_id: str) -> None:
        self._request(
            "DELETE", "/v1/memory", params={"memory_id": memory_id}
        )

    def recall(self, query: str, top_k: int = 8, **options: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": query,
            "top_k_memories": top_k,
            "retrieval_mode": options.pop("retrieval_mode", "lexical"),
            "disable_knowledge": options.pop("disable_knowledge", True),
            "disable_graph": options.pop("disable_graph", True),
        }
        payload.update(options)
        response = self._request("POST", "/v1/recall", payload)
        return response if isinstance(response, dict) else {}

    def search_memory(
        self, query: str, top_k: int = 8, **options: Any
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        payload.update(options)
        response = self._request("POST", "/v1/memory/search", payload)
        return response if isinstance(response, dict) else {}

    def rules_apply(self, **payload: Any) -> Dict[str, Any]:
        response = self._request("POST", "/v1/tools/rules_apply", payload)
        return response if isinstance(response, dict) else {}

    def sweep_expired(self) -> Dict[str, Any]:
        response = self._request("POST", "/v1/tools/sweep_expired", {})
        return response if isinstance(response, dict) else {}

    def call_tool(self, name: str, payload: Dict[str, Any]) -> Any:
        return self._request("POST", "/v1/tools/" + name, payload)
