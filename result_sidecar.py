"""Best-effort private result handoff when RunPod's status channel loses output."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

MAX_RESULT_BYTES = 32 * 1024 * 1024


def validate_result_put_url(value: Any, allowed_host_suffix: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("input.result_put_url must be an HTTPS R2 URL")
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or not allowed_host_suffix.startswith(".")
        or not host.endswith(allowed_host_suffix)
    ):
        raise ValueError("input.result_put_url must target the allowed R2 endpoint")
    return value


def put_result_sidecar(url: str, job_id: str, output: dict[str, Any]) -> bool:
    """Never log the signed URL, output, or provider response body."""
    if not job_id:
        return False
    import httpx
    try:
        body = json.dumps({"job_id": job_id, "output": output}, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        return False
    if len(body) > MAX_RESULT_BYTES:
        return False
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10, read=30, write=30, pool=10), follow_redirects=False) as client:
            response = client.put(url, content=body, headers={"Content-Type": "application/json"})
            return 200 <= response.status_code < 300
    except httpx.HTTPError:
        return False
