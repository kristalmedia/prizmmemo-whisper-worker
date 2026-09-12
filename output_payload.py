from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def compact_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the transcript fields consumed by PrizmMemo.

    WhisperX segments can also contain word, character, and tensor-backed
    diagnostic fields. Sending those through the serverless result webhook is
    both unnecessary and vulnerable to JSON serialization/response-size
    failures after GPU work has already completed.
    """
    compact: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start = _finite_float(segment.get("start"), f"segments[{index}].start")
        end = _finite_float(segment.get("end"), f"segments[{index}].end")
        if start < 0 or end < start:
            raise ValueError(f"segments[{index}] has an invalid time range")
        text = segment.get("text")
        if not isinstance(text, str):
            raise ValueError(f"segments[{index}].text must be a string")
        speaker = segment.get("speaker")
        if speaker is not None and not isinstance(speaker, str):
            raise ValueError(f"segments[{index}].speaker must be a string or null")
        compact.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return compact


def normalize_speaker_embeddings(value: Any) -> dict[str, list[float]] | None:
    """Convert WhisperX 3.8 speaker vectors into strict, finite JSON values."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("speaker embeddings must be keyed by speaker label")

    normalized: dict[str, list[float]] = {}
    for speaker, vector in value.items():
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError(f"speaker embedding {speaker!s} must be a non-empty vector")
        normalized[str(speaker)] = [
            _finite_float(component, f"speaker_embeddings.{speaker}[{index}]")
            for index, component in enumerate(vector)
        ]
    return normalized

