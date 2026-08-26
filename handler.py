from __future__ import annotations

import gc
import importlib.metadata
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import runpod
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


ENGINE_VERSION = f"prizmmemo-runpod/1.0.0 whisperx/{importlib.metadata.version('whisperx')}"
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2,4})?$")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


MAX_AUDIO_BYTES = _positive_int_env("MAX_AUDIO_BYTES", 512 * 1024 * 1024)
DOWNLOAD_TIMEOUT_SEC = _positive_int_env("DOWNLOAD_TIMEOUT_SEC", 600)
ALLOWED_AUDIO_HOST_SUFFIX = os.getenv(
    "ALLOWED_AUDIO_HOST_SUFFIX", ".r2.cloudflarestorage.com"
).strip().lower()


def _progress(job: dict[str, Any], stage: str) -> None:
    runpod.serverless.progress_update(job, stage)


def _validate_audio_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("input.audio_url is required")
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
        raise ValueError("input.audio_url must be an HTTPS URL without credentials")
    if not ALLOWED_AUDIO_HOST_SUFFIX.startswith("."):
        raise RuntimeError("ALLOWED_AUDIO_HOST_SUFFIX must start with a dot")
    if not hostname.endswith(ALLOWED_AUDIO_HOST_SUFFIX):
        raise ValueError("input.audio_url host is not an allowed R2 endpoint")
    return value


def _optional_language(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("input.language must be a language code")
    language = value.strip().lower()
    if not LANGUAGE_RE.fullmatch(language):
        raise ValueError("input.language must be a language code")
    return language


def _optional_speaker_count(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
        raise ValueError(f"input.{name} must be an integer from 1 to 50")
    return value


def _validate_input(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input")
    if not isinstance(payload, dict):
        raise ValueError("input must be an object")

    meeting_id = payload.get("meeting_id")
    if not isinstance(meeting_id, str) or not meeting_id.strip() or len(meeting_id) > 128:
        raise ValueError("input.meeting_id must be a non-empty string up to 128 characters")

    diarize = payload.get("diarize", True)
    if not isinstance(diarize, bool):
        raise ValueError("input.diarize must be a boolean")

    min_speakers = _optional_speaker_count("min_speakers", payload.get("min_speakers"))
    max_speakers = _optional_speaker_count("max_speakers", payload.get("max_speakers"))
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError("input.min_speakers cannot exceed input.max_speakers")

    return {
        "meeting_id": meeting_id.strip(),
        "audio_url": _validate_audio_url(payload.get("audio_url")),
        "language": _optional_language(payload.get("language")),
        "diarize": diarize,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }


def _download_audio(audio_url: str) -> Path:
    suffix = Path(urlparse(audio_url).path).suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".audio"

    handle = tempfile.NamedTemporaryFile(prefix="prizmmemo-", suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()

    timeout = httpx.Timeout(connect=20, read=DOWNLOAD_TIMEOUT_SEC, write=30, pool=30)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("GET", audio_url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) > MAX_AUDIO_BYTES:
                    raise ValueError("audio object exceeds MAX_AUDIO_BYTES")

                downloaded = 0
                with path.open("wb") as output:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > MAX_AUDIO_BYTES:
                            raise ValueError("audio object exceeds MAX_AUDIO_BYTES")
                        output.write(chunk)
        if path.stat().st_size == 0:
            raise ValueError("audio object is empty")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class WhisperXEngine:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is false")

        self.device = "cuda"
        self.model_name = os.getenv("WHISPER_MODEL", "large-v3").strip()
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16").strip()
        self.batch_size = _positive_int_env("WHISPER_BATCH_SIZE", 16)
        self.cache_dir = os.getenv("HF_HOME", "/root/.cache/huggingface")
        hf_token = os.getenv("HF_TOKEN", "").strip()
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for pyannote diarization")

        print(f"[startup] loading WhisperX model={self.model_name} compute={self.compute_type}")
        self.asr_model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            download_root=self.cache_dir,
            vad_method="silero",
        )
        print("[startup] loading pyannote speaker-diarization-community-1")
        self.diarizer = DiarizationPipeline(token=hf_token, device=self.device, cache_dir=self.cache_dir)
        print(f"[startup] ready engine={ENGINE_VERSION} gpu={torch.cuda.get_device_name(0)}")

    def transcribe(self, job: dict[str, Any], request: dict[str, Any], audio_path: Path) -> dict[str, Any]:
        audio = whisperx.load_audio(str(audio_path))

        _progress(job, "transcribing")
        result = self.asr_model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=request["language"],
        )
        language = result["language"]

        alignment_applied = False
        _progress(job, "aligning")
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
                model_dir=self.cache_dir,
            )
            try:
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                alignment_applied = True
            finally:
                del align_model
                gc.collect()
                torch.cuda.empty_cache()
        except ValueError as error:
            print(f"[job {job.get('id', 'unknown')}] alignment skipped for language={language}: {error}")

        speaker_embeddings = None
        if request["diarize"]:
            _progress(job, "diarizing")
            diarization, speaker_embeddings = self.diarizer(
                audio,
                min_speakers=request["min_speakers"],
                max_speakers=request["max_speakers"],
                return_embeddings=True,
            )
            result = whisperx.assign_word_speakers(
                diarization,
                result,
                speaker_embeddings=speaker_embeddings,
                fill_nearest=True,
            )

        return _json_safe(
            {
                "meeting_id": request["meeting_id"],
                "segments": result["segments"],
                "detected_language": language,
                "alignment_applied": alignment_applied,
                "speaker_embeddings": speaker_embeddings,
                "engine_version": ENGINE_VERSION,
            }
        )


ENGINE = WhisperXEngine()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    request = _validate_input(job)
    audio_path: Path | None = None
    try:
        _progress(job, "downloading")
        audio_path = _download_audio(request["audio_url"])
        result = ENGINE.transcribe(job, request, audio_path)
        _progress(job, "complete")
        return result
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
