from __future__ import annotations

import gc
import importlib.metadata
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
import runpod
import torch
import whisperx
from faster_whisper import BatchedInferencePipeline
from whisperx.diarize import DiarizationPipeline

from language_selection import score_language_candidates, select_language_candidate
from log_safety import suppress_signed_request_logging
from speaker_segments import split_segments_by_word_speaker


suppress_signed_request_logging()


ENGINE_VERSION = (
    f"prizmmemo-runpod/1.3.2 "
    f"whisperx/{importlib.metadata.version('whisperx')} "
    f"faster-whisper/{importlib.metadata.version('faster-whisper')}"
)
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2,4})?$")
CONTIGUOUS_SPEECH_GAP_SEC = 0.15


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
MULTILINGUAL_CHUNK_LENGTH_SEC = _positive_int_env("MULTILINGUAL_CHUNK_LENGTH_SEC", 8)
if not 4 <= MULTILINGUAL_CHUNK_LENGTH_SEC <= 15:
    raise RuntimeError("MULTILINGUAL_CHUNK_LENGTH_SEC must be from 4 to 15")
ALLOWED_AUDIO_HOST_SUFFIX = os.getenv(
    "ALLOWED_AUDIO_HOST_SUFFIX", ".r2.cloudflarestorage.com"
).strip().lower()
RUNPOD_MODEL_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")


def _resolve_cached_snapshot(model_id: str) -> Path | None:
    if "/" not in model_id:
        return None
    organization, name = model_id.split("/", 1)
    model_root = RUNPOD_MODEL_CACHE_ROOT / f"models--{organization}--{name}"
    snapshots_root = model_root / "snapshots"
    ref = model_root / "refs" / "main"
    if ref.is_file():
        candidate = snapshots_root / ref.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate
    if snapshots_root.is_dir():
        snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
        if snapshots:
            return snapshots[0]
    return None


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


def _expected_languages(value: Any) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError("input.languages must contain 1-3 language codes")
    languages: list[str] = []
    for item in value:
        if not isinstance(item, str) or not LANGUAGE_RE.fullmatch(item.strip().lower()):
            raise ValueError("input.languages must contain only language codes")
        languages.append(item.strip().lower())
    if len(set(languages)) != len(languages):
        raise ValueError("input.languages must not contain duplicates")
    return languages


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
        "languages": _expected_languages(payload.get("languages")),
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


class ExpectedLanguageBatchedInferencePipeline(BatchedInferencePipeline):
    def __init__(self, model: Any, expected_languages: list[str]) -> None:
        super().__init__(model)
        self.expected_languages = tuple(expected_languages)
        self.language_chunks: list[dict[str, Any]] = []
        self._batch_choices: list[dict[str, Any]] = []
        self._current_chunks_metadata: list[dict[str, Any]] = []
        self._previous_language: str | None = None
        self._previous_speech_end_sec: float | None = None

    def forward(
        self,
        features: np.ndarray,
        tokenizer: Any,
        chunks_metadata: list[dict[str, Any]],
        options: Any,
    ) -> list[list[dict[str, Any]]]:
        self._current_chunks_metadata = chunks_metadata
        try:
            outputs = super().forward(features, tokenizer, chunks_metadata, options)
        finally:
            self._current_chunks_metadata = []
        if len(self._batch_choices) != len(chunks_metadata):
            raise RuntimeError("language candidate choices do not match speech chunks")
        for metadata, choice in zip(chunks_metadata, self._batch_choices):
            self.language_chunks.append(
                {
                    "speech_offset_sec": float(metadata["offset"]),
                    "duration_sec": float(metadata["duration"]),
                    **choice,
                }
            )
        return outputs

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Any,
        options: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens=(
                tokenizer.encode(options.initial_prompt)
                if options.initial_prompt is not None
                else []
            ),
            without_timestamps=options.without_timestamps,
            hotwords=options.hotwords,
        )
        if options.max_new_tokens is not None:
            max_length = len(prompt) + options.max_new_tokens
        else:
            max_length = self.model.max_length
        if max_length > self.model.max_length:
            raise ValueError(
                f"prompt plus max_new_tokens exceeds model max_length={self.model.max_length}"
            )

        language_token_index = prompt.index(tokenizer.language)
        language_token_ids: list[int] = []
        for language in self.expected_languages:
            token_id = tokenizer.tokenizer.token_to_id(f"<|{language}|>")
            if token_id is None:
                raise ValueError(f"unsupported expected language: {language}")
            language_token_ids.append(token_id)

        candidate_count = len(self.expected_languages)
        expanded_features = np.repeat(features, candidate_count, axis=0)
        encoder_output = self.model.encode(expanded_features)
        prompts: list[list[int]] = []
        for _ in range(features.shape[0]):
            for language_token_id in language_token_ids:
                candidate_prompt = prompt.copy()
                candidate_prompt[language_token_index] = language_token_id
                prompts.append(candidate_prompt)

        results = self.model.model.generate(
            encoder_output,
            prompts,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
            return_scores=True,
            return_no_speech_prob=True,
            sampling_temperature=options.temperatures[0],
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
        )
        detections = self.model.model.detect_language(encoder_output)

        candidates: list[dict[str, Any]] = []
        for generation in results:
            sequence_length = len(generation.sequences_ids[0])
            cumulative_log_probability = generation.scores[0] * (
                sequence_length**options.length_penalty
            )
            candidates.append(
                {
                    "avg_logprob": cumulative_log_probability / (sequence_length + 1),
                    "no_speech_prob": generation.no_speech_prob,
                    "tokens": generation.sequences_ids[0],
                }
            )

        selected_outputs: list[dict[str, Any]] = []
        self._batch_choices = []
        for chunk_index in range(features.shape[0]):
            start = chunk_index * candidate_count
            end = start + candidate_count
            chunk_candidates = candidates[start:end]
            detection_probabilities = {
                token[2:-2]: float(probability)
                for token, probability in detections[start]
            }
            expected_probabilities = [
                detection_probabilities.get(language, 1e-8)
                for language in self.expected_languages
            ]
            average_log_probabilities = [
                float(candidate["avg_logprob"]) for candidate in chunk_candidates
            ]
            metadata = self._current_chunks_metadata[chunk_index]
            speech_offset_sec = float(metadata["offset"])
            speech_end_sec = speech_offset_sec + float(metadata["duration"])
            gap_from_previous_sec = (
                None
                if self._previous_speech_end_sec is None
                else speech_offset_sec - self._previous_speech_end_sec
            )
            continuous_with_previous = (
                gap_from_previous_sec is not None
                and -0.05 <= gap_from_previous_sec <= CONTIGUOUS_SPEECH_GAP_SEC
            )
            candidate_scores = score_language_candidates(
                self.expected_languages,
                average_log_probabilities,
                expected_probabilities,
                primary_language=self.expected_languages[0],
                previous_language=self._previous_language,
                continuous_with_previous=continuous_with_previous,
            )
            selected_index = select_language_candidate(
                self.expected_languages,
                average_log_probabilities,
                expected_probabilities,
                primary_language=self.expected_languages[0],
                previous_language=self._previous_language,
                continuous_with_previous=continuous_with_previous,
            )
            selected_outputs.append(chunk_candidates[selected_index])
            raw_selected_index = max(
                range(candidate_count),
                key=lambda index: (candidate_scores[index]["base_score"], -index),
            )
            candidate_diagnostics = [
                {
                    "language": language,
                    "asr_avg_logprob": average_log_probabilities[index],
                    "detection_probability": expected_probabilities[index],
                    **candidate_scores[index],
                }
                for index, language in enumerate(self.expected_languages)
            ]
            self._batch_choices.append(
                {
                    "language": self.expected_languages[selected_index],
                    "asr_avg_logprob": average_log_probabilities[selected_index],
                    "detection_probability": expected_probabilities[selected_index],
                    "selection_reason": (
                        "adjusted_ambiguous_choice"
                        if selected_index != raw_selected_index
                        else "best_base_score"
                    ),
                    "continuous_with_previous": continuous_with_previous,
                    "gap_from_previous_sec": gap_from_previous_sec,
                    "candidates": candidate_diagnostics,
                }
            )
            print(
                f"[language-choice] offset={speech_offset_sec:.2f}s "
                f"duration={float(metadata['duration']):.2f}s "
                f"selected={self.expected_languages[selected_index]} "
                f"reason={self._batch_choices[-1]['selection_reason']} "
                f"continuous={continuous_with_previous} candidates="
                + ",".join(
                    f"{candidate['language']}:asr={candidate['asr_avg_logprob']:.4f}:"
                    f"detect={candidate['detection_probability']:.4f}:"
                    f"score={candidate['selection_score']:.4f}"
                    for candidate in candidate_diagnostics
                ),
                flush=True,
            )
            self._previous_language = self.expected_languages[selected_index]
            self._previous_speech_end_sec = speech_end_sec

        selected_encoder_output = (
            self.model.encode(features) if options.word_timestamps else encoder_output
        )
        return selected_encoder_output, selected_outputs


class WhisperXEngine:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is false")

        self.device = "cuda"
        self.model_id = os.getenv(
            "WHISPER_MODEL_ID", "Systran/faster-whisper-large-v3"
        ).strip()
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16").strip()
        self.batch_size = _positive_int_env("WHISPER_BATCH_SIZE", 16)
        self.multilingual_chunk_length_sec = MULTILINGUAL_CHUNK_LENGTH_SEC
        self.hf_cache_dir = os.getenv("HF_HOME", "/root/.cache/huggingface")
        cached_model = _resolve_cached_snapshot(self.model_id)
        model_source = str(cached_model) if cached_model is not None else self.model_id
        download_root = None if cached_model is not None else os.getenv(
            "ASR_DOWNLOAD_ROOT", "/root/.cache/prizmmemo-whisper"
        )
        hf_token = os.getenv("HF_TOKEN", "").strip()
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for pyannote diarization")

        source = "Runpod cache" if cached_model is not None else "Hugging Face"
        print(
            f"[startup] loading WhisperX model={self.model_id} source={source} "
            f"compute={self.compute_type} multilingual_chunk="
            f"{self.multilingual_chunk_length_sec}s"
        )
        self.asr_model = whisperx.load_model(
            model_source,
            self.device,
            compute_type=self.compute_type,
            download_root=download_root,
            local_files_only=cached_model is not None,
            vad_method="silero",
        )
        print("[startup] loading pyannote speaker-diarization-community-1")
        self.diarizer = DiarizationPipeline(
            token=hf_token,
            device=self.device,
            cache_dir=self.hf_cache_dir,
        )
        print(f"[startup] ready engine={ENGINE_VERSION} gpu={torch.cuda.get_device_name(0)}")

    def transcribe(self, job: dict[str, Any], request: dict[str, Any], audio_path: Path) -> dict[str, Any]:
        audio = whisperx.load_audio(str(audio_path))
        languages = request["languages"]
        _progress(job, "transcribing")
        if len(languages) == 1:
            result = self.asr_model.transcribe(
                audio,
                batch_size=self.batch_size,
                language=languages[0],
            )
            language_chunks: list[dict[str, Any]] = []
        else:
            pipeline = ExpectedLanguageBatchedInferencePipeline(
                self.asr_model.model,
                languages,
            )
            raw_segments, _ = pipeline.transcribe(
                audio,
                language=languages[0],
                task="transcribe",
                multilingual=False,
                batch_size=max(1, self.batch_size // len(languages)),
                chunk_length=self.multilingual_chunk_length_sec,
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=False,
            )
            segments = [
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text,
                    "words": [
                        {
                            "start": float(word.start),
                            "end": float(word.end),
                            "word": word.word,
                            "score": float(word.probability),
                        }
                        for word in (segment.words or [])
                    ],
                }
                for segment in raw_segments
            ]
            language_chunks = pipeline.language_chunks
            language_durations: Counter[str] = Counter()
            for chunk in language_chunks:
                language_durations[chunk["language"]] += chunk["duration_sec"]
            dominant_language = max(
                languages,
                key=lambda expected: language_durations[expected],
            )
            result = {
                "segments": segments,
                "language": dominant_language,
            }
        language = result["language"]

        alignment_applied = False
        _progress(job, "aligning")
        try:
            if len(languages) > 1:
                raise ValueError("multiple expected languages use segment-level timestamps")
            align_model, metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
                model_dir=self.hf_cache_dir,
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
            if len(languages) > 1:
                result["segments"] = split_segments_by_word_speaker(result["segments"])

        return _json_safe(
            {
                "meeting_id": request["meeting_id"],
                "segments": result["segments"],
                "detected_language": language,
                "language_mode": (
                    "forced_single_language"
                    if len(languages) == 1
                    else "expected_language_candidate_decoding"
                ),
                "multilingual_chunk_length_sec": (
                    None if len(languages) == 1 else self.multilingual_chunk_length_sec
                ),
                "language_chunks": language_chunks,
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
