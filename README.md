# PrizmMemo WhisperX Runpod worker

Queue-based Runpod Serverless worker for GPU transcription, word alignment, and speaker diarization.

`input.languages` contains one to three expected language codes. A single code is forced through
WhisperX's fast, stable single-language path. With two or more codes, engine 1.3.0 treats those
codes as the allowed candidate set. Every short VAD speech chunk is decoded once under each
configured language token in one expanded GPU batch. The worker selects the candidate with the
best ASR likelihood plus a small language-detector prior, preventing accented English from being
decoded as malformed Bahasa Melayu merely because a short-window classifier guessed `ms`.
This mode intentionally skips single-language wav2vec alignment and retains segment-level
timestamps for diarization.
The image pins faster-whisper 1.2.0 because that release correctly pads feature input when
`chunk_length` is below 30 seconds.

## Job contract

Submit the object in test_input.json to the Runpod /run endpoint. audio_url must be a short-lived HTTPS presigned URL whose host ends in .r2.cloudflarestorage.com. The worker never logs the signed URL and deletes the downloaded object from local disk after every job.

The completed output contains:

- meeting_id
- segments with relative timestamps, text, words, and speaker labels when diarization is enabled
- detected_language
- language_mode, multilingual_chunk_length_sec, and per-chunk language diagnostics
- alignment_applied
- speaker_embeddings for later meeting-wide speaker reconciliation
- engine_version

## Required Runpod secret

- HF_TOKEN: a Hugging Face token with access to pyannote/speaker-diarization-community-1. Accept that model's user agreement before deploying.

## Optional environment variables

| Name | Default | Purpose |
| --- | --- | --- |
| WHISPER_MODEL_ID | Systran/faster-whisper-large-v3 | Faster-Whisper Hugging Face repository |
| WHISPER_COMPUTE_TYPE | float16 | CTranslate2 compute type |
| WHISPER_BATCH_SIZE | 16 | ASR batch size; lower this if GPU memory is exhausted |
| MULTILINGUAL_CHUNK_LENGTH_SEC | 8 | Maximum VAD speech-chunk length used for configured-language candidate decoding |
| MAX_AUDIO_BYTES | 536870912 | Maximum downloaded input size |
| DOWNLOAD_TIMEOUT_SEC | 600 | Per-read timeout while downloading from R2 |
| ALLOWED_AUDIO_HOST_SUFFIX | .r2.cloudflarestorage.com | Required signed-URL hostname suffix |
| HF_HOME | /root/.cache/huggingface | Runtime Hugging Face cache for pyannote and alignment models |

For the initial endpoint use a 24 GB GPU, Queue mode, concurrency 1, max workers 1, and active workers 0. Set the Runpod Model field to Systran/faster-whisper-large-v3. The worker resolves that cached snapshot automatically and downloads the gated pyannote model at runtime with HF_TOKEN.

## Deployment gate

Do not connect this worker to the main PrizmMemo job pipeline until one short signed-R2 audio file completes successfully and the output has transcript segments, speaker labels, and no signed URL in logs.
