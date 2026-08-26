# PrizmMemo WhisperX Runpod worker

Queue-based Runpod Serverless worker for GPU transcription, word alignment, and speaker diarization.

## Job contract

Submit the object in test_input.json to the Runpod /run endpoint. audio_url must be a short-lived HTTPS presigned URL whose host ends in .r2.cloudflarestorage.com. The worker never logs the signed URL and deletes the downloaded object from local disk after every job.

The completed output contains:

- meeting_id
- segments with relative timestamps, text, words, and speaker labels when diarization is enabled
- detected_language
- alignment_applied
- speaker_embeddings for later meeting-wide speaker reconciliation
- engine_version

## Required Runpod secret

- HF_TOKEN: a Hugging Face token with access to pyannote/speaker-diarization-community-1. Accept that model's user agreement before deploying.

## Optional environment variables

| Name | Default | Purpose |
| --- | --- | --- |
| WHISPER_MODEL | large-v3 | Faster-Whisper model identifier |
| WHISPER_COMPUTE_TYPE | float16 | CTranslate2 compute type |
| WHISPER_BATCH_SIZE | 16 | ASR batch size; lower this if GPU memory is exhausted |
| MAX_AUDIO_BYTES | 536870912 | Maximum downloaded input size |
| DOWNLOAD_TIMEOUT_SEC | 600 | Per-read timeout while downloading from R2 |
| ALLOWED_AUDIO_HOST_SUFFIX | .r2.cloudflarestorage.com | Required signed-URL hostname suffix |
| HF_HOME | /root/.cache/huggingface | Hugging Face and model cache root; override only when attaching persistent cache storage |

For the initial endpoint use a 24 GB GPU, Queue mode, concurrency 1, max workers 1, and active workers 0. The first cold start downloads the model artifacts unless Runpod's model cache or a network volume already contains them.

## Deployment gate

Do not connect this worker to the main PrizmMemo job pipeline until one short signed-R2 audio file completes successfully and the output has transcript segments, speaker labels, and no signed URL in logs.
