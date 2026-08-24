import runpod
import requests
import tempfile
import os

def handler(job):
    job_input = job.get("input", {})
    audio_url = job_input.get("audio_url")
    meeting_id = job_input.get("meeting_id", "unknown")

    if not audio_url:
        return {"error": "Missing 'audio_url' in input"}

    # Download audio to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"{meeting_id}.opus")

        r = requests.get(audio_url, timeout=300)
        r.raise_for_status()
        with open(audio_path, "wb") as f:
            f.write(r.content)

        # TODO:
        # 1) Run WhisperX on audio_path (GPU)
        # 2) Run PyAnnote diarization
        # 3) Return transcript JSON (or upload to R2 and return key)

        return {
            "message": "Worker received audio successfully (pipeline not implemented yet).",
            "meeting_id": meeting_id,
            "downloaded_bytes": len(r.content),
        }

runpod.serverless.start({"handler": handler})
