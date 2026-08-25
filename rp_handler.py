import runpod

def handler(event):
    return {"ok": True}

runpod.serverless.start({"handler": handler})
