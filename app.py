from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import io
import soundfile as sf

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str = Header(None),
    model: str = "whisper-1"
):
    # Check authorization
    if authorization != "Bearer EMPTY":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Read the audio file
    audio_bytes = await file.read()
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Process the audio file through the pipeline
    result = pipe({"array": audio, "sampling_rate": sample_rate})

    return JSONResponse({"text": result["text"]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
