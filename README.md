# audio2txt

Here is a step-by-step guide to install and run the provided FastAPI application for speech-to-text transcription using OpenAI's Whisper model on GitHub.

## 1. Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or later
- Git
- CUDA (for GPU support, if available)

## 2. Clone the Repository
First, you need to clone the repository from GitHub. Open your terminal and run:

```sh
git clone <your-github-repo-url>
cd <your-repo-directory>
```

## 3. Create a Virtual Environment
It's a good practice to create a virtual environment to manage your dependencies. Run:

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

## 4. Install Dependencies
Next, install the required Python packages. Create a `requirements.txt` file with the following contents:

```plaintext
fastapi
uvicorn
torch
transformers
soundfile
```

Then, run:

```sh
pip install -r requirements.txt
```

## 5. Create the FastAPI Application
Create a directory structure as follows:

```plaintext
<repo-directory>/
├── app.py
└── requirements.txt
```

Copy the provided `app.py` content into the `app.py` file:

```python
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
```

## 6. Run the Application
Finally, you can run the FastAPI application using Uvicorn:

```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000/v1/audio/transcriptions`.

## 7. Testing the API
You can test the API using a tool like `curl` or Postman. Here is an example using `curl`:

```sh
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
-H "accept: application/json" \
-H "authorization: Bearer EMPTY" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/your/audio/file"
```

Replace `path/to/your/audio/file` with the actual path to your audio file.

This completes the setup and installation guide for the FastAPI application.
