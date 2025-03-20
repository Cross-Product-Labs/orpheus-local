from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import gguf_orpheus
import io
import wave

app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stream-audio")
async def stream_audio(text: str = "Hello, this is a test of the Orpheus text to speech model."):
    # Create an in-memory bytes buffer
    audio_buffer = io.BytesIO()

    # Create a WAV file in the buffer
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(24000)  # 24kHz sample rate

        # Get audio chunks and write them to the WAV file
        for chunk in gguf_orpheus.generate_audio(text):
            wav_file.writeframes(chunk)

    # Reset buffer position to start
    audio_buffer.seek(0)

    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=audio.wav",
            "Accept-Ranges": "bytes",
            "Content-Type": "audio/wav"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)