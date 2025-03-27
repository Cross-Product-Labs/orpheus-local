import torch
from fastapi import FastAPI, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import gguf_orpheus
import json
import asyncio
from typing import AsyncGenerator
import numpy as np

app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add startup event to warm up the model
@app.on_event("startup")
async def startup_event():
    # Import here to avoid circular imports
    from decoder import model, convert_to_audio, snac_device

    # Warm up the model with a small inference
    dummy_codes = [
        torch.zeros((1, 3), dtype=torch.int32, device=snac_device),
        torch.zeros((1, 6), dtype=torch.int32, device=snac_device),
        torch.zeros((1, 12), dtype=torch.int32, device=snac_device)
    ]

    with torch.no_grad():
        model.decode(dummy_codes)

    print("Model warmed up and ready for inference")

async def token_to_audio_stream(
    text: str,
    voice: str = "tara",
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    max_tokens: int = 1200
) -> AsyncGenerator[bytes, None]:
    buffer = []
    count = 0
    speech_started = False

    # Convert to async generator for better performance
    async def async_token_gen():
        for token_text in gguf_orpheus.generate_tokens_from_api(
            prompt=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens
        ):
            yield token_text

    async for token_text in async_token_gen():
        token = gguf_orpheus.turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Process audio when we have enough tokens
            if count >= 21 and count % 7 == 0:
                # Use a fixed slice length instead of calculating it each time
                slice_length = 21 if count < 28 else 28
                buffer_to_proc = buffer[-slice_length:]

                audio_samples = gguf_orpheus.convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    # Ultra-lightweight silence detection
                    # Only check a few samples from the beginning of the chunk
                    if not speech_started:
                        # Check just 10 samples for non-zero values
                        # This is much faster than numpy operations
                        check_size = min(20, len(audio_samples) // 2)
                        for i in range(0, check_size * 2, 2):
                            # Get the 16-bit sample value (little endian)
                            sample = int.from_bytes(audio_samples[i:i+2], byteorder='little', signed=True)
                            if abs(sample) > 500:  # Threshold for 16-bit audio
                                speech_started = True
                                break

                    # Only yield if we've found speech
                    if speech_started:
                        yield audio_samples

@app.get("/stream-audio")
async def stream_audio(
    text: str = "Hello, this is a test of the Orpheus text to speech model.",
    voice: str = "tara",
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    max_tokens: int = 1200
):
    # Validate parameters early to avoid unnecessary processing
    if voice not in gguf_orpheus.AVAILABLE_VOICES:
        voice = gguf_orpheus.DEFAULT_VOICE

    # Constrain parameters to valid ranges
    temperature = max(0.1, min(1.5, temperature))
    top_p = max(0.1, min(1.0, top_p))
    repetition_penalty = max(1.0, min(2.0, repetition_penalty))
    max_tokens = max(100, min(4096, max_tokens))

    async def audio_stream():
        # Send audio format information
        yield "event: format\n"
        yield f"data: {json.dumps({'sampleRate': 24000, 'channels': 1, 'bitsPerSample': 16})}\n\n"

        # Stream audio chunks as they're generated
        async for chunk in token_to_audio_stream(
            text=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens
        ):
            # Use hex encoding which is faster than base64
            chunk_b64 = chunk.hex()
            yield "event: audio\n"
            yield f"data: {chunk_b64}\n\n"

    return StreamingResponse(
        audio_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)