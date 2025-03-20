from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import gguf_orpheus
import json
import asyncio
from typing import AsyncGenerator

app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def token_to_audio_stream(text: str) -> AsyncGenerator[bytes, None]:
    buffer = []
    count = 0

    # Get the token generator
    token_gen = gguf_orpheus.generate_tokens_from_api(text)

    for token_text in token_gen:
        token = gguf_orpheus.turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Process audio when we have enough tokens (7 tokens per frame)
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]  # Get last 28 tokens (4 frames worth)
                audio_samples = gguf_orpheus.convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples
                    # Small delay to maintain ~12 frames per second
                    await asyncio.sleep(0.08)  # 1/12 ≈ 0.083 seconds

@app.get("/stream-audio")
async def stream_audio(text: str = "Hello, this is a test of the Orpheus text to speech model."):
    async def audio_stream():
        # Send audio format information
        yield "event: format\n"
        yield f"data: {json.dumps({'sampleRate': 24000, 'channels': 1, 'bitsPerSample': 16})}\n\n"

        # Stream audio chunks as they're generated
        async for chunk in token_to_audio_stream(text):
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