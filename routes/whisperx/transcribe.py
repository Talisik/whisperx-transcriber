from time import perf_counter

import whisperx
from fastapi.responses import JSONResponse

from helpers.whisperx_remote import WhisperXRemote

from .router import router


@router.get("/transcribe")
async def transcribe(
    url: str,
    model: WhisperXRemote.MODEL = WhisperXRemote.DEFAULT_MODEL,
    device: WhisperXRemote.DEVICE = WhisperXRemote.DEFAULT_DEVICE,
    compute_type: WhisperXRemote.COMPUTE_TYPE = WhisperXRemote.DEFAULT_COMPUTE_TYPE,
):
    t1 = perf_counter()

    print(f"🚧 Loading audio from {url}...")

    audio = whisperx.load_audio(url)

    print("✅ Audio loaded.")

    print("🚧 Loading model...")

    remote = WhisperXRemote.load(
        model=model,
        device=device,
        compute_type=compute_type,
    )

    print("✅ Model loaded.")

    print("🚧 Transcribing...")
    result = remote.model.transcribe(
        audio,
        batch_size=16,
    )

    print("🎉 Transcription done.")

    print("🚧 Loading align model...")

    model_a, metadata = remote.load_align_model(result["language"])

    print("✅ Align model loaded.")

    print("🚧 Aligning...")

    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        remote.model.device,  # type: ignore
        return_char_alignments=False,
    )

    print("🎉 Alignment done.")

    return JSONResponse(
        content={
            "elapsed_time": perf_counter() - t1,
            "result": result,
        },
        status_code=200,
    )
