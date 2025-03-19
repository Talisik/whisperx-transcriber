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
    return JSONResponse(
        content=WhisperXRemote.do(
            url,
            model,
            device,
            compute_type,
        ).model_dump(),
        status_code=200,
    )
