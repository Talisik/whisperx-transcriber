from functools import lru_cache
from time import perf_counter
from typing import Any, Literal

import torch
import whisperx
from pydantic import BaseModel
from whisperx.asr import FasterWhisperPipeline


class WhisperXRemote:
    class Payload(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        elapsed_time: float
        result: Any

    MODEL = Literal[
        "large-v2",
        "tiny.en",
        "base.en",
        "small.en",
    ]
    DEVICE = Literal[
        "cuda",
        "cpu",
    ]
    COMPUTE_TYPE = Literal[
        "float16",
        "float32",
        "int8",
    ]

    DEFAULT_MODEL: MODEL = "tiny.en"
    DEFAULT_DEVICE: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULT_COMPUTE_TYPE: COMPUTE_TYPE = "int8"

    model: FasterWhisperPipeline

    def __init__(self):
        self.load_align_model = lru_cache(None)(
            self.load_align_model,
        )

    @classmethod
    @lru_cache(None)
    def load(
        cls,
        *,
        model: MODEL = DEFAULT_MODEL,
        device: DEVICE = DEFAULT_DEVICE,
        compute_type: COMPUTE_TYPE = DEFAULT_COMPUTE_TYPE,
    ):
        instance = WhisperXRemote()
        instance.model = whisperx.load_model(
            model,
            device,
            compute_type=compute_type,
            # download_root=volume_path,
            # language=language_code,
        )

        return instance

    def load_align_model(
        self,
        language_code: str,
    ):
        return whisperx.load_align_model(
            language_code=language_code,
            device=self.model.device,  # type: ignore
        )

    @staticmethod
    def do(
        url: str,
        model: MODEL = DEFAULT_MODEL,
        device: DEVICE = DEFAULT_DEVICE,
        compute_type: COMPUTE_TYPE = DEFAULT_COMPUTE_TYPE,
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

        return WhisperXRemote.Payload(
            elapsed_time=perf_counter() - t1,
            result=result,
        )
