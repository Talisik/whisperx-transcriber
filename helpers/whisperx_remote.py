from functools import lru_cache
from typing import Literal

import torch
import whisperx
from whisperx.asr import FasterWhisperPipeline


class WhisperXRemote:
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
