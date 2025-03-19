import argparse
import json

from helpers.whisperx_remote import WhisperXRemote

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WhisperX transcriber.",
    )

    parser.add_argument(
        "url",
        type=str,
        help="The URL of the audio file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=WhisperXRemote.DEFAULT_MODEL,
        help="The model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=WhisperXRemote.DEFAULT_DEVICE,
        help="The device to use.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=WhisperXRemote.DEFAULT_COMPUTE_TYPE,
        help="The compute type to use.",
    )

    args = parser.parse_args()

    payload = WhisperXRemote.do(
        args.url,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    print(
        json.dumps(
            payload.model_dump(),
            indent=2,
        )
    )
