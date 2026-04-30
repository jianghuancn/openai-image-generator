#!/usr/bin/env python3
"""Generate images with the OpenAI Images API."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with OpenAI.")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--model", default="gpt-image-1", help="OpenAI image model")
    parser.add_argument("--size", default="1024x1024", help="Image size, e.g. 1024x1024")
    parser.add_argument(
        "--quality",
        default="medium",
        choices=["low", "medium", "high", "auto"],
        help="Generation quality",
    )
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to outputs/<timestamp>",
    )
    return parser.parse_args()


def decode_image(item: Any) -> bytes:
    """Decode an image response item into bytes."""
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    raise ValueError("No base64 image data returned by the API")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir or f"outputs/{timestamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    result = client.images.generate(
        model=args.model,
        prompt=args.prompt,
        size=args.size,
        quality=args.quality,
        n=args.n,
    )

    paths: list[str] = []
    for index, item in enumerate(result.data, start=1):
        image_path = outdir / f"image_{index}.png"
        image_path.write_bytes(decode_image(item))
        paths.append(str(image_path))

    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "quality": args.quality,
        "n": args.n,
        "paths": paths,
        "created_at": timestamp,
    }
    metadata_path = outdir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
