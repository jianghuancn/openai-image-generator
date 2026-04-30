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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--reference-image",
        "--image",
        dest="reference_image",
        default=None,
        help="Path to a reference image for image editing mode",
    )
    return parser.parse_args(argv)


def decode_image(item: Any) -> bytes:
    """Decode an image response item into bytes."""
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    raise ValueError("No base64 image data returned by the API")


def build_metadata(
    model: str,
    prompt: str,
    size: str,
    quality: str,
    n: int,
    paths: list[str],
    timestamp: str,
    reference_image: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": n,
        "paths": paths,
        "created_at": timestamp,
    }
    if reference_image is not None:
        meta["reference_image"] = reference_image
    return meta


def save_images(items: list[Any], outdir: Path) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, item in enumerate(items, start=1):
        image_path = outdir / f"image_{index}.png"
        image_path.write_bytes(decode_image(item))
        paths.append(str(image_path))
    return paths


def run(
    client: Any,
    model: str,
    prompt: str,
    size: str,
    quality: str,
    n: int,
    outdir: Path,
    timestamp: str,
    reference_image: str | None = None,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if reference_image is None:
        result = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
    else:
        with open(reference_image, "rb") as image_file:
            result = client.images.edit(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
                image=image_file,
            )

    paths = save_images(result.data, outdir)
    meta = build_metadata(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
        paths=paths,
        timestamp=timestamp,
        reference_image=reference_image,
    )
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


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
    metadata = run(
        client,
        args.model,
        args.prompt,
        args.size,
        args.quality,
        args.n,
        outdir,
        timestamp,
        reference_image=args.reference_image,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
