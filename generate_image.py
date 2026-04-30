#!/usr/bin/env python3
"""Generate images with the OpenAI Images API."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

_RATIO_MAP: dict[str, str] = {
    "square": "1024x1024",
    "1:1":    "1024x1024",
    "portrait": "1024x1536",
    "2:3":    "1024x1536",
    "landscape": "1536x1024",
    "3:2":    "1536x1024",
    "vertical": "1024x1792",
    "9:16":   "1024x1792",
    "wide":   "1792x1024",
    "16:9":   "1792x1024",
}

_RATIO_CHOICES = list(_RATIO_MAP)

# Sizes the edit API does not support (only 1024x1024, 1024x1536, 1536x1024 are valid).
_EDIT_UNSUPPORTED_SIZES: set[str] = {"1024x1792", "1792x1024"}

# Per-image output pricing (USD).  Source: GPT-image pricing table.
_IMAGE_OUTPUT_PRICING: dict[tuple[str, str], float] = {
    ("1024x1024", "low"):    0.006,
    ("1024x1024", "medium"): 0.053,
    ("1024x1024", "high"):   0.211,
    ("1024x1536", "low"):    0.005,
    ("1024x1536", "medium"): 0.041,
    ("1024x1536", "high"):   0.165,
    ("1536x1024", "low"):    0.005,
    ("1536x1024", "medium"): 0.041,
    ("1536x1024", "high"):   0.165,
}

# Sizes not in the table; mapped to the nearest defined size for estimation.
_APPROXIMATE_SIZE_MAP: dict[str, str] = {
    "1024x1792": "1024x1536",
    "1792x1024": "1536x1024",
}

_TEXT_INPUT_PRICE_PER_TOKEN: float = 5.00 / 1_000_000    # USD per token
_IMAGE_INPUT_PRICE_PER_TOKEN: float = 8.00 / 1_000_000  # USD per token
_OUTPUT_IMAGE_PRICE_PER_TOKEN: float = 30.00 / 1_000_000 # USD per token


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert SDK/SimpleNamespace objects to JSON-serializable dicts."""
    if hasattr(obj, "model_dump"):
        return _to_json_serializable(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _to_json_serializable(v) for k, v in vars(obj).items()}
    return obj


def extract_usage(result: Any) -> dict[str, Any] | None:
    """Return a normalised usage dict from an API result, or None if absent.

    Supports both attribute-style (result.usage) and dict-style access.
    """
    usage = getattr(result, "usage", None)
    if usage is None and isinstance(result, dict):
        usage = result.get("usage")
    if usage is None:
        return None

    details = _get_attr_or_key(usage, "input_tokens_details")
    text_tokens: int | None = None
    image_tokens: int | None = None
    if details is not None:
        text_tokens = _get_attr_or_key(details, "text_tokens")
        image_tokens = _get_attr_or_key(details, "image_tokens")

    return {
        "input_tokens": _get_attr_or_key(usage, "input_tokens"),
        "output_tokens": _get_attr_or_key(usage, "output_tokens"),
        "total_tokens": _get_attr_or_key(usage, "total_tokens"),
        "input_text_tokens": text_tokens,
        "input_image_tokens": image_tokens,
    }


def estimate_cost(
    model: str,
    size: str,
    quality: str,
    n: int,
    prompt: str,
    reference_image: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute or estimate USD cost for one image generation call.

    When *usage* is supplied (extracted from the API response), actual token
    counts are used for input costs and output-token pricing. Otherwise tokens
    are estimated from the prompt length and a conservative image-input figure,
    and output cost falls back to the per-image pricing table.
    """
    notes: list[str] = []
    is_size_approximate = size in _APPROXIMATE_SIZE_MAP
    is_quality_auto = quality == "auto"

    # ── text input tokens ──────────────────────────────────────────────────
    if usage is not None and usage.get("input_text_tokens") is not None:
        text_tokens: int = usage["input_text_tokens"]
        text_tokens_estimated = False
    else:
        # ~4 characters per token (ceiling division)
        text_tokens = max(1, (len(prompt) + 3) // 4)
        text_tokens_estimated = True
        notes.append(
            f"Text tokens estimated from prompt length "
            f"({len(prompt)} chars → {text_tokens} tokens)"
        )

    # ── image input tokens (reference-image edits) ─────────────────────────
    if usage is not None and usage.get("input_image_tokens") is not None:
        image_input_tokens: int = usage["input_image_tokens"]
        image_tokens_estimated = False
    elif reference_image is not None:
        image_input_tokens = 1024  # conservative estimate for a typical image
        image_tokens_estimated = True
        notes.append(
            "Image input tokens estimated at 1024 "
            "(conservative estimate for reference image; actual value varies by image)"
        )
    else:
        image_input_tokens = 0
        image_tokens_estimated = False

    # ── output image tokens and output cost ────────────────────────────────
    output_image_tokens: int | None = (
        usage.get("output_tokens") if usage is not None else None
    )

    if output_image_tokens is not None:
        # API provided output token count — use token-based pricing ($30/1M)
        output_cost = output_image_tokens * _OUTPUT_IMAGE_PRICE_PER_TOKEN
        output_tokens_estimated = False
    else:
        # No output token count — fall back to per-image pricing table
        output_tokens_estimated = True
        lookup_size = _APPROXIMATE_SIZE_MAP.get(size, size)
        lookup_quality = "medium" if is_quality_auto else quality
        if is_size_approximate:
            notes.append(
                f"Size {size} not in pricing table; "
                f"using {lookup_size} pricing as closest match (estimated)"
            )
        if is_quality_auto:
            notes.append("Quality 'auto' treated as 'medium' for cost estimation")
        key = (lookup_size, lookup_quality)
        if key in _IMAGE_OUTPUT_PRICING:
            output_cost = _IMAGE_OUTPUT_PRICING[key] * n
        else:
            output_cost = 0.0
            notes.append(f"Output cost unavailable for size={size} quality={quality}")

    # ── totals ─────────────────────────────────────────────────────────────
    text_cost = text_tokens * _TEXT_INPUT_PRICE_PER_TOKEN
    image_input_cost = image_input_tokens * _IMAGE_INPUT_PRICE_PER_TOKEN
    total_usd = text_cost + image_input_cost + output_cost

    estimated = (
        text_tokens_estimated
        or image_tokens_estimated
        or output_tokens_estimated
        or is_size_approximate
        or is_quality_auto
    )

    return {
        "estimated": estimated,
        "currency": "USD",
        "total_usd": total_usd,
        "input_text_tokens": text_tokens,
        "input_image_tokens": image_input_tokens,
        "output_image_tokens": output_image_tokens,
        "notes": notes,
    }


def validate_edit_ratio_compat(
    ratio: str | None,
    explicit_size: str | None,
    reference_image: str | None,
) -> None:
    """Raise SystemExit when edit mode would use a size unsupported by OpenAI's edit API.

    Only fires when: reference_image is set, no explicit --size given, and the
    ratio resolves to 1024x1792 or 1792x1024 (vertical/wide).  An explicit
    --size always wins, so the caller is free to pass any literal size they want.
    """
    if reference_image is None or ratio is None or explicit_size is not None:
        return
    resolved = _RATIO_MAP.get(ratio)
    if resolved in _EDIT_UNSUPPORTED_SIZES:
        raise SystemExit(
            f"Error: --ratio '{ratio}' resolves to {resolved}, which is not supported "
            f"in reference-image edit mode. Edit mode only accepts square (1024x1024), "
            f"portrait (1024x1536), or landscape (1536x1024) sizes. "
            f"Use a generation-only call (omit --reference-image) or pass an explicit "
            f"--size to override."
        )


def resolve_size(size: str | None, ratio: str | None) -> str:
    if size is not None:
        return size
    if ratio is not None:
        if ratio not in _RATIO_MAP:
            raise ValueError(f"Unknown ratio '{ratio}'. Valid choices: {list(_RATIO_MAP)}")
        return _RATIO_MAP[ratio]
    return "1024x1024"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with OpenAI.")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--model", default="gpt-image-1", help="OpenAI image model")
    parser.add_argument("--size", default=None, help="Image size, e.g. 1024x1024 (overrides --ratio)")
    parser.add_argument(
        "--ratio",
        "--aspect-ratio",
        dest="ratio",
        default=None,
        choices=_RATIO_CHOICES,
        help="Aspect ratio shorthand (e.g. square, portrait, 16:9). Overridden by --size.",
    )
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
    b64 = _get_attr_or_key(item, "b64_json")
    if b64:
        try:
            return base64.b64decode(b64, validate=True)
        except binascii.Error as exc:
            raise ValueError("Invalid base64 image data returned by the API") from exc
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
    ratio: str | None = None,
    usage: dict[str, Any] | None = None,
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
    if ratio is not None:
        meta["ratio"] = ratio
    if usage is not None:
        meta["usage"] = usage
    # When usage is the combined {"raw": ..., "normalized": ...} form (from run()),
    # pass only the normalized part to estimate_cost.
    cost_usage = (
        usage["normalized"]
        if isinstance(usage, dict) and "normalized" in usage
        else usage
    )
    meta["cost"] = estimate_cost(
        model=model,
        size=size,
        quality=quality,
        n=n,
        prompt=prompt,
        reference_image=reference_image,
        usage=cost_usage,
    )
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
    ratio: str | None = None,
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

    # Capture raw usage object before normalizing, for JSON persistence.
    raw_usage_obj = _get_attr_or_key(result, "usage")
    normalized_usage = extract_usage(result)
    if normalized_usage is not None:
        usage_for_meta: dict[str, Any] | None = {
            "raw": _to_json_serializable(raw_usage_obj),
            "normalized": normalized_usage,
        }
    else:
        usage_for_meta = None

    data = _get_attr_or_key(result, "data", [])
    paths = save_images(data, outdir)
    meta = build_metadata(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
        paths=paths,
        timestamp=timestamp,
        reference_image=reference_image,
        ratio=ratio,
        usage=usage_for_meta,
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

    size = resolve_size(args.size, args.ratio)
    validate_edit_ratio_compat(args.ratio, args.size, args.reference_image)
    client = OpenAI()
    metadata = run(
        client,
        args.model,
        args.prompt,
        size,
        args.quality,
        args.n,
        outdir,
        timestamp,
        reference_image=args.reference_image,
        ratio=args.ratio,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
