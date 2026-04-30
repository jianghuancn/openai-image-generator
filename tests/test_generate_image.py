"""Tests for generate_image module — written first (TDD RED phase)."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _b64_item(raw: bytes) -> SimpleNamespace:
    """Fake API response item with b64_json set."""
    return SimpleNamespace(b64_json=base64.b64encode(raw).decode())


def _mock_generate_client(raw: bytes = b"fake_png") -> MagicMock:
    client = MagicMock()
    result = MagicMock(data=[_b64_item(raw)])
    result.usage = None
    client.images.generate.return_value = result
    return client


def _mock_edit_client(raw: bytes = b"edited_png") -> MagicMock:
    client = MagicMock()
    result = MagicMock(data=[_b64_item(raw)])
    result.usage = None
    client.images.edit.return_value = result
    return client


def _mock_generate_client_with_usage(usage_obj: Any, raw: bytes = b"fake_png") -> MagicMock:
    """Mock client whose generate result exposes a specific usage object."""
    client = MagicMock()
    result = MagicMock(data=[_b64_item(raw)])
    result.usage = usage_obj
    client.images.generate.return_value = result
    return client


# ── decode_image ──────────────────────────────────────────────────────────────

class TestDecodeImage:
    def test_decodes_valid_b64_json(self):
        from generate_image import decode_image

        raw = b"hello world"
        item = _b64_item(raw)
        assert decode_image(item) == raw

    def test_raises_value_error_when_b64_json_is_none(self):
        from generate_image import decode_image

        item = SimpleNamespace(b64_json=None)
        with pytest.raises(ValueError, match="No base64"):
            decode_image(item)

    def test_raises_value_error_when_b64_json_missing(self):
        from generate_image import decode_image

        item = SimpleNamespace()  # no b64_json attribute at all
        with pytest.raises(ValueError, match="No base64"):
            decode_image(item)

    def test_raises_value_error_when_b64_json_is_invalid(self):
        from generate_image import decode_image

        item = SimpleNamespace(b64_json="!!!!")
        with pytest.raises(ValueError, match="Invalid base64"):
            decode_image(item)


# ── build_metadata ────────────────────────────────────────────────────────────

class TestBuildMetadata:
    def test_contains_all_required_fields(self):
        from generate_image import build_metadata

        meta = build_metadata(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1024",
            quality="medium",
            n=1,
            paths=["outputs/ts/image_1.png"],
            timestamp="20260101-120000",
        )
        assert meta["model"] == "gpt-image-1"
        assert meta["prompt"] == "a dog"
        assert meta["size"] == "1024x1024"
        assert meta["quality"] == "medium"
        assert meta["n"] == 1
        assert meta["paths"] == ["outputs/ts/image_1.png"]
        assert meta["created_at"] == "20260101-120000"

    def test_excludes_reference_image_when_not_provided(self):
        from generate_image import build_metadata

        meta = build_metadata(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1024",
            quality="medium",
            n=1,
            paths=[],
            timestamp="20260101-120000",
        )
        assert "reference_image" not in meta

    def test_excludes_reference_image_when_none(self):
        from generate_image import build_metadata

        meta = build_metadata(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1024",
            quality="medium",
            n=1,
            paths=[],
            timestamp="20260101-120000",
            reference_image=None,
        )
        assert "reference_image" not in meta

    def test_includes_reference_image_when_provided(self):
        from generate_image import build_metadata

        meta = build_metadata(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1024",
            quality="medium",
            n=1,
            paths=[],
            timestamp="20260101-120000",
            reference_image="/tmp/ref.png",
        )
        assert meta["reference_image"] == "/tmp/ref.png"


# ── save_images ───────────────────────────────────────────────────────────────

class TestSaveImages:
    def test_saves_each_item_as_png(self, tmp_path):
        from generate_image import save_images

        items = [_b64_item(b"img1"), _b64_item(b"img2")]
        paths = save_images(items, tmp_path)
        assert len(paths) == 2
        assert all(Path(p).exists() for p in paths)

    def test_returns_correct_paths(self, tmp_path):
        from generate_image import save_images

        items = [_b64_item(b"img1")]
        paths = save_images(items, tmp_path)
        assert paths[0] == str(tmp_path / "image_1.png")

    def test_saves_correct_bytes(self, tmp_path):
        from generate_image import save_images

        items = [_b64_item(b"PNG_DATA")]
        paths = save_images(items, tmp_path)
        assert Path(paths[0]).read_bytes() == b"PNG_DATA"

    def test_filenames_indexed_from_one(self, tmp_path):
        from generate_image import save_images

        items = [_b64_item(b"a"), _b64_item(b"b"), _b64_item(b"c")]
        paths = save_images(items, tmp_path)
        names = [Path(p).name for p in paths]
        assert names == ["image_1.png", "image_2.png", "image_3.png"]


# ── parse_args — new --reference-image / --image option ──────────────────────

class TestParseArgs:
    def test_reference_image_long_form(self):
        from generate_image import parse_args

        args = parse_args(["--prompt", "test", "--reference-image", "/tmp/img.png"])
        assert args.reference_image == "/tmp/img.png"

    def test_image_alias_sets_same_dest(self):
        from generate_image import parse_args

        args = parse_args(["--prompt", "test", "--image", "/tmp/img.png"])
        assert args.reference_image == "/tmp/img.png"

    def test_no_reference_image_defaults_to_none(self):
        from generate_image import parse_args

        args = parse_args(["--prompt", "test"])
        assert args.reference_image is None

    def test_existing_args_still_work(self):
        from generate_image import parse_args

        args = parse_args([
            "--prompt", "a cat",
            "--model", "gpt-image-1",
            "--size", "512x512",
            "--quality", "high",
            "--n", "3",
        ])
        assert args.prompt == "a cat"
        assert args.model == "gpt-image-1"
        assert args.size == "512x512"
        assert args.quality == "high"
        assert args.n == 3

    def test_outdir_still_works(self):
        from generate_image import parse_args

        args = parse_args(["--prompt", "x", "--outdir", "/tmp/out"])
        assert args.outdir == "/tmp/out"


# ── run() — core orchestration ────────────────────────────────────────────────

class TestRun:
    def test_calls_generate_when_no_reference_image(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
            tmp_path, "20260101-120000", reference_image=None)
        client.images.generate.assert_called_once()
        client.images.edit.assert_not_called()

    def test_calls_edit_when_reference_image_provided(self, tmp_path):
        from generate_image import run

        client = _mock_edit_client()
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"REF")
        outdir = tmp_path / "out"
        run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
            outdir, "20260101-120000", reference_image=str(ref))
        client.images.edit.assert_called_once()
        client.images.generate.assert_not_called()

    def test_generate_params_forwarded_correctly(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        run(client, "gpt-image-1", "a cat", "512x512", "high", 2,
            tmp_path, "ts", reference_image=None)
        client.images.generate.assert_called_once_with(
            model="gpt-image-1",
            prompt="a cat",
            size="512x512",
            quality="high",
            n=2,
        )

    def test_edit_params_forwarded_correctly(self, tmp_path):
        from generate_image import run

        client = _mock_edit_client()
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"REF")
        outdir = tmp_path / "out"
        run(client, "gpt-image-1", "a cat", "512x512", "high", 2,
            outdir, "ts", reference_image=str(ref))
        kw = client.images.edit.call_args.kwargs
        assert kw["model"] == "gpt-image-1"
        assert kw["prompt"] == "a cat"
        assert kw["size"] == "512x512"
        assert kw["quality"] == "high"
        assert kw["n"] == 2

    def test_edit_passes_file_like_object(self, tmp_path):
        from generate_image import run

        client = _mock_edit_client()
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"REF_BYTES")
        outdir = tmp_path / "out"
        run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
            outdir, "ts", reference_image=str(ref))
        kw = client.images.edit.call_args.kwargs
        assert "image" in kw
        assert hasattr(kw["image"], "read")

    def test_metadata_excludes_reference_image_when_not_used(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "20260101-120000", reference_image=None)
        assert "reference_image" not in meta

    def test_metadata_includes_reference_image_when_used(self, tmp_path):
        from generate_image import run

        client = _mock_edit_client()
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"REF")
        outdir = tmp_path / "out"
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   outdir, "ts", reference_image=str(ref))
        assert meta["reference_image"] == str(ref)

    def test_saves_metadata_json_file(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
            tmp_path, "ts")
        meta_file = tmp_path / "metadata.json"
        assert meta_file.exists()
        saved = json.loads(meta_file.read_text())
        assert saved["prompt"] == "a dog"

    def test_creates_outdir_if_missing(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        outdir = tmp_path / "deep" / "nested"
        run(client, "gpt-image-1", "test", "1024x1024", "medium", 1,
            outdir, "ts")
        assert outdir.exists()

    def test_returns_metadata_dict(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "20260101-120000")
        assert isinstance(meta, dict)
        assert meta["model"] == "gpt-image-1"
        assert meta["prompt"] == "a dog"
        assert "paths" in meta
        assert "created_at" in meta


# ── resolve_size ──────────────────────────────────────────────────────────────

class TestResolveSize:
    def test_returns_default_when_neither_size_nor_ratio(self):
        from generate_image import resolve_size

        assert resolve_size(None, None) == "1024x1024"

    def test_returns_explicit_size_unchanged(self):
        from generate_image import resolve_size

        assert resolve_size("512x512", None) == "512x512"

    def test_explicit_size_overrides_ratio(self):
        from generate_image import resolve_size

        assert resolve_size("512x512", "portrait") == "512x512"

    def test_square_maps_to_1024x1024(self):
        from generate_image import resolve_size

        assert resolve_size(None, "square") == "1024x1024"

    def test_portrait_maps_to_1024x1536(self):
        from generate_image import resolve_size

        assert resolve_size(None, "portrait") == "1024x1536"

    def test_landscape_maps_to_1536x1024(self):
        from generate_image import resolve_size

        assert resolve_size(None, "landscape") == "1536x1024"

    def test_vertical_maps_to_1024x1792(self):
        from generate_image import resolve_size

        assert resolve_size(None, "vertical") == "1024x1792"

    def test_wide_maps_to_1792x1024(self):
        from generate_image import resolve_size

        assert resolve_size(None, "wide") == "1792x1024"

    def test_1_1_maps_to_1024x1024(self):
        from generate_image import resolve_size

        assert resolve_size(None, "1:1") == "1024x1024"

    def test_2_3_maps_to_1024x1536(self):
        from generate_image import resolve_size

        assert resolve_size(None, "2:3") == "1024x1536"

    def test_3_2_maps_to_1536x1024(self):
        from generate_image import resolve_size

        assert resolve_size(None, "3:2") == "1536x1024"

    def test_9_16_maps_to_1024x1792(self):
        from generate_image import resolve_size

        assert resolve_size(None, "9:16") == "1024x1792"

    def test_16_9_maps_to_1792x1024(self):
        from generate_image import resolve_size

        assert resolve_size(None, "16:9") == "1792x1024"

    def test_unknown_ratio_raises_value_error(self):
        from generate_image import resolve_size

        with pytest.raises(ValueError, match="Unknown ratio"):
            resolve_size(None, "bad")


# ── parse_args — --ratio / --aspect-ratio option ──────────────────────────────

class TestParseArgsRatio:
    def test_ratio_long_form(self):
        from generate_image import parse_args

        args = parse_args(["--prompt", "test", "--ratio", "square"])
        assert args.ratio == "square"

    def test_aspect_ratio_alias(self):
        from generate_image import parse_args

        args = parse_args(["--prompt", "test", "--aspect-ratio", "portrait"])
        assert args.ratio == "portrait"

    def test_ratio_defaults_to_none(self):
        from generate_image import parse_args

        args = parse_args(["--prompt", "test"])
        assert args.ratio is None

    def test_ratio_accepts_all_choices(self):
        from generate_image import parse_args

        choices = ["square", "portrait", "landscape", "vertical", "wide",
                   "1:1", "2:3", "3:2", "9:16", "16:9"]
        for choice in choices:
            args = parse_args(["--prompt", "x", "--ratio", choice])
            assert args.ratio == choice

    def test_invalid_ratio_choice_exits(self):
        from generate_image import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--prompt", "x", "--ratio", "invalid"])

    def test_size_default_is_none(self):
        from generate_image import parse_args

        args = parse_args(["--prompt", "test"])
        assert args.size is None


# ── build_metadata — ratio field ──────────────────────────────────────────────

class TestBuildMetadataRatio:
    def test_includes_ratio_when_provided(self):
        from generate_image import build_metadata

        meta = build_metadata(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1536",
            quality="medium",
            n=1,
            paths=[],
            timestamp="20260101-120000",
            ratio="portrait",
        )
        assert meta["ratio"] == "portrait"

    def test_excludes_ratio_when_none(self):
        from generate_image import build_metadata

        meta = build_metadata(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1024",
            quality="medium",
            n=1,
            paths=[],
            timestamp="20260101-120000",
        )
        assert "ratio" not in meta

    def test_size_present_alongside_ratio(self):
        from generate_image import build_metadata

        meta = build_metadata(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1536",
            quality="medium",
            n=1,
            paths=[],
            timestamp="20260101-120000",
            ratio="portrait",
        )
        assert meta["size"] == "1024x1536"
        assert meta["ratio"] == "portrait"


# ── run() — ratio integration ─────────────────────────────────────────────────

class TestRunRatio:
    def test_generate_receives_resolved_size_from_ratio(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        run(client, "gpt-image-1", "a dog", "1024x1536", "medium", 1,
            tmp_path, "ts", ratio="portrait")
        kw = client.images.generate.call_args.kwargs
        assert kw["size"] == "1024x1536"

    def test_metadata_includes_ratio_when_provided(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        meta = run(client, "gpt-image-1", "a dog", "1024x1536", "medium", 1,
                   tmp_path, "ts", ratio="portrait")
        assert meta["ratio"] == "portrait"
        assert meta["size"] == "1024x1536"

    def test_metadata_excludes_ratio_when_not_provided(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")
        assert "ratio" not in meta

    def test_edit_receives_resolved_size_from_ratio(self, tmp_path):
        from generate_image import run

        client = _mock_edit_client()
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"REF")
        outdir = tmp_path / "out"
        run(client, "gpt-image-1", "a dog", "1792x1024", "medium", 1,
            outdir, "ts", reference_image=str(ref), ratio="wide")
        kw = client.images.edit.call_args.kwargs
        assert kw["size"] == "1792x1024"


# ── validate_edit_ratio_compat — edit mode rejects unsupported ratios ──────────

_EDIT_UNSUPPORTED_RATIOS = ["vertical", "9:16", "wide", "16:9"]
_EDIT_SUPPORTED_RATIOS = ["square", "portrait", "landscape", "1:1", "2:3", "3:2"]


class TestValidateEditRatioCompat:
    """validate_edit_ratio_compat(ratio, explicit_size, reference_image) raises
    SystemExit when edit mode is active (reference_image set), no explicit size
    override is given, and the ratio resolves to a size unsupported by the edit API.
    """

    def _call(self, ratio, explicit_size, reference_image):
        from generate_image import validate_edit_ratio_compat
        validate_edit_ratio_compat(ratio, explicit_size, reference_image)

    # --- ratios that must be rejected in edit mode ---

    def test_vertical_rejected_in_edit_mode(self):
        with pytest.raises(SystemExit):
            self._call("vertical", None, "ref.png")

    def test_9_16_rejected_in_edit_mode(self):
        with pytest.raises(SystemExit):
            self._call("9:16", None, "ref.png")

    def test_wide_rejected_in_edit_mode(self):
        with pytest.raises(SystemExit):
            self._call("wide", None, "ref.png")

    def test_16_9_rejected_in_edit_mode(self):
        with pytest.raises(SystemExit):
            self._call("16:9", None, "ref.png")

    # --- error message must be informative ---

    def test_rejection_message_names_the_ratio(self):
        with pytest.raises(SystemExit) as exc_info:
            self._call("vertical", None, "ref.png")
        assert "vertical" in str(exc_info.value).lower() or "vertical" in str(exc_info.value)

    def test_rejection_message_mentions_edit_mode(self):
        with pytest.raises(SystemExit) as exc_info:
            self._call("wide", None, "ref.png")
        msg = str(exc_info.value).lower()
        assert "edit" in msg or "reference" in msg

    # --- ratios that must be accepted in edit mode ---

    def test_square_accepted_in_edit_mode(self):
        self._call("square", None, "ref.png")  # must not raise

    def test_portrait_accepted_in_edit_mode(self):
        self._call("portrait", None, "ref.png")

    def test_landscape_accepted_in_edit_mode(self):
        self._call("landscape", None, "ref.png")

    def test_1_1_accepted_in_edit_mode(self):
        self._call("1:1", None, "ref.png")

    def test_2_3_accepted_in_edit_mode(self):
        self._call("2:3", None, "ref.png")

    def test_3_2_accepted_in_edit_mode(self):
        self._call("3:2", None, "ref.png")

    # --- explicit --size overrides ratio: no rejection even for unsupported ratios ---

    def test_explicit_size_overrides_vertical_ratio(self):
        self._call("vertical", "1024x1024", "ref.png")  # must not raise

    def test_explicit_size_overrides_wide_ratio(self):
        self._call("wide", "1536x1024", "ref.png")  # must not raise

    def test_explicit_size_overrides_9_16_ratio(self):
        self._call("9:16", "1024x1024", "ref.png")  # must not raise

    def test_explicit_size_overrides_16_9_ratio(self):
        self._call("16:9", "1024x1536", "ref.png")  # must not raise

    # --- no reference image: text-only generation, all ratios allowed ---

    def test_vertical_allowed_in_text_only_mode(self):
        self._call("vertical", None, None)  # must not raise

    def test_wide_allowed_in_text_only_mode(self):
        self._call("wide", None, None)  # must not raise

    def test_9_16_allowed_in_text_only_mode(self):
        self._call("9:16", None, None)  # must not raise

    def test_16_9_allowed_in_text_only_mode(self):
        self._call("16:9", None, None)  # must not raise

    # --- no ratio set: nothing to validate ---

    def test_no_ratio_in_edit_mode_is_allowed(self):
        self._call(None, None, "ref.png")  # must not raise

    def test_no_ratio_no_reference_image_is_allowed(self):
        self._call(None, None, None)  # must not raise


# ── extract_usage ─────────────────────────────────────────────────────────────

class TestExtractUsage:
    """extract_usage() pulls token counts from an API result object or dict."""

    def test_returns_none_when_no_usage_attribute(self):
        from generate_image import extract_usage

        result = SimpleNamespace()  # no .usage attr
        assert extract_usage(result) is None

    def test_returns_none_when_usage_attribute_is_none(self):
        from generate_image import extract_usage

        result = SimpleNamespace(usage=None)
        assert extract_usage(result) is None

    def test_extracts_input_tokens_from_attribute_style(self):
        from generate_image import extract_usage

        result = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        ))
        usage = extract_usage(result)
        assert usage is not None
        assert usage["input_tokens"] == 100

    def test_extracts_output_tokens(self):
        from generate_image import extract_usage

        result = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        ))
        assert extract_usage(result)["output_tokens"] == 200

    def test_extracts_text_tokens_from_details(self):
        from generate_image import extract_usage

        result = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        ))
        assert extract_usage(result)["input_text_tokens"] == 80

    def test_extracts_image_tokens_from_details(self):
        from generate_image import extract_usage

        result = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        ))
        assert extract_usage(result)["input_image_tokens"] == 20

    def test_handles_dict_style_usage(self):
        from generate_image import extract_usage

        result = {"data": [], "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150}}
        usage = extract_usage(result)
        assert usage is not None
        assert usage["input_tokens"] == 50

    def test_returns_none_text_and_image_tokens_when_details_absent(self):
        from generate_image import extract_usage

        result = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
        ))
        usage = extract_usage(result)
        assert usage is not None
        assert usage["input_text_tokens"] is None
        assert usage["input_image_tokens"] is None


# ── estimate_cost ─────────────────────────────────────────────────────────────

_TEXT_PRICE_PER_TOKEN = 5.00 / 1_000_000
_IMG_INPUT_PRICE_PER_TOKEN = 8.00 / 1_000_000


class TestEstimateCost:
    """estimate_cost() is a pure function — no live API calls needed."""

    def _zero_input_usage(self) -> dict[str, Any]:
        """Usage that zeros out input costs; output_tokens=None so table pricing applies."""
        return {"input_text_tokens": 0, "input_image_tokens": 0, "output_tokens": None}

    def test_currency_is_usd(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "low", 1, "x")
        assert cost["currency"] == "USD"

    def test_required_fields_present(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog")
        for field in ("estimated", "currency", "total_usd", "input_text_tokens",
                      "input_image_tokens", "output_image_tokens", "notes"):
            assert field in cost, f"cost missing field: {field}"

    def test_1024x1024_low_output_cost(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "low", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.006)

    def test_1024x1024_medium_output_cost(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.053)

    def test_1024x1024_high_output_cost(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "high", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.211)

    def test_1024x1536_low_output_cost(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1536", "low", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.005)

    def test_1024x1536_medium_output_cost(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1536", "medium", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.041)

    def test_1024x1536_high_output_cost(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1536", "high", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.165)

    def test_1536x1024_medium_output_cost(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1536x1024", "medium", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.041)

    def test_1024x1792_uses_estimated_closest_pricing(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1792", "low", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.005)
        assert cost["estimated"] is True

    def test_1792x1024_uses_estimated_closest_pricing(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1792x1024", "low", 1, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(0.005)
        assert cost["estimated"] is True

    def test_n_multiplies_output_cost(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 3, "x",
                             usage=self._zero_input_usage())
        assert cost["total_usd"] == pytest.approx(3 * 0.053)

    def test_estimated_true_when_no_usage(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog")
        assert cost["estimated"] is True

    def test_estimated_false_when_usage_provided_and_standard_size(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 10, "input_image_tokens": 0, "output_tokens": 0}
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog",
                             usage=usage)
        assert cost["estimated"] is False

    def test_text_tokens_estimated_from_4_char_prompt(self):
        from generate_image import estimate_cost

        # "cats" = 4 chars → ceil(4/4) = 1 token
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "cats")
        assert cost["input_text_tokens"] == 1

    def test_text_tokens_estimated_from_12_char_prompt(self):
        from generate_image import estimate_cost

        # "twelve chars" = 12 chars → ceil(12/4) = 3 tokens
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "twelve chars")
        assert cost["input_text_tokens"] == 3

    def test_text_tokens_from_usage_when_present(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 42, "input_image_tokens": 0, "output_tokens": 0}
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "any prompt",
                             usage=usage)
        assert cost["input_text_tokens"] == 42

    def test_image_input_tokens_zero_without_reference_image(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog")
        assert cost["input_image_tokens"] == 0

    def test_image_input_tokens_estimated_1024_with_reference_image_no_usage(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog",
                             reference_image="/tmp/ref.png")
        assert cost["input_image_tokens"] == 1024

    def test_image_input_tokens_from_usage(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 5, "input_image_tokens": 256, "output_tokens": 0}
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog",
                             usage=usage)
        assert cost["input_image_tokens"] == 256

    def test_output_image_tokens_none_when_no_usage(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog")
        assert cost["output_image_tokens"] is None

    def test_output_image_tokens_from_usage(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 5, "input_image_tokens": 0, "output_tokens": 500}
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog",
                             usage=usage)
        assert cost["output_image_tokens"] == 500

    def test_text_input_cost_included_in_total(self):
        from generate_image import estimate_cost

        # 10 text tokens at $5/1M; zero image input; output_tokens=None → table pricing
        usage = {"input_text_tokens": 10, "input_image_tokens": 0, "output_tokens": None}
        cost = estimate_cost("gpt-image-1", "1024x1024", "low", 1, "x", usage=usage)
        expected = 10 * _TEXT_PRICE_PER_TOKEN + 0.006
        assert cost["total_usd"] == pytest.approx(expected)

    def test_image_input_cost_included_in_total(self):
        from generate_image import estimate_cost

        # 100 image tokens at $8/1M; zero text; output_tokens=None → table pricing
        usage = {"input_text_tokens": 0, "input_image_tokens": 100, "output_tokens": None}
        cost = estimate_cost("gpt-image-1", "1024x1024", "low", 1, "x", usage=usage)
        expected = 100 * _IMG_INPUT_PRICE_PER_TOKEN + 0.006
        assert cost["total_usd"] == pytest.approx(expected)

    def test_notes_is_a_list(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog")
        assert isinstance(cost["notes"], list)

    def test_notes_mention_estimation_when_no_usage(self):
        from generate_image import estimate_cost

        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog")
        notes_text = " ".join(cost["notes"]).lower()
        assert "token" in notes_text or "estimat" in notes_text

    def test_quality_auto_gives_same_total_as_medium(self):
        from generate_image import estimate_cost

        cost_auto = estimate_cost("gpt-image-1", "1024x1024", "auto", 1, "x",
                                  usage=self._zero_input_usage())
        cost_medium = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "x",
                                    usage=self._zero_input_usage())
        assert cost_auto["total_usd"] == pytest.approx(cost_medium["total_usd"])


# ── build_metadata — cost and usage fields ────────────────────────────────────

class TestBuildMetadataCost:
    """build_metadata() must always include a 'cost' field and optionally 'usage'."""

    def _meta(self, **kwargs: Any) -> dict[str, Any]:
        from generate_image import build_metadata

        defaults: dict[str, Any] = dict(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1024",
            quality="medium",
            n=1,
            paths=[],
            timestamp="20260101-120000",
        )
        defaults.update(kwargs)
        return build_metadata(**defaults)

    def test_cost_field_is_present(self):
        assert "cost" in self._meta()

    def test_cost_has_required_fields(self):
        cost = self._meta()["cost"]
        for field in ("estimated", "currency", "total_usd", "input_text_tokens",
                      "input_image_tokens", "output_image_tokens", "notes"):
            assert field in cost, f"cost missing field: {field}"

    def test_cost_currency_is_usd(self):
        assert self._meta()["cost"]["currency"] == "USD"

    def test_cost_estimated_when_no_usage(self):
        assert self._meta()["cost"]["estimated"] is True

    def test_cost_not_estimated_when_usage_provided(self):
        usage = {"input_text_tokens": 10, "input_image_tokens": 0, "output_tokens": 0}
        meta = self._meta(usage=usage)
        assert meta["cost"]["estimated"] is False

    def test_raw_usage_in_metadata_when_provided(self):
        usage = {"input_text_tokens": 10, "input_image_tokens": 0, "output_tokens": 500}
        meta = self._meta(usage=usage)
        assert "usage" in meta
        assert meta["usage"] == usage

    def test_usage_absent_from_metadata_when_not_provided(self):
        assert "usage" not in self._meta()

    def test_accepts_usage_kwarg_without_raising(self):
        from generate_image import build_metadata

        build_metadata(
            model="gpt-image-1",
            prompt="a dog",
            size="1024x1024",
            quality="medium",
            n=1,
            paths=[],
            timestamp="ts",
            usage={"input_text_tokens": 5, "input_image_tokens": 0, "output_tokens": 0},
        )


# ── run() — usage capture and cost in metadata ────────────────────────────────

class TestRunUsageCost:
    """run() must capture usage from the API result and embed cost in metadata."""

    def test_metadata_has_cost_field(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")
        assert "cost" in meta

    def test_cost_estimated_when_api_returns_no_usage(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()  # result.usage = None
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")
        assert meta["cost"]["estimated"] is True

    def test_cost_not_estimated_when_api_provides_usage(self, tmp_path):
        from generate_image import run

        usage_ns = SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        )
        client = _mock_generate_client_with_usage(usage_ns)
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")
        assert meta["cost"]["estimated"] is False

    def test_metadata_includes_raw_usage_when_api_provides_it(self, tmp_path):
        from generate_image import run

        usage_ns = SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        )
        client = _mock_generate_client_with_usage(usage_ns)
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")
        assert "usage" in meta
        assert meta["usage"]["normalized"]["input_tokens"] == 100
        assert meta["usage"]["normalized"]["input_text_tokens"] == 80

    def test_metadata_no_usage_key_when_api_has_no_usage(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()  # result.usage = None
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")
        assert "usage" not in meta

    def test_cost_total_usd_positive_float(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")
        assert isinstance(meta["cost"]["total_usd"], float)
        assert meta["cost"]["total_usd"] > 0

    def test_cost_saved_in_metadata_json(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()
        run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
            tmp_path, "ts")
        saved = json.loads((tmp_path / "metadata.json").read_text())
        assert "cost" in saved
        assert saved["cost"]["currency"] == "USD"


# ── decode_image — dict-shaped items (blocker 4) ──────────────────────────────

class TestDecodeImageDictItem:
    def test_decodes_dict_item_with_b64_json(self):
        from generate_image import decode_image

        raw = b"png data"
        item = {"b64_json": base64.b64encode(raw).decode()}
        assert decode_image(item) == raw

    def test_raises_when_dict_item_has_no_b64_json(self):
        from generate_image import decode_image

        with pytest.raises(ValueError, match="No base64"):
            decode_image({"url": "http://example.com/img.png"})


# ── extract_usage — nested dict input_tokens_details (blocker 5) ──────────────

class TestExtractUsageNestedDict:
    def test_nested_dict_details_text_tokens(self):
        from generate_image import extract_usage

        result = {
            "data": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "input_tokens_details": {"text_tokens": 60, "image_tokens": 40},
            },
        }
        usage = extract_usage(result)
        assert usage is not None
        assert usage["input_text_tokens"] == 60

    def test_nested_dict_details_image_tokens(self):
        from generate_image import extract_usage

        result = {
            "data": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "input_tokens_details": {"text_tokens": 60, "image_tokens": 40},
            },
        }
        usage = extract_usage(result)
        assert usage["input_image_tokens"] == 40

    def test_nested_dict_all_top_level_fields(self):
        from generate_image import extract_usage

        result = {
            "usage": {
                "input_tokens": 50,
                "output_tokens": 100,
                "total_tokens": 150,
                "input_tokens_details": {"text_tokens": 30, "image_tokens": 20},
            }
        }
        usage = extract_usage(result)
        assert usage["input_tokens"] == 50
        assert usage["output_tokens"] == 100
        assert usage["total_tokens"] == 150


# ── estimate_cost — output token pricing (blockers 1, 2, 5) ──────────────────

_OUTPUT_IMAGE_PRICE_PER_TOKEN = 30.00 / 1_000_000


class TestEstimateCostOutputTokenPricing:
    """Output cost uses $30/1M token pricing when usage.output_tokens is present."""

    def test_output_cost_uses_30_per_million_when_output_tokens_in_usage(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 0, "input_image_tokens": 0, "output_tokens": 1000}
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "x", usage=usage)
        expected = 1000 * _OUTPUT_IMAGE_PRICE_PER_TOKEN
        assert cost["total_usd"] == pytest.approx(expected)

    def test_output_cost_uses_table_when_output_tokens_none_in_usage(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 0, "input_image_tokens": 0, "output_tokens": None}
        cost = estimate_cost("gpt-image-1", "1024x1024", "low", 1, "x", usage=usage)
        assert cost["total_usd"] == pytest.approx(0.006)

    def test_output_cost_uses_table_when_usage_absent(self):
        from generate_image import estimate_cost

        # No usage at all → table pricing; estimated should be True
        cost = estimate_cost("gpt-image-1", "1024x1024", "low", 1, "x")
        assert cost["estimated"] is True

    def test_estimated_true_when_output_tokens_absent_from_usage(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 10, "input_image_tokens": 0, "output_tokens": None}
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog", usage=usage)
        assert cost["estimated"] is True

    def test_estimated_false_when_all_tokens_from_usage(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 10, "input_image_tokens": 5, "output_tokens": 200}
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "a dog", usage=usage)
        assert cost["estimated"] is False

    def test_estimated_true_when_approx_size_even_with_full_usage(self):
        from generate_image import estimate_cost

        usage = {"input_text_tokens": 10, "input_image_tokens": 0, "output_tokens": 200}
        cost = estimate_cost("gpt-image-1", "1024x1792", "medium", 1, "a dog", usage=usage)
        assert cost["estimated"] is True

    def test_estimated_true_when_reference_image_no_usage(self):
        from generate_image import estimate_cost

        cost = estimate_cost(
            "gpt-image-1", "1024x1024", "medium", 1, "edit this",
            reference_image="/tmp/ref.png",
        )
        assert cost["estimated"] is True

    def test_output_tokens_zero_uses_token_pricing_giving_zero_output_cost(self):
        from generate_image import estimate_cost

        # output_tokens=0 is present (not None) → token pricing → output cost = $0
        usage = {"input_text_tokens": 0, "input_image_tokens": 0, "output_tokens": 0}
        cost = estimate_cost("gpt-image-1", "1024x1024", "medium", 1, "x", usage=usage)
        assert cost["total_usd"] == pytest.approx(0.0)
        assert cost["estimated"] is False


# ── run() — usage persisted as raw + normalized (blocker 3) ──────────────────

class TestRunUsagePersistence:
    def test_usage_stored_as_raw_and_normalized(self, tmp_path):
        from generate_image import run

        usage_ns = SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        )
        client = _mock_generate_client_with_usage(usage_ns)
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")

        assert "usage" in meta
        assert "raw" in meta["usage"]
        assert "normalized" in meta["usage"]

    def test_usage_normalized_contains_token_fields(self, tmp_path):
        from generate_image import run

        usage_ns = SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        )
        client = _mock_generate_client_with_usage(usage_ns)
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")

        norm = meta["usage"]["normalized"]
        assert norm["input_tokens"] == 100
        assert norm["output_tokens"] == 200
        assert norm["input_text_tokens"] == 80
        assert norm["input_image_tokens"] == 20

    def test_usage_raw_is_json_serializable_and_contains_input_tokens(self, tmp_path):
        from generate_image import run

        usage_ns = SimpleNamespace(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_tokens_details=SimpleNamespace(text_tokens=80, image_tokens=20),
        )
        client = _mock_generate_client_with_usage(usage_ns)
        run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
            tmp_path, "ts")

        saved = json.loads((tmp_path / "metadata.json").read_text())
        assert saved["usage"]["raw"]["input_tokens"] == 100

    def test_usage_absent_when_api_returns_no_usage(self, tmp_path):
        from generate_image import run

        client = _mock_generate_client()  # result.usage = None
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")
        assert "usage" not in meta

    def test_dict_shaped_result_extracts_usage_and_saves_images(self, tmp_path):
        from generate_image import run

        image_bytes = b"PNG_BINARY"
        b64 = base64.b64encode(image_bytes).decode()
        dict_result = {
            "data": [{"b64_json": b64}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 100,
                "total_tokens": 150,
            },
        }
        client = MagicMock()
        client.images.generate.return_value = dict_result
        meta = run(client, "gpt-image-1", "a cat", "1024x1024", "medium", 1,
                   tmp_path, "ts")

        assert len(meta["paths"]) == 1
        assert Path(meta["paths"][0]).read_bytes() == image_bytes
        assert "usage" in meta
        assert meta["usage"]["normalized"]["input_tokens"] == 50

    def test_cost_uses_output_tokens_from_usage_when_available(self, tmp_path):
        from generate_image import run

        usage_ns = SimpleNamespace(
            input_tokens=10,
            output_tokens=500,
            total_tokens=510,
            input_tokens_details=SimpleNamespace(text_tokens=10, image_tokens=0),
        )
        client = _mock_generate_client_with_usage(usage_ns)
        meta = run(client, "gpt-image-1", "a dog", "1024x1024", "medium", 1,
                   tmp_path, "ts")

        expected_output_cost = 500 * _OUTPUT_IMAGE_PRICE_PER_TOKEN
        expected_text_cost = 10 * _TEXT_PRICE_PER_TOKEN
        assert meta["cost"]["total_usd"] == pytest.approx(
            expected_text_cost + expected_output_cost
        )
