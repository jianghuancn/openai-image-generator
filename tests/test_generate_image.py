"""Tests for generate_image module — written first (TDD RED phase)."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _b64_item(raw: bytes) -> SimpleNamespace:
    """Fake API response item with b64_json set."""
    return SimpleNamespace(b64_json=base64.b64encode(raw).decode())


def _mock_generate_client(raw: bytes = b"fake_png") -> MagicMock:
    client = MagicMock()
    client.images.generate.return_value = MagicMock(data=[_b64_item(raw)])
    return client


def _mock_edit_client(raw: bytes = b"edited_png") -> MagicMock:
    client = MagicMock()
    client.images.edit.return_value = MagicMock(data=[_b64_item(raw)])
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
