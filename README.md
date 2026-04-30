# OpenAI Image Generator

A small Python project that uses the OpenAI Images API to generate images from text prompts and save the results locally.

## Features

- Text-to-image generation with OpenAI image models
- Optional reference-image editing mode with `--reference-image` / `--image`
- ChatGPT-like aspect ratio presets with `--ratio` / `--aspect-ratio`
- Saves generated images to `outputs/`
- Saves generation metadata (including ratio when used) next to each image
- CLI options for prompt, size, quality, model, number of images, reference image, and ratio
- Safe `.env.example` template for local API key setup

## Requirements

- Python 3.10+
- An OpenAI API key

> Note: ChatGPT subscription billing is separate from OpenAI API billing. You need an API key with billing enabled on the OpenAI platform.

## Quick start

```bash
git clone https://github.com/jianghuancn/openai-image-generator.git
cd openai-image-generator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your API key:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Generate an image with the default square size:

```bash
python generate_image.py \
  --prompt "A cinematic product photo of a futuristic AI camera, neon reflections, premium advertising style" \
  --quality medium
```

Use a ChatGPT-style aspect ratio preset (portrait — 1024×1536):

```bash
python generate_image.py \
  --prompt "A tall portrait of a cinematic character, dramatic lighting, bokeh background" \
  --ratio portrait \
  --quality high
```

Generate a widescreen banner (16:9 — 1792×1024):

```bash
python generate_image.py \
  --prompt "A minimalist hero banner for a SaaS landing page, soft gradient, glassmorphism" \
  --ratio 16:9
```

Generate a vertical mobile wallpaper (9:16 — 1024×1792):

```bash
python generate_image.py \
  --prompt "A vertical phone wallpaper, neon city night, anime style" \
  --aspect-ratio 9:16
```

Use `--size` for a precise custom size (overrides `--ratio` if both are given):

```bash
python generate_image.py \
  --prompt "A square product thumbnail" \
  --size 1024x1024
```

Use a reference image for edit / variation workflows:

```bash
python generate_image.py \
  --prompt "Turn this into a polished Douyin thumbnail with dramatic lighting, clean title space, and high contrast" \
  --reference-image ./reference.png \
  --ratio landscape \
  --quality medium
```

`--image` is a shorter alias for `--reference-image`.

Generated files are saved under `outputs/<timestamp>/`.

## CLI usage

```bash
python generate_image.py --help
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt` | _(required)_ | Text prompt for image generation |
| `--model` | `gpt-image-1` | OpenAI image model |
| `--ratio`, `--aspect-ratio` | `None` | Aspect ratio preset (see table below). Ignored when `--size` is set. |
| `--size` | `None` | Explicit OpenAI size string, e.g. `1024x1024`. Overrides `--ratio`. Falls back to `1024x1024` if neither is set. |
| `--quality` | `medium` | `low`, `medium`, `high`, or `auto` |
| `--n` | `1` | Number of images to generate |
| `--reference-image`, `--image` | `None` | Local image path for OpenAI image edit mode |
| `--outdir` | `outputs/<timestamp>` | Output folder |

### Aspect ratio presets (`--ratio`)

These match the ChatGPT Create Image options:

| Preset | Alias | OpenAI size | Use case | Edit mode? |
|--------|-------|-------------|----------|------------|
| `square` | `1:1` | `1024×1024` | Profile photos, thumbnails | Yes |
| `portrait` | `2:3` | `1024×1536` | Character art, posters | Yes |
| `landscape` | `3:2` | `1536×1024` | Banners, panoramics | Yes |
| `vertical` | `9:16` | `1024×1792` | Mobile wallpapers, TikTok/Reels | Generation only |
| `wide` | `16:9` | `1792×1024` | YouTube thumbnails, hero images | Generation only |

When `--ratio` is used the `metadata.json` file includes both `size` (the resolved OpenAI string) and `ratio` (the preset name) for traceability.

> **Note — reference-image edit mode:** OpenAI's image edit API only accepts `1024×1024`, `1024×1536`, and `1536×1024`. The `vertical` (`9:16`) and `wide` (`16:9`) presets are **generation-only** and will produce a clear CLI error when combined with `--reference-image`. To use a specific size in edit mode, pass it explicitly with `--size` (e.g. `--reference-image ref.png --size 1024x1024 --ratio wide` is valid — `--size` takes precedence).

## Example prompts

```text
A premium Douyin thumbnail for an AI livestream repurposing startup, neon studio, realistic Chinese creator, clean title space, high contrast, no watermark.
```

```text
A minimalist landing-page hero image for an AI image generation app, glassmorphism UI, soft gradient background, cinematic lighting, no text.
```

## Security

Never commit `.env` or your real API key. This repo includes `.gitignore` to keep secrets and generated outputs out of Git.

## License

MIT
