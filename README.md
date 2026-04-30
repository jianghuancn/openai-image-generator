# OpenAI Image Generator

A small Python project that uses the OpenAI Images API to generate images from text prompts and save the results locally.

## Features

- Text-to-image generation with OpenAI image models
- Optional reference-image editing mode with `--reference-image` / `--image`
- Saves generated images to `outputs/`
- Saves generation metadata next to each image
- CLI options for prompt, size, quality, model, number of images, and reference image
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

Generate an image:

```bash
python generate_image.py \
  --prompt "A cinematic product photo of a futuristic AI camera, neon reflections, premium advertising style" \
  --size 1024x1024 \
  --quality medium
```

Use a reference image for edit / variation workflows:

```bash
python generate_image.py \
  --prompt "Turn this into a polished Douyin thumbnail with dramatic lighting, clean title space, and high contrast" \
  --reference-image ./reference.png \
  --size 1024x1024 \
  --quality medium
```

`--image` is a shorter alias for `--reference-image`.

Generated files are saved under `outputs/<timestamp>/`.

## CLI usage

```bash
python generate_image.py --help
```

Options:

- `--prompt`: Required text prompt
- `--model`: OpenAI image model, default `gpt-image-1`
- `--size`: Image size, default `1024x1024`
- `--quality`: `low`, `medium`, `high`, or `auto`; default `medium`
- `--n`: Number of images to generate; default `1`
- `--reference-image`, `--image`: Optional local image path. When provided, the script uses OpenAI image edit mode and includes the reference path in `metadata.json`.
- `--outdir`: Output folder; default `outputs/<timestamp>`

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
