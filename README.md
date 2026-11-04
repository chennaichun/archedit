# ControlNet Studio

Upload a photo, then restyle it, redraw it from line art, or change one brushed area with FLUX.2. Images are generated on Cloudflare Workers AI's free daily allowance, so no GPU is needed. Claude can write the prompt for you (optional).

## Cost

| Service | Free? | Notes |
|---|---|---|
| Cloudflare Workers AI | 10,000 Neurons per day | FLUX.2 [klein] 4B ≈ 90 images/day, klein 9B ≈ 6, [dev] ≈ 2 (estimates for 1024×1024 with one reference image). Resets daily at 00:00 UTC. |
| Claude API | Paid | Only used by the "Write prompt with Claude" button. You can always type the prompt yourself. |

## Setup

1. Sign up at Cloudflare (free) and open **AI → Workers AI → REST API**.
2. Copy the **Account ID**, then **Create a Workers AI API Token** and copy the token.
3. Put both in `.env` (copy `.env.example` if it does not exist yet):

```
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
```

## Run

Double-click **start.command**. It installs packages on the first run, starts the server and opens http://localhost:8765.

Or from a terminal:

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/uvicorn app.main:app --port 8765
```

## Modes

- **Restyle**: the photo is the reference; layout stays, style, materials and light change.
- **Line art**: traces the photo into line art, then redraws from it.
- **Text**: no reference image.
- **Local**: brush an area and only that area changes. The server crops around it, sends the crop to FLUX.2 (klein 4B by default), and pastes the result back through a feathered mask, so pixels outside the brushed area stay identical. **Edit result** lets you continue with the next change.
- **Style image**: optionally add a second image as a style reference.

The app opens with the bedroom photo from the original [StableDiffusionWithGradio](https://github.com/chennaichun/StableDiffusionWithGradio) repo (`examples/`).

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest
```

Tests use the real photos in `examples/`; only the Cloudflare and Claude responses are faked, so they run offline without keys.

## Layout

```
app/main.py        HTTP server
app/cloudflare.py  FLUX.2 on Cloudflare Workers AI
app/llm.py         Claude prompt writer
app/preprocess.py  cropping, line art, local-edit crop and paste-back
app/catalog.py     models, modes, sizes
static/            web UI
examples/          photos from the original repo
start.command      double-click launcher
```
