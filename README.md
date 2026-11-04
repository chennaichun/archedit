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

## Deploy

The repo ships a `Dockerfile` and a Render Blueprint (`render.yaml`).

1. Push the repo to GitHub.
2. On [Render](https://render.com): **New → Blueprint**, pick the repo.
3. Enter `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` when asked (optionally `ANTHROPIC_API_KEY`). Every push to `main` redeploys.

Any host that runs a Docker image works the same way; the container listens on `$PORT`.

Public-deployment guards, all set in the Dockerfile and overridable with env vars:

| Variable | Default in Docker | Why |
|---|---|---|
| `DAILY_LIMIT_PER_VISITOR` | 10 | One visitor can't use up the shared free allowance |
| `DAILY_LIMIT_TOTAL` | 80 | Stay inside Cloudflare's ~90 free klein-4B images per day |
| `OUTPUT_TTL_HOURS` | 24 | Generated images are deleted after a day |
| `SECURE_COOKIES` | 1 | Visitor cookie only over HTTPS |

Each browser gets an anonymous `HttpOnly` cookie, so visitors only see their own history. Requests with a fixed seed are cached, so repeating one costs nothing. Cloudflare 5xx errors and timeouts are retried twice with exponential backoff; 4xx errors are not retried. Limits, cache and history live in memory and on local disk, which fits a single small instance.

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
