"""Image models and generation modes.

Free-use estimates assume Cloudflare's 10,000 free Neurons per day and a
1024x1024 output (4 tiles) with one 512px reference image (1 tile):
  klein-4b: 4*26.05 + 1*5.37            ~= 110 Neurons  -> ~90 images/day
  klein-9b: 1363.64 + 181.82 (1 input)  ~= 1545 Neurons -> ~6 images/day
  dev (20 steps): 20*(4*37.5 + 1*18.75) ~= 3375 Neurons -> ~2 images/day
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImageModel:
    key: str
    model_id: str
    label: str
    adjustable_steps: bool
    free_per_day: str
    description: str


MODELS: dict[str, ImageModel] = {
    m.key: m
    for m in [
        ImageModel(
            "klein-4b",
            "@cf/black-forest-labs/flux-2-klein-4b",
            "FLUX.2 [klein] 4B",
            False,
            "~90",
            "Fastest, most free images. Fixed 4 steps.",
        ),
        ImageModel(
            "klein-9b",
            "@cf/black-forest-labs/flux-2-klein-9b",
            "FLUX.2 [klein] 9B",
            False,
            "~6",
            "Better detail than 4B. Fixed 4 steps.",
        ),
        ImageModel(
            "dev",
            "@cf/black-forest-labs/flux-2-dev",
            "FLUX.2 [dev]",
            True,
            "~2",
            "Best quality, slowest. Adjustable steps.",
        ),
    ]
}


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    description: str


MODES: dict[str, Mode] = {
    m.key: m
    for m in [
        Mode("edit", "Restyle", "Uses the photo as reference: keeps the layout, changes style, materials and light."),
        Mode("edges", "Line art", "Traces the photo into line art and redraws from it."),
        Mode("text", "Text", "No reference image, prompt only."),
        Mode("local", "Local", "Brush an area on the photo; only that area changes."),
    ]
}

# FLUX.2 works on any size 256-1920; these are ~1 MP presets.
SIZES = ["1024x1024", "1216x832", "832x1216", "1344x768", "768x1344", "1152x896", "896x1152"]


def public_catalog() -> dict:
    return {
        "models": [asdict(m) for m in MODELS.values()],
        "modes": [asdict(m) for m in MODES.values()],
        "sizes": SIZES,
    }
