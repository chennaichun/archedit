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
        Mode("edit", "Whole image", "The photo is the reference: layout stays, style, materials and light change."),
        Mode("sketch", "Sketch", "The input is already a line drawing or sketch; it is rendered as drawn."),
        Mode("edges", "Traced photo", "The photo is traced into line art first, then rendered from the lines."),
        Mode("text", "Text", "No reference image, prompt only."),
        Mode("local", "Brushed area", "Brush an area on the photo; only that area changes."),
    ]
}

# FLUX.2 works on any size 256-1920; these are ~1 MP presets.
# "auto" keeps the input image's own aspect ratio at about 1 MP. It is the default
# because multi-round editing feeds each result back in: a fixed size would crop
# the picture a little more every round.
SIZES = ["auto", "1024x1024", "1216x832", "832x1216", "1344x768", "768x1344", "1152x896", "896x1152"]


# ---------------------------------------------------------------------------
# Architecture tools: what the UI offers. Each tool runs on one generation mode
# and comes with presets that fill in a FLUX.2 prompt written for that job.
# ---------------------------------------------------------------------------

KEEP_GEOMETRY = (
    "Keep the architecture, walls, windows, ceiling, floor plan, furniture layout and camera angle "
    "of image 0 exactly the same."
)


@dataclass(frozen=True)
class Preset:
    label: str
    prompt: str
    # True when the preset only makes sense on a brushed area ("remove the object").
    area: bool = False


@dataclass(frozen=True)
class Tool:
    key: str
    label: str
    mode: str
    hint: str
    example: str
    presets: tuple[Preset, ...]
    # Tools whose prompts contain "{target}" let the user pick what is being changed.
    targets: tuple[str, ...] = ()
    # Tools that can run on the whole image or only on a brushed area.
    brushable: bool = False


def _render(scene: str) -> str:
    return (
        f"Turn the architectural line drawing in image 0 into a photorealistic render of {scene}. "
        "Follow every line, wall, opening and object in image 0 exactly, with the same perspective. "
        "Architectural photography, realistic materials, soft natural shadows."
    )


def _material(surface: str) -> str:
    # Naming the target matters: with a vague "replace the surface" the model picked
    # whatever surface it liked (it bricked the rug when asked to brick a wall).
    return (
        f"Change the {{target}} in image 0 to {surface}. Match the existing perspective, scale and lighting, "
        "with realistic texture and subtle wear."
    )


def _light(atmosphere: str) -> str:
    return f"{KEEP_GEOMETRY} Change only the lighting and atmosphere: {atmosphere}. Photorealistic architectural photography."


def _style(style: str) -> str:
    return (
        f"Restyle the interior in image 0 as {style}. Keep the walls, windows, ceiling and camera angle exactly "
        "the same; change the furniture, materials, textiles and decor. Photorealistic interior photography."
    )


TOOLS: dict[str, Tool] = {
    t.key: t
    for t in [
        Tool(
            "edit", "Image edit", "edit",
            "Describe any change. Switch to \"Brushed area\" to change only the part you paint.",
            "sample_input.png",
            (
                Preset("Declutter", (
                    "Declutter image 0: clear loose items from tables, counters and shelves, tidy the bedding "
                    "and cushions. Keep the furniture, architecture and camera angle exactly the same.")),
                Preset("Add plants", (
                    "Add two large leafy indoor plants in simple ceramic pots to image 0, placed naturally on "
                    "the floor, with matching light and shadows.")),
                Preset("Listing photo", (
                    "Make image 0 look like a professional real-estate photo: bright balanced exposure, "
                    "neutral white balance, crisp detail and straight vertical lines. Keep every object, "
                    "the layout and the camera angle exactly the same.")),
                Preset("Remove object", (
                    # Describing what should be there instead works better than "remove X": with only
                    # "remove", klein 4B often returned the object unchanged.
                    "Remove the {target} from image 0 completely. Where it was, show the empty space behind it: "
                    "continue the surrounding wall, floor and furniture surfaces as if the {target} had never "
                    "been there."), area=True),
                Preset("Repaint", (
                    "Repaint the {target} in image 0 in a soft sage green matte finish, keeping its shape, "
                    "texture and shadows."), area=True),
            ),
            targets=("plant", "chair", "lamp", "rug", "TV", "picture frame", "table", "sofa"),
            brushable=True,
        ),
        Tool(
            "sketch", "Sketch to render", "sketch",
            "Upload a line drawing or sketch. Turn on \"Trace a photo\" to start from a photo instead.",
            "line.png",
            (
                Preset("Bright living room", _render(
                    "a bright modern living room with white plaster walls, light oak floor, a low linen sofa, "
                    "recessed ceiling downlights and floor-to-ceiling glazing with daylight")),
                Preset("Concrete gallery", _render(
                    "a minimalist gallery with board-formed concrete walls, polished concrete floor, "
                    "recessed ceiling spotlights and a single sculptural bench")),
                Preset("Hotel lobby", _render(
                    "a warm boutique hotel lobby with travertine walls, walnut joinery, brass details "
                    "and a round upholstered ottoman lit by warm downlights")),
                Preset("Night interior", _render(
                    "a calm living room at night with warm lamp light, dark timber floor and city lights "
                    "visible through the glazing")),
            ),
        ),
        Tool(
            "materials", "Materials", "local",
            "Brush a wall, floor or object, then pick a material.",
            "sample_input.png",
            (
                Preset("Exposed brick", _material("exposed red brick laid in running bond with recessed grey mortar joints")),
                Preset("Board-formed concrete", _material("board-formed architectural concrete with visible timber grain and tie holes")),
                Preset("Light oak", _material("light oak with fine straight grain and a matte oiled finish")),
                Preset("White marble", _material("polished white Carrara marble with soft grey veining")),
                Preset("Terrazzo", _material("terrazzo with white base and small grey, green and terracotta chips")),
                Preset("Lime plaster", _material("warm off-white lime plaster with a soft hand-troweled texture")),
                Preset("Green velvet", _material("deep green velvet fabric with soft folds and sheen")),
            ),
            targets=("wall", "floor", "ceiling", "cabinets", "countertop", "bed", "sofa", "facade"),
        ),
        Tool(
            "lighting", "Lighting & time", "edit",
            "Changes the light of the whole scene and keeps everything else.",
            "sample_input.png",
            (
                Preset("Golden hour", _light("low warm golden-hour sunlight streaming through the windows with long soft shadows")),
                Preset("Blue hour", _light("blue hour dusk outside, all interior lamps and ceiling lights switched on with a warm glow")),
                Preset("Night", _light("night outside, warm interior lighting from lamps and downlights, reflections in the glass")),
                Preset("Overcast", _light("soft even overcast daylight with diffuse shadows and a neutral color balance")),
                Preset("Snowy morning", _light("a bright snowy winter morning outside the windows, cool soft daylight inside")),
            ),
        ),
        Tool(
            "style", "Style & staging", "edit",
            "Restyles or furnishes the room and keeps the architecture.",
            "sample_input.png",
            (
                Preset("Japandi", _style("a Japandi space: light oak, linen in sand and beige, low furniture, paper lamps, a few plants")),
                Preset("Scandinavian", _style("a Scandinavian space: white walls, pale birch, wool textiles, simple lamps, soft daylight")),
                Preset("Industrial loft", _style("an industrial loft: dark steel, worn leather, reclaimed timber, black metal pendant lights")),
                Preset("Mid-century", _style("a mid-century modern space: walnut furniture, mustard and teal accents, sculptural lamps")),
                Preset("Virtual staging", (
                    "Furnish the room in image 0 as a lived-in, styled space ready for a real-estate listing: "
                    "add tasteful furniture, a rug, lamps, plants and artwork. " + KEEP_GEOMETRY.replace(
                        "floor plan, furniture layout and camera angle", "floor plan and camera angle") + " Photorealistic interior photography."
                )),
            ),
        ),
    ]
}


def public_catalog() -> dict:
    return {
        "models": [asdict(m) for m in MODELS.values()],
        "modes": [asdict(m) for m in MODES.values()],
        "tools": [asdict(t) for t in TOOLS.values()],
        "sizes": SIZES,
    }
