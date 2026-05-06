# ComfyUI-Kepri-Nodes-Pack

A lightweight ComfyUI custom-node pack for the **Kepri Background Removal V2** pipeline. Contains two nodes:

1. **`KepriMaskMerge`** — merges multiple detection masks into a single mask.
2. **`KepriImageFinalize`** — resizes by longest edge, crops/pads to a chosen aspect ratio, and composites onto a background (transparent, solid colour, or image preset).

Both nodes use **only `torch`** (already required by ComfyUI) and expose every parameter as a widget so they can be wired to **workflow API inputs** when the graph is executed from Modal / RunPod / Kepri.

---

## Node 1 — KepriMaskMerge

When `GroundingDetector` detects multiple objects (e.g. 2 shoes, 3-piece outfit) it outputs **N masks** `[N, H, W]`. Downstream nodes (`GrowMask`, `MaskBoundingBox+`, `RMBG`) expect a **single mask** `[1, H, W]`.

`KepriMaskMerge` bridges this gap:

- **Union** all detected masks (`max(dim=0)`) so the bounding box englobes every part.
- **Filter noise** by dropping masks whose area is below a configurable % of the image.
- **Graceful fallback**: if zero detection or all masks are noise, returns a full-image mask so the pipeline does not crash.

### Inputs

| Name | Type | Default | Description |
|---|---|---|---|
| `masks` | MASK | — | From `GroundingDetector.masks`. Accepts `[N, H, W]` or `[H, W]`. |
| `min_mask_area_percent` | FLOAT slider | `1.0` | Masks < 1% of image area are discarded as noise. Set `0` to keep everything. |

### Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `merged_mask` | MASK | `[1, H, W]` | Single unified mask ready for downstream nodes. |

---

## Node 2 — KepriImageFinalize

One-stop finalisation after background removal. Replaces the old chain `ImageResizeKJv2 → SaveImage` (or any post-RMBG resize node) with a single node that does three things:

1. **Resize by longest edge** to a fixed pixel count (e.g. `2048`).
2. **Crop / pad** to a target aspect ratio (`1:1`, `4:3`, `16:9`, `2:3` … or keep `original`).
3. **Composite** onto a background:
   - **transparent** — keeps the alpha mask for later use
   - **solid colour** — `#hex` colour exposed via API
   - **image preset** — a reference photo (concrete, wood, marble …) exposed via API

### Inputs

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | Cut-out image from RMBG or any upstream node. `[B, H, W, 3]` |
| `longest_edge` | INT | `2048` | Target size for the longest side (e.g. `2048` for e-commerce). |
| `aspect_ratio` | COMBO | `1:1` | Target crop ratio: `original`, `1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `2:3`, `3:2`, `5:4`, `4:5`. |
| `background_mode` | COMBO | `color` | `transparent` (pass mask through), `color` (solid fill), or `image_preset` (photo background). |
| `background_color` | STRING | `#FFFFFF` | Hex colour used when `background_mode == color`. Exposed to API. |
| `mask` (optional) | MASK | — | Alpha mask from RMBG. Required for `transparent` and `color` modes to composite cleanly. |
| `background_image` (optional) | IMAGE | — | Preset photo (concrete, wood, etc.) used when `background_mode == image_preset`. Resized automatically. |

### Outputs

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Final resized / cropped / composited image. |
| `mask` | MASK | Final alpha mask (all-white if background was opaque, original alpha if transparent). |

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wearekhepri/comfyui-kepri-nodes-pack.git
```

Restart ComfyUI and refresh the browser. Both nodes appear under **Kepri/Background**.

**Zero extra dependencies** — only uses `torch`.

---

## Full V2 Pipeline (step-by-step)

```
LoadImage
    -> ResizeImagesByLongerEdge (2048px)
        -> GroundingDetector (prompt = "shoe"  ← exposed to API)
            -> masks [N, H, W]
                -> [KepriMaskMerge]
                    -> merged_mask [1, H, W]
                        -> GrowMask (+25 px padding)
                            -> MaskBoundingBox+ (crop to bbox)
                                -> RMBG-2.0 (background removal)
                                    -> [KepriImageFinalize]
                                        -> image  (final product photo)
                                        -> mask  (alpha if transparent)
                                            -> SaveImage
```

### Parameters exposed to the Kepri App API

When the workflow is executed via Modal / RunPod / any FastAPI wrapper, these are the variables the app should send per image:

| Parameter | Source in graph | Example values |
|---|---|---|
| `category_prompt` | `GroundingDetector.prompt` | `"shoe"`, `"clothing"`, `"bag"`, `"accessory"` |
| `longest_edge` | `KepriImageFinalize.longest_edge` | `2048` |
| `aspect_ratio` | `KepriImageFinalize.aspect_ratio` | `"1:1"`, `"4:3"`, `"3:4"`, `"original"` |
| `background_mode` | `KepriImageFinalize.background_mode` | `"color"`, `"transparent"`, `"image_preset"` |
| `background_color` | `KepriImageFinalize.background_color` | `"#FFFFFF"`, `"#222222"`, `"#E8DCC8"` |
| `background_image_preset` | `KepriImageFinalize.background_image` | path/URL to preset photo (concrete, wood, marble, linen …) |

---

## Edge-Case Behaviour (KepriMaskMerge)

| Scenario | Result |
|---|---|
| No detection (`N == 0`) | Full-image white mask `[1, H, W]` — pipeline continues, user can review |
| Single detection (`N == 1`) | Passes through (reshaped to `[1, H, W]`) |
| Multiple detections (`N >= 2`) | Union of all masks, noise filtered, englobes every part |
| All masks below noise threshold | Full-image white mask — prevents silent data loss |

---

## Why Union (OR) and Not Intersection (AND)?

A pair of shoes = two separate masks with **no overlap**. Intersection would be empty. Union (`max`) englobes both shoes, so the crop keeps them together.

---

## Testing / Cloud Deployment

Tested and deployed on **ComfyUI Cloud** (Modal, RunPod, ComfyAI). Zero extra dependencies = no Docker image bloat.

To verify registration after install:
1. Start ComfyUI server
2. In the web UI, right-click → `Add Node` → `Kepri/Background`
3. You should see both nodes listed.

---

## File Structure

```
comfyui-kepri-nodes-pack/
├── README.md
├── __init__.py              # ComfyUI registration (imports all nodes)
└── nodes/
    ├── kepri_mask_merge.py    # KepriMaskMerge — multi-object mask union
    └── kepri_image_finalize.py # KepriImageFinalize — resize + crop/pad + background
```

---

## License

MIT — same as ComfyUI-Grounding and ComfyUI_essentials.

---

## Related Projects

- [ComfyUI-Grounding](https://github.com/PozzettiAndrea/ComfyUI-Grounding) — `GroundingDetector` (PozzettiAndrea)
- [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials) — `MaskBoundingBox+` (cubiq)
- [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) — background removal (`RMBG-2.0`)
- **Kepri** — inventory & cross-listing app for fashion resellers
