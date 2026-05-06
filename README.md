# ComfyUI-Kepri-Mask-Merge

A lightweight ComfyUI custom node that merges multiple binary detection masks into a single unified mask. Built for the **Kepri Background Removal V2** pipeline, where `GroundingDetector` can return **N masks** (e.g. 2 shoes, 3-piece outfit) but downstream nodes (`GrowMask`, `MaskBoundingBox+`, `RMBG`) expect a **single mask** `[1, H, W]`.

---

## What It Does

`KepriMaskMerge` solves the multi-object detection gap in ComfyUI grounding workflows:

- **Input**: `MASK` tensor with shape `[N, H, W]` (multiple objects detected) or `[H, W]` (empty detection).
- **Output**: `MASK` tensor with shape `[1, H, W]` — a single unified mask.
- **Logic**:
  1. **Union** all detected masks (`max(dim=0)`) so the bounding box englobes every part.
  2. **Filter noise** by dropping masks whose area is below a configurable % of the image.
  3. **Graceful fallback**: if zero detection or all masks are noise, returns a full-image mask so the pipeline does not crash.

---

## Installation

Drop the folder into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-org/ComfyUI-Kepri-Mask-Merge.git
# or copy the folder manually
```

Restart ComfyUI and refresh the browser. The node appears under **Kepri/Background**.

**Zero extra dependencies** — only uses `torch`, already required by ComfyUI.

---

## Node Specification

| Property | Value |
|---|---|
| **Class** | `KepriMaskMerge` |
| **Display Name** | `Kepri Mask Merge (Multi-Object -> 1 BBox)` |
| **Category** | `Kepri/Background` |
| **Inputs** | `masks` (MASK), `min_mask_area_percent` (FLOAT slider 0-100) |
| **Outputs** | `merged_mask` (MASK) |

### Inputs

- **`masks`** — Connect the `masks` output from `GroundingDetector` (ComfyUI-Grounding). Accepts `[N, H, W]` or `[H, W]`.
- **`min_mask_area_percent`** — Default `1.0`. Any mask whose pixel count is < 1% of the image area is discarded as noise/artefact. Set to `0` to keep everything.

### Outputs

- **`merged_mask`** — Single mask `[1, H, W]` ready for `GrowMask`, `MaskBoundingBox+`, `RMBG`, or any ComfyUI node that expects a standard MASK.

---

## How to Use with GroundingDINO

### In the ComfyUI graph (step-by-step)

1. **LoadImage** → load your product photo.
2. **GroundingModelLoader** → select `GroundingDINO: SwinB (938MB)` (or any model).
3. **GroundingDetector**:
   - `model` ← from loader
   - `image` ← from LoadImage (resized to 2048px recommended)
   - `prompt` = `"clothing"` or `"shoe"` or `"bag"` (whatever matches the object category in your inventory system)
   - `output_masks` = enabled (so `masks` port is active)
   - `bbox_output_format` = `list_only` (standard for downstream mask nodes)
4. **KepriMaskMerge**:
   - `masks` ← from `GroundingDetector.masks`
   - `min_mask_area_percent` = `1.0` (tune if you see tiny phantom masks)
5. **GrowMask** (optional):
   - `mask` ← from `KepriMaskMerge.merged_mask`
   - `expand` = `25` (padding in pixels, keeps a margin around the object)
6. **MaskBoundingBox+**:
   - `mask` ← from `GrowMask`
   - `image_optional` ← original image (for cropped image output)
7. **RMBG** (ComfyUI-RMBG):
   - `image` ← cropped image from `MaskBoundingBox+`
   - `background_color` ← your desired background preset
8. **ImageResizeKJv2** → square 2048px → **SaveImage**

### Visual pipeline

```
LoadImage
    -> ResizeImagesByLongerEdge (2048px)
        -> GroundingDetector (prompt="shoe")
            -> masks [N, H, W]
                -> [KepriMaskMerge]
                    -> merged_mask [1, H, W]
                        -> GrowMask (+25 px padding)
                            -> MaskBoundingBox+ (crop to bbox)
                                -> RMBG-2.0 (background removal)
                                    -> ImageResizeKJv2 (square 2048)
                                        -> SaveImage
```

---

## Edge-Case Behavior

| Scenario | Result |
|---|---|
| No detection (`N == 0`) | Returns full-image white mask `[1, H, W]` — pipeline continues, user can review |
| Single detection (`N == 1`) | Passes through (reshaped to `[1, H, W]`) |
| Multiple detections (`N >= 2`) | Union of all masks, noise filtered, cropped to tightest bounding box |
| All masks below noise threshold | Returns full-image white mask — prevents silent data loss |

---

## Why Union (OR) and Not Intersection (AND)?

We want the **tightest crop that still contains every detected part**. Example: a pair of shoes — two separate masks, no overlap. Intersection would be empty. Union (`max`) gives a single mask that englobes both shoes, so the crop keeps them together.

---

## Testing / Cloud Deployment

This node is tested and deployed on **ComfyUI Cloud** (e.g. Modal, RunPod, ComfyAI). Because it has **zero extra dependencies**, it adds no weight to your Docker image.

To verify registration after install:
1. Start ComfyUI server
2. In the web UI, right-click → `Add Node` → `Kepri/Background`
3. You should see `Kepri Mask Merge (Multi-Object -> 1 BBox)`

---

## File Structure

```
ComfyUI-Kepri-Mask-Merge/
├── README.md
├── __init__.py          # ComfyUI registration
└── nodes.py             # KepriMaskMerge implementation
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
