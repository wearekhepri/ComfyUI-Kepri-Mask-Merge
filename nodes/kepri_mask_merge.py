import torch


class KepriMaskMerge:
    """
    ComfyUI custom node that merges multiple masks from a Grounding Detector
    into a single bounding-box mask.

    When a GroundingDetector detects multiple objects (e.g. 2 shoes, 3-piece
    outfit), it outputs N masks of shape [N, H, W].  The downstream
    MaskBoundingBox+ node only accepts a single mask [1, H, W].  This node
    bridges the gap:

    1. Computes the union of all detected masks (logical OR).
    2. Optionally filters out very small masks (noise / background artefacts).
    3. Returns one mask of shape [1, H, W].

    If no mask is detected (N == 0) it outputs a full-image mask so the
    pipeline keeps running and the user can review the result instead of
    hitting a hard crash.
    """

    CATEGORY = "Kepri/Background"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("merged_mask",)
    FUNCTION = "merge_masks"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "min_mask_area_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "display": "slider",
                        "label": "Min mask area % (noise filter)",
                    },
                ),
            }
        }

    def merge_masks(self, masks: torch.Tensor, min_mask_area_percent: float):
        """
        Parameters
        ----------
        masks : torch.Tensor
            Input from GroundingDetector.
            - N > 0  -> shape [N, H, W]  (float 0..1 or bool)
            - N == 0 -> shape [H, W]     (empty detection, Comfy passes it
                                          as [H,W] not [0,H,W])
        min_mask_area_percent : float
            Masks whose pixel count is < this percentage of the total image
            area are discarded as noise.  0 = keep everything.

        Returns
        -------
        tuple(torch.Tensor)  -> ([1, H, W],)
        """
        # -- 0. Normalise la shape ------------------------------------- #
        if masks.ndim == 2:
            # Aucune detection : Comfy envoie [H, W] au lieu de [0, H, W]
            # On renvoie un masque plein pour ne pas casser la pipeline.
            h, w = masks.shape
            full = torch.ones((1, h, w), dtype=torch.float32, device=masks.device)
            return (full,)

        if masks.ndim == 3:
            n, h, w = masks.shape
        else:
            raise ValueError(f"KepriMaskMerge: unexpected mask ndim={masks.ndim}")

        if n == 0:
            full = torch.ones((1, h, w), dtype=torch.float32, device=masks.device)
            return (full,)

        # -- 1. Filtrage anti-artefacts -------------------------------- #
        total_pixels = h * w
        threshold_pixels = (min_mask_area_percent / 100.0) * total_pixels

        kept_masks = []
        for i in range(n):
            m = masks[i]
            area = m.sum().item()
            if area >= threshold_pixels:
                kept_masks.append(m)

        if not kept_masks:
            # Tous les masks etaient du bruit -> masque plein
            full = torch.ones((1, h, w), dtype=torch.float32, device=masks.device)
            return (full,)

        kept = torch.stack(kept_masks, dim=0)  # [K, H, W]

        # -- 2. Union logique ------------------------------------------ #
        # On prend le max sur la dimension des masks : 1 si au moins un mask
        # couvre le pixel, 0 sinon.
        union = kept.max(dim=0).values  # [H, W]
        union = union.unsqueeze(0)      # [1, H, W]

        # -- 3. S'assurer qu'on sort bien en float32 ------------------- #
        union = union.to(dtype=torch.float32)
        return (union,)
