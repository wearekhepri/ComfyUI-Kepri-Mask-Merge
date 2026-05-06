import torch


class KepriImageFinalize:
    """
    One-stop finalisation node for the Kepri Background Removal V2 pipeline.

    After RMBG (or any background-removal step) you usually need to:
      1. Resize by longest edge to a fixed pixel count (e.g. 2048).
      2. Crop / pad to a target aspect ratio (square, 4:3, …).
      3. Composite the cut-out onto a background:
         - transparent  (keep RGB + mask for downstream use)
         - solid color  (#hex colour exposed via API)
         - image preset (concrete, wood, marble … exposed via API).

    All three steps are done in a single torch-only node so the workflow
    stays compact and every parameter can be wired to a workflow API input
    (prompt variable) when the workflow is executed from Modal / RunPod / Kepri.
    """

    CATEGORY = "Kepri/Background"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "finalize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "longest_edge": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 8}),
                "aspect_ratio": (
                    ["original", "1:1", "4:3", "3:4", "16:9", "9:16", "2:3", "3:2", "5:4", "4:5"],
                    {"default": "1:1"},
                ),
                "background_mode": (
                    ["transparent", "color", "image_preset"],
                    {"default": "color"},
                ),
                "background_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
            },
            "optional": {
                "mask": ("MASK",),
                "background_image": ("IMAGE",),
            },
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _hex_to_rgb_tensor(hex_str, device):
        h = hex_str.lstrip("#")
        if len(h) == 3:          # #RGB shorthand
            h = "".join([c * 2 for c in h])
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return torch.tensor([r, g, b], dtype=torch.float32, device=device)

    @staticmethod
    def _resize_longest(img, msk, target):
        """Resize so that the longest side == target.  img: [B,H,W,C]."""
        B, H, W, C = img.shape
        if H >= W:
            new_h, new_w = target, int(round(W * target / H))
        else:
            new_w, new_h = target, int(round(H * target / W))
        img = (
            torch.nn.functional.interpolate(
                img.permute(0, 3, 1, 2),
                size=(new_h, new_w),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .clamp(0.0, 1.0)
        )
        if msk is not None:
            msk = (
                torch.nn.functional.interpolate(
                    msk.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .clamp(0.0, 1.0)
            )
        return img, msk, new_h, new_w

    @staticmethod
    def _crop_or_pad(img, msk, target_h, target_w):
        """Center crop if too big, center pad (black/transparent) if too small."""
        B, H, W, C = img.shape

        # ---- pad ----
        pad_h = max(0, target_h - H)
        pad_w = max(0, target_w - W)
        if pad_h or pad_w:
            top = pad_h // 2
            bot = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            img = torch.nn.functional.pad(
                img.permute(0, 3, 1, 2),
                (left, right, top, bot),
                mode="constant",
                value=0,
            ).permute(0, 2, 3, 1)
            if msk is not None:
                msk = torch.nn.functional.pad(
                    msk.unsqueeze(1),
                    (left, right, top, bot),
                    mode="constant",
                    value=0,
                ).squeeze(1)
            B, H, W, C = img.shape

        # ---- crop ----
        if H > target_h or W > target_w:
            y0 = (H - target_h) // 2
            x0 = (W - target_w) // 2
            img = img[:, y0 : y0 + target_h, x0 : x0 + target_w, :]
            if msk is not None:
                msk = msk[:, y0 : y0 + target_h, x0 : x0 + target_w]

        return img, msk

    # ------------------------------------------------------------------ #
    def finalize(
        self,
        image,
        longest_edge,
        aspect_ratio,
        background_mode,
        background_color,
        mask=None,
        background_image=None,
    ):
        dev = image.device
        B, H, W, C = image.shape

        # -- 1. resize by longest edge --------------------------------- #
        img, msk, new_h, new_w = self._resize_longest(image, mask, longest_edge)

        # -- 2. target aspect ratio ------------------------------------ #
        if aspect_ratio == "original":
            target_h, target_w = new_h, new_w
        else:
            a, b = map(int, aspect_ratio.split(":"))
            # keep the *larger* dimension, fit the other
            if new_w >= new_h:
                target_w = new_w
                target_h = int(round(new_w * b / a))
            else:
                target_h = new_h
                target_w = int(round(new_h * a / b))

        img, msk = self._crop_or_pad(img, msk, target_h, target_w)

        # -- 3. background composite ----------------------------------- #
        if background_mode == "transparent":
            # passthrough — if user supplied a mask we pass it along,
            # otherwise we output an all-white mask so downstream nodes
            # do not crash.
            if msk is None:
                msk = torch.ones((B, target_h, target_w), dtype=torch.float32, device=dev)
            return (img, msk)

        # modes "color" and "image_preset" need a mask to composite cleanly.
        # If the user did not plug a mask we assume the image is already
        # flattened and just return it resized.
        if msk is None:
            return (img, torch.ones((B, target_h, target_w), dtype=torch.float32, device=dev))

        # expand mask to [B,H,W,1]
        alpha = msk.unsqueeze(-1)  # [B,H,W,1]
        inv = 1.0 - alpha

        if background_mode == "color":
            bg_col = self._hex_to_rgb_tensor(background_color, dev)  # [3]
            bg = bg_col.view(1, 1, 1, 3).expand(B, target_h, target_w, 3)
        else:  # image_preset
            if background_image is None:
                # no preset provided → black fallback (user will see it and fix)
                bg = torch.zeros((B, target_h, target_w, 3), dtype=torch.float32, device=dev)
            else:
                bg = background_image.to(dev)
                # resize background to target if needed
                if bg.shape[1:3] != (target_h, target_w):
                    bg = (
                        torch.nn.functional.interpolate(
                            bg.permute(0, 3, 1, 2),
                            size=(target_h, target_w),
                            mode="bicubic",
                            align_corners=False,
                        )
                        .permute(0, 2, 3, 1)
                        .clamp(0.0, 1.0)
                    )
                # match batch size
                if bg.shape[0] < B:
                    bg = bg.repeat(B, 1, 1, 1)
                elif bg.shape[0] > B:
                    bg = bg[:B]

        out = img * alpha + bg * inv
        return (out, msk.squeeze(-1) if msk.ndim == 4 else msk)
