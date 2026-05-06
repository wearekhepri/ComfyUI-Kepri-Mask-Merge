from .nodes.kepri_mask_merge import KepriMaskMerge
from .nodes.kepri_image_finalize import KepriImageFinalize

NODE_CLASS_MAPPINGS = {
    "KepriMaskMerge": KepriMaskMerge,
    "KepriImageFinalize": KepriImageFinalize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KepriMaskMerge": "Kepri Mask Merge (Multi-Object → 1 BBox)",
    "KepriImageFinalize": "Kepri Image Finalize (Resize + Crop/Pad + Bg)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
