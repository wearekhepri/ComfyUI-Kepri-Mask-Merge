import os
import sys

# Add the custom node directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
