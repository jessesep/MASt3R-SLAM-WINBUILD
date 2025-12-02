"""
Compatibility wrapper for lietorch using pure PyTorch implementation.
This allows the codebase to work on Windows without lietorch CUDA crashes.

Usage: Replace `import lietorch` with `import mast3r_slam.lietorch_compat as lietorch`
"""

from mast3r_slam.sim3_pytorch import Sim3, SE3

# Export the same interface as lietorch
__all__ = ['Sim3', 'SE3']

# You can add other Lie groups here if needed (SO3, etc.)
