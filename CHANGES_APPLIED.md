## Changes Applied

- Updated `thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu` to use `tokens.scalar_type()` when dispatching CUDA kernels so the build works with recent PyTorch versions.
- Switched `mast3r_slam/backend/src/matching_kernels.cu` to call `AT_DISPATCH_FLOATING_TYPES_AND_HALF` with `D11.scalar_type()` to avoid deprecated type errors during compilation.
- Replaced the PyTorch linear algebra norm call in `mast3r_slam/backend/src/gn_kernels.cu` with `dx.flatten().norm()` at the three convergence checks for better compatibility.
- Pointed `thirdparty/in3d/setup.py` at the PyPI `imgui` package instead of the incomplete bundled submodule so the dependency installs cleanly.
- Set `weights_only=False` when loading MASt3R checkpoints (`thirdparty/mast3r/mast3r/model.py`) and retrieval weights (`thirdparty/mast3r/mast3r/retrieval/processor.py`) so PyTorch 2.6+ can deserialize them.
