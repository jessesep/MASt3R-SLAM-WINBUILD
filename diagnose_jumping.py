#!/usr/bin/env python3
"""
Diagnostic script to track when backend optimization causes pose "jumps"
This adds extensive logging to identify the jumping behavior
"""

import sys
import time
from pathlib import Path

# Monkey-patch the factor graph to log optimization
original_file = Path(__file__).parent / "mast3r_slam" / "factor_graph.py"

def create_diagnostic_patch():
    """Create a patched version with diagnostic logging"""
    print("=" * 60)
    print("DIAGNOSTIC MODE: Tracking backend optimization")
    print("=" * 60)

    # We'll modify main.py to add logging before/after optimization
    return """
# Add this to run_backend in main.py after line 140:

        # DIAGNOSTIC: Log pose changes before optimization
        poses_before = {}
        for i in range(len(keyframes)):
            if keyframes[i] is not None:
                try:
                    T = keyframes[i].T_WC
                    if T is not None:
                        mat = T.matrix()
                        if mat.dim() == 3:
                            poses_before[i] = mat[0, :3, 3].cpu().numpy()
                        else:
                            poses_before[i] = mat[:3, 3].cpu().numpy()
                except:
                    pass

        # Run optimization
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        # DIAGNOSTIC: Log pose changes after optimization
        poses_after = {}
        max_change = 0.0
        changed_kfs = []
        for i in range(len(keyframes)):
            if keyframes[i] is not None:
                try:
                    T = keyframes[i].T_WC
                    if T is not None:
                        mat = T.matrix()
                        if mat.dim() == 3:
                            poses_after[i] = mat[0, :3, 3].cpu().numpy()
                        else:
                            poses_after[i] = mat[:3, 3].cpu().numpy()

                        if i in poses_before:
                            import numpy as np
                            diff = np.linalg.norm(poses_after[i] - poses_before[i])
                            if diff > 0.01:  # More than 1cm change
                                changed_kfs.append((i, diff))
                                max_change = max(max_change, diff)
                except:
                    pass

        if changed_kfs:
            print(f"[DIAGNOSTIC] Backend optimization changed {len(changed_kfs)} keyframe poses:")
            print(f"  Max change: {max_change:.3f}m")
            print(f"  Changed keyframes: {[kf for kf, _ in changed_kfs]}")
            if max_change > 0.5:
                print(f"  WARNING: Large jump detected! ({max_change:.3f}m)")
"""

if __name__ == "__main__":
    patch = create_diagnostic_patch()
    print(patch)
    print("\n" + "=" * 60)
    print("To enable diagnostic mode:")
    print("1. Apply the patch above to main.py run_backend function")
    print("2. Run with: --use-threading (backend enabled)")
    print("3. Watch for [DIAGNOSTIC] messages showing pose changes")
    print("=" * 60)
