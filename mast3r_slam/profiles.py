"""
MASt3R-SLAM Configuration Profiles
Provides preset configurations for different use cases and hardware
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SLAMProfile:
    """SLAM configuration profile"""
    name: str
    description: str

    # Matching parameters
    matching_max_iter: int
    matching_lambda_init: float
    matching_dist_thresh: float
    matching_radius: int

    # Tracking parameters
    tracking_max_iters: int
    tracking_C_conf: float
    tracking_Q_conf: float
    tracking_min_match_frac: float

    # Local optimization
    local_opt_max_iters: int
    local_opt_use_cuda: bool
    local_opt_window_size: float

    # Visualization
    viz_max_keyframes_render: int
    viz_point_skip: int

    # Performance hints
    expected_fps_range: str
    memory_usage: str  # "low", "medium", "high"
    gpu_usage: str     # "low", "medium", "high"


# Preset profiles
PROFILES = {
    "fast": SLAMProfile(
        name="Fast",
        description="Fast processing with lower accuracy - Good for real-time preview",

        # Matching - fewer iterations
        matching_max_iter=5,
        matching_lambda_init=1e-8,
        matching_dist_thresh=0.15,
        matching_radius=2,

        # Tracking - fewer iterations, higher thresholds
        tracking_max_iters=20,
        tracking_C_conf=0.5,
        tracking_Q_conf=2.0,
        tracking_min_match_frac=0.1,

        # Local opt - minimal
        local_opt_max_iters=3,
        local_opt_use_cuda=True,
        local_opt_window_size=500000.0,

        # Visualization - very limited
        viz_max_keyframes_render=3,
        viz_point_skip=4,

        # Performance
        expected_fps_range="8-12 FPS",
        memory_usage="low",
        gpu_usage="medium"
    ),

    "balanced": SLAMProfile(
        name="Balanced",
        description="Balanced speed and accuracy - Recommended for most users",

        # Matching - moderate
        matching_max_iter=10,
        matching_lambda_init=1e-8,
        matching_dist_thresh=0.1,
        matching_radius=3,

        # Tracking - moderate
        tracking_max_iters=50,
        tracking_C_conf=0.0,
        tracking_Q_conf=1.5,
        tracking_min_match_frac=0.05,

        # Local opt - balanced
        local_opt_max_iters=10,
        local_opt_use_cuda=True,
        local_opt_window_size=1000000.0,

        # Visualization - moderate
        viz_max_keyframes_render=10,
        viz_point_skip=1,

        # Performance
        expected_fps_range="3-7 FPS",
        memory_usage="medium",
        gpu_usage="medium"
    ),

    "quality": SLAMProfile(
        name="Quality",
        description="Maximum quality reconstruction - Slower but most accurate",

        # Matching - maximum iterations
        matching_max_iter=20,
        matching_lambda_init=1e-9,
        matching_dist_thresh=0.05,
        matching_radius=5,

        # Tracking - maximum precision
        tracking_max_iters=100,
        tracking_C_conf=0.0,
        tracking_Q_conf=1.0,
        tracking_min_match_frac=0.01,

        # Local opt - thorough
        local_opt_max_iters=20,
        local_opt_use_cuda=True,
        local_opt_window_size=2000000.0,

        # Visualization - full quality
        viz_max_keyframes_render=20,
        viz_point_skip=1,

        # Performance
        expected_fps_range="1-3 FPS",
        memory_usage="high",
        gpu_usage="high"
    ),

    "lightweight": SLAMProfile(
        name="Lightweight",
        description="Minimal resource usage - For low-end hardware",

        # Matching - minimal
        matching_max_iter=3,
        matching_lambda_init=1e-7,
        matching_dist_thresh=0.2,
        matching_radius=2,

        # Tracking - minimal
        tracking_max_iters=15,
        tracking_C_conf=1.0,
        tracking_Q_conf=2.5,
        tracking_min_match_frac=0.15,

        # Local opt - minimal
        local_opt_max_iters=2,
        local_opt_use_cuda=False,  # CPU only for compatibility
        local_opt_window_size=250000.0,

        # Visualization - very limited
        viz_max_keyframes_render=2,
        viz_point_skip=8,

        # Performance
        expected_fps_range="10-15 FPS",
        memory_usage="low",
        gpu_usage="low"
    ),
}


def get_profile(profile_name: str) -> Optional[SLAMProfile]:
    """Get a profile by name"""
    return PROFILES.get(profile_name.lower())


def list_profiles():
    """List all available profiles"""
    return list(PROFILES.keys())


def profile_to_config_updates(profile: SLAMProfile) -> dict:
    """
    Convert a profile to config dictionary updates

    Returns:
        Dictionary of config updates that can be merged with base config
    """
    return {
        'matching': {
            'max_iter': profile.matching_max_iter,
            'lambda_init': profile.matching_lambda_init,
            'dist_thresh': profile.matching_dist_thresh,
            'radius': profile.matching_radius,
        },
        'tracking': {
            'max_iters': profile.tracking_max_iters,
            'C_conf': profile.tracking_C_conf,
            'Q_conf': profile.tracking_Q_conf,
            'min_match_frac': profile.tracking_min_match_frac,
        },
        'local_opt': {
            'max_iters': profile.local_opt_max_iters,
            'use_cuda': profile.local_opt_use_cuda,
            'window_size': profile.local_opt_window_size,
        },
    }


def save_profile_as_config(profile: SLAMProfile, base_config_path: str, output_path: str):
    """
    Save a profile as a YAML config file

    Args:
        profile: Profile to save
        base_config_path: Path to base config file
        output_path: Output path for new config
    """
    # Load base config
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Update with profile settings
    updates = profile_to_config_updates(profile)
    for section, values in updates.items():
        if section in config:
            config[section].update(values)
        else:
            config[section] = values

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved profile '{profile.name}' to {output_path}")


def print_profile_comparison():
    """Print a comparison of all profiles"""
    print("\n" + "=" * 80)
    print("MASt3R-SLAM Configuration Profiles")
    print("=" * 80)

    for name, profile in PROFILES.items():
        print(f"\n{profile.name.upper()}")
        print("-" * 40)
        print(f"Description: {profile.description}")
        print(f"Expected FPS: {profile.expected_fps_range}")
        print(f"Memory: {profile.memory_usage} | GPU: {profile.gpu_usage}")
        print(f"Tracking iterations: {profile.tracking_max_iters}")
        print(f"Matching iterations: {profile.matching_max_iter}")
        print(f"Viz keyframes: {profile.viz_max_keyframes_render} (skip: {profile.viz_point_skip})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Print profile comparison
    print_profile_comparison()

    # Test saving profiles as configs
    for name in list_profiles():
        profile = get_profile(name)
        save_profile_as_config(
            profile,
            "config/base.yaml",
            f"config/profiles/{name}.yaml"
        )
