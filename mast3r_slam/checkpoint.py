"""
MASt3R-SLAM Checkpoint System
Provides checkpoint saving and recovery for SLAM sessions
"""

import torch
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

from mast3r_slam.logger import get_component_logger

logger = get_component_logger("Checkpoint")


class CheckpointManager:
    """Manages checkpoints for SLAM sessions"""

    def __init__(self, checkpoint_dir="checkpoints", auto_save_interval=100):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            auto_save_interval: Save checkpoint every N frames (0 to disable)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save_interval = auto_save_interval
        self.frame_count = 0
        self.last_checkpoint_frame = 0

        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
        if auto_save_interval > 0:
            logger.info(f"Auto-save enabled: every {auto_save_interval} frames")

    def should_save(self):
        """Check if it's time to auto-save"""
        if self.auto_save_interval <= 0:
            return False

        return (self.frame_count - self.last_checkpoint_frame) >= self.auto_save_interval

    def save_checkpoint(
        self,
        keyframes,
        frame_idx,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None
    ):
        """
        Save a checkpoint

        Args:
            keyframes: Keyframes object
            frame_idx: Current frame index
            metadata: Optional metadata dictionary
            checkpoint_name: Custom checkpoint name (auto-generated if None)

        Returns:
            Path to saved checkpoint
        """
        try:
            # Generate checkpoint name
            if checkpoint_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_frame{frame_idx}_{timestamp}"

            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.ckpt"

            # Prepare checkpoint data
            checkpoint_data = {
                'frame_idx': frame_idx,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'keyframes_data': self._serialize_keyframes(keyframes),
            }

            # Save checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            # Save metadata JSON for easy inspection
            metadata_path = checkpoint_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'frame_idx': frame_idx,
                    'timestamp': checkpoint_data['timestamp'],
                    'metadata': metadata or {},
                    'num_keyframes': len(keyframes) if hasattr(keyframes, '__len__') else 0
                }, f, indent=2)

            self.last_checkpoint_frame = frame_idx
            logger.info(f"Saved checkpoint: {checkpoint_path.name} (frame {frame_idx})")

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data dictionary
        """
        try:
            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return None

            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            logger.info(f"Loaded checkpoint: {checkpoint_path.name} (frame {checkpoint_data['frame_idx']})")

            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("*.ckpt"))
        return [ckpt.stem for ckpt in checkpoints]

    def get_latest_checkpoint(self):
        """Get the most recent checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        return checkpoints[-1] if checkpoints else None

    def delete_checkpoint(self, checkpoint_name: str):
        """Delete a checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.ckpt"
        metadata_path = checkpoint_path.with_suffix('.json')

        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False

    def cleanup_old_checkpoints(self, keep_last_n=5):
        """
        Keep only the N most recent checkpoints

        Args:
            keep_last_n: Number of checkpoints to keep
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for checkpoint in checkpoints[keep_last_n:]:
            try:
                checkpoint.unlink()
                metadata_path = checkpoint.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                logger.debug(f"Cleaned up old checkpoint: {checkpoint.name}")
            except Exception as e:
                logger.warning(f"Failed to delete old checkpoint {checkpoint.name}: {e}")

    def _serialize_keyframes(self, keyframes):
        """
        Serialize keyframes for checkpoint

        Note: This is a simplified version. Full serialization would need
        to handle all keyframe data including tensors.
        """
        try:
            # For now, save minimal data
            # In a full implementation, you'd save:
            # - T_WC poses
            # - Images
            # - Confidence maps
            # - Point clouds
            # etc.

            serialized = {
                'num_keyframes': len(keyframes) if hasattr(keyframes, '__len__') else 0,
                'frame_ids': [],
                # Add more fields as needed
            }

            return serialized

        except Exception as e:
            logger.warning(f"Failed to serialize keyframes: {e}")
            return {}

    def update_frame_count(self, frame_idx):
        """Update the current frame count"""
        self.frame_count = frame_idx


# Global checkpoint manager instance
_checkpoint_manager = None


def get_checkpoint_manager(checkpoint_dir="checkpoints", auto_save_interval=100):
    """Get or create the global checkpoint manager"""
    global _checkpoint_manager

    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            auto_save_interval=auto_save_interval
        )

    return _checkpoint_manager


if __name__ == "__main__":
    # Test checkpoint system
    manager = CheckpointManager(checkpoint_dir="test_checkpoints", auto_save_interval=10)

    print(f"Checkpoint directory: {manager.checkpoint_dir}")
    print(f"Available checkpoints: {manager.list_checkpoints()}")

    # Test save
    class DummyKeyframes:
        def __len__(self):
            return 5

    keyframes = DummyKeyframes()
    checkpoint_path = manager.save_checkpoint(
        keyframes,
        frame_idx=42,
        metadata={'test': True}
    )

    print(f"Saved checkpoint: {checkpoint_path}")

    # Test load
    data = manager.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint: frame_idx={data['frame_idx']}")

    # List checkpoints
    print(f"All checkpoints: {manager.list_checkpoints()}")
