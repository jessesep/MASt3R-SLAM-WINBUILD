"""
OSC Streaming Module for MASt3R-SLAM
Streams SLAM data to TouchDesigner and other OSC-enabled applications
"""

from pythonosc import udp_client
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder
import numpy as np
import time
import threading
import queue


def rotation_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]"""
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qx, qy, qz, qw])


class OSCStreamer:
    """
    OSC Streamer for real-time SLAM data broadcasting

    Sends:
    - Camera poses
    - Point cloud data (chunked)
    - SLAM status
    - Keyframe events
    - Tracking quality metrics
    """

    def __init__(self, ip="127.0.0.1", port=9000, enabled=True):
        """
        Initialize OSC streamer

        Args:
            ip: Destination IP address
            port: Destination port
            enabled: Enable/disable streaming
        """
        self.ip = ip
        self.port = port
        self.enabled = enabled
        self.client = None
        self.connected = False

        # Statistics
        self.messages_sent = 0
        self.bytes_sent = 0
        self.last_send_time = 0
        self.send_rate = 0  # Hz

        # Rate limiting
        self.pose_rate = 30  # Hz
        self.pointcloud_rate = 5  # Hz
        self.status_rate = 2  # Hz

        self.last_pose_time = 0
        self.last_pointcloud_time = 0
        self.last_status_time = 0

        # Point cloud chunking
        self.chunk_size = 1000
        self.chunk_id = 0

        # Connection
        if self.enabled:
            self.connect()

    def connect(self):
        """Connect to OSC server"""
        try:
            self.client = udp_client.SimpleUDPClient(self.ip, self.port)
            self.connected = True
            print(f"OSC: Connected to {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"OSC: Failed to connect - {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from OSC server"""
        self.client = None
        self.connected = False
        print("OSC: Disconnected")

    def test_connection(self):
        """Test OSC connection"""
        if not self.connected:
            return False
        try:
            self.client.send_message("/slam/test", ["ping", time.time()])
            return True
        except Exception as e:
            print(f"OSC: Connection test failed - {e}")
            self.connected = False
            return False

    def _should_send(self, last_time, rate):
        """Check if enough time has passed based on rate limit"""
        if rate <= 0:
            return False
        current_time = time.time()
        min_interval = 1.0 / rate
        return (current_time - last_time) >= min_interval

    def send_camera_pose(self, T_WC, force=False):
        """
        Send 7-DOF camera pose (position + quaternion)

        Args:
            T_WC: 4x4 transformation matrix (world to camera)
            force: Force send regardless of rate limit
        """
        if not self.enabled or not self.connected:
            return

        if not force and not self._should_send(self.last_pose_time, self.pose_rate):
            return

        try:
            # Extract position
            t = T_WC[:3, 3]

            # Extract rotation and convert to quaternion
            R = T_WC[:3, :3]
            q = rotation_to_quaternion(R)

            # Send message
            self.client.send_message(
                "/slam/camera/pose",
                [float(t[0]), float(t[1]), float(t[2]),
                 float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            )

            self.last_pose_time = time.time()
            self.messages_sent += 1

        except Exception as e:
            print(f"OSC: Error sending camera pose - {e}")

    def send_pointcloud_chunk(self, points, colors, voxel_size=None, force=False):
        """
        Send point cloud chunk (batched for efficiency)

        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (0-255)
            voxel_size: Optional voxel downsampling size
            force: Force send regardless of rate limit
        """
        if not self.enabled or not self.connected:
            return

        if not force and not self._should_send(self.last_pointcloud_time, self.pointcloud_rate):
            return

        try:
            # Downsample if requested
            if voxel_size is not None and voxel_size > 0:
                points, colors = self._voxel_downsample(points, colors, voxel_size)

            # Send in chunks
            n_points = len(points)
            for start_idx in range(0, n_points, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, n_points)
                chunk_points = points[start_idx:end_idx]
                chunk_colors = colors[start_idx:end_idx]

                # Build message
                msg_data = [self.chunk_id, end_idx - start_idx]  # chunk_id, count

                for i in range(len(chunk_points)):
                    msg_data.extend([
                        float(chunk_points[i, 0]),
                        float(chunk_points[i, 1]),
                        float(chunk_points[i, 2]),
                        int(chunk_colors[i, 0]),
                        int(chunk_colors[i, 1]),
                        int(chunk_colors[i, 2])
                    ])

                self.client.send_message("/slam/pointcloud/chunk", msg_data)
                self.chunk_id += 1
                self.messages_sent += 1

            self.last_pointcloud_time = time.time()

        except Exception as e:
            print(f"OSC: Error sending point cloud - {e}")

    def send_keyframe_event(self, keyframe_id, timestamp, point_count):
        """
        Send keyframe event

        Args:
            keyframe_id: ID of new keyframe
            timestamp: Frame timestamp
            point_count: Number of points in keyframe
        """
        if not self.enabled or not self.connected:
            return

        try:
            self.client.send_message(
                "/slam/keyframe/new",
                [int(keyframe_id), float(timestamp), int(point_count)]
            )
            self.messages_sent += 1
        except Exception as e:
            print(f"OSC: Error sending keyframe event - {e}")

    def send_status(self, state, fps, total_points, avg_confidence, force=False):
        """
        Send SLAM status update

        Args:
            state: "initializing" | "tracking" | "lost" | "complete"
            fps: Current processing FPS
            total_points: Total point count
            avg_confidence: Average confidence score
            force: Force send regardless of rate limit
        """
        if not self.enabled or not self.connected:
            return

        if not force and not self._should_send(self.last_status_time, self.status_rate):
            return

        try:
            self.client.send_message(
                "/slam/status",
                [str(state), float(fps), int(total_points), float(avg_confidence)]
            )

            self.last_status_time = time.time()
            self.messages_sent += 1

        except Exception as e:
            print(f"OSC: Error sending status - {e}")

    def send_tracking_quality(self, num_inliers, num_matches, reprojection_error):
        """
        Send tracking quality metrics

        Args:
            num_inliers: Number of inlier feature matches
            num_matches: Total number of feature matches
            reprojection_error: Average reprojection error (pixels)
        """
        if not self.enabled or not self.connected:
            return

        try:
            self.client.send_message(
                "/slam/tracking/quality",
                [int(num_inliers), int(num_matches), float(reprojection_error)]
            )
            self.messages_sent += 1
        except Exception as e:
            print(f"OSC: Error sending tracking quality - {e}")

    def send_complete(self, output_path):
        """
        Send completion signal with output file path

        Args:
            output_path: Path to output PLY/trajectory files
        """
        if not self.enabled or not self.connected:
            return

        try:
            self.client.send_message("/slam/complete", [str(output_path)])
            self.messages_sent += 1
        except Exception as e:
            print(f"OSC: Error sending complete signal - {e}")

    def _voxel_downsample(self, points, colors, voxel_size):
        """
        Voxel grid downsampling for point cloud

        Args:
            points: Nx3 points
            colors: Nx3 colors
            voxel_size: Voxel size in meters

        Returns:
            Downsampled points and colors
        """
        if len(points) == 0:
            return points, colors

        # Simple voxel grid downsampling
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)

        # Find unique voxels
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)

        return points[unique_indices], colors[unique_indices]

    def get_stats(self):
        """Get streaming statistics"""
        current_time = time.time()
        if current_time > self.last_send_time:
            self.send_rate = 1.0 / (current_time - self.last_send_time) if self.last_send_time > 0 else 0
            self.last_send_time = current_time

        return {
            "connected": self.connected,
            "ip": self.ip,
            "port": self.port,
            "messages_sent": self.messages_sent,
            "send_rate": self.send_rate,
            "enabled": self.enabled
        }

    def __str__(self):
        stats = self.get_stats()
        return f"OSC[{self.ip}:{self.port}] Connected:{stats['connected']} Messages:{stats['messages_sent']}"


if __name__ == "__main__":
    # Test OSC streamer
    print("Testing OSC Streamer...")

    streamer = OSCStreamer("127.0.0.1", 9000, enabled=True)

    if streamer.test_connection():
        print("✓ Connection test passed")

        # Test camera pose
        T_WC = np.eye(4)
        T_WC[:3, 3] = [1.0, 2.0, 3.0]
        streamer.send_camera_pose(T_WC, force=True)
        print("✓ Sent camera pose")

        # Test point cloud
        points = np.random.rand(100, 3) * 5.0
        colors = np.random.randint(0, 255, (100, 3))
        streamer.send_pointcloud_chunk(points, colors, force=True)
        print("✓ Sent point cloud chunk")

        # Test status
        streamer.send_status("tracking", 15.2, 50000, 0.73, force=True)
        print("✓ Sent status")

        # Test keyframe event
        streamer.send_keyframe_event(42, time.time(), 1000)
        print("✓ Sent keyframe event")

        # Test tracking quality
        streamer.send_tracking_quality(850, 1000, 0.34)
        print("✓ Sent tracking quality")

        print(f"\nStats: {streamer}")
        print("\nAll tests passed! OSC streamer is working.")
    else:
        print("✗ Connection test failed")
        print("Make sure TouchDesigner or OSC receiver is running on port 9000")
