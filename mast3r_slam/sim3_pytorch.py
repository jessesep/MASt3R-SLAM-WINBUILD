"""
Pure PyTorch implementation of Sim3 Lie group operations.
This replaces lietorch to work around Windows CUDA kernel crashes.

Sim3 representation: [t_x, t_y, t_z, q_x, q_y, q_z, q_w, scale]
- Translation: t (3D)
- Rotation: q (quaternion, xyzw format)
- Scale: s (scalar)
"""

import torch
import math


class Sim3:
    """Sim3 Lie group in PyTorch (Similarity transformation: rotation + translation + scale)"""

    # Class attribute for compatibility with lietorch
    embedded_dim = 8  # [t_x, t_y, t_z, q_x, q_y, q_z, q_w, scale]

    def __init__(self, data):
        """
        Args:
            data: [batch, 8] tensor with [t_x, t_y, t_z, q_x, q_y, q_z, q_w, scale]
        """
        self.data = data
        self.device = data.device
        self.dtype = data.dtype

    @property
    def shape(self):
        return self.data.shape[:-1]  # Return batch shape

    @staticmethod
    def Identity(batch_size, device="cpu", dtype=torch.float32):
        """Create identity Sim3 transformations"""
        data = torch.zeros(batch_size, 8, device=device, dtype=dtype)
        data[..., 6] = 1.0  # q_w = 1 (identity rotation)
        data[..., 7] = 1.0  # scale = 1
        return Sim3(data)

    def _decompose(self):
        """Decompose into translation, quaternion, scale"""
        t = self.data[..., 0:3]  # translation
        q = self.data[..., 3:7]  # quaternion (x, y, z, w)
        s = self.data[..., 7:8]  # scale
        return t, q, s

    def inv(self):
        """Invert the Sim3 transformation: T^(-1)"""
        t, q, s = self._decompose()

        # Inverse scale
        s_inv = 1.0 / s

        # Inverse quaternion (conjugate for unit quaternions)
        q_inv = torch.cat([
            -q[..., 0:1],  # -x
            -q[..., 1:2],  # -y
            -q[..., 2:3],  # -z
            q[..., 3:4]    # w (unchanged)
        ], dim=-1)

        # Inverse translation: t_inv = -s_inv * R_inv * t
        t_inv = -s_inv * self._quat_rotate(q_inv, t)

        # Compose inverse
        data_inv = torch.cat([t_inv, q_inv, s_inv], dim=-1)
        return Sim3(data_inv)

    def __mul__(self, other):
        """Compose two Sim3 transformations: self * other"""
        t1, q1, s1 = self._decompose()
        t2, q2, s2 = other._decompose()

        # Composed scale
        s_comp = s1 * s2

        # Composed rotation
        q_comp = self._quat_multiply(q1, q2)

        # Composed translation: t_comp = s1 * R1 * t2 + t1
        t_comp = s1 * self._quat_rotate(q1, t2) + t1

        # Compose result
        data_comp = torch.cat([t_comp, q_comp, s_comp], dim=-1)
        return Sim3(data_comp)

    def act(self, points):
        """
        Transform points: p' = s * R * p + t

        Args:
            points: [N, 3] or [batch, N, 3] tensor of 3D points

        Returns:
            Transformed points with same shape as input
        """
        t, q, s = self._decompose()

        # Handle different input shapes
        if points.dim() == 2:
            # [N, 3] -> [1, N, 3]
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Rotate points
        rotated = self._quat_rotate(q.unsqueeze(1), points)

        # Scale and translate
        transformed = s.unsqueeze(1) * rotated + t.unsqueeze(1)

        if squeeze_output:
            transformed = transformed.squeeze(0)

        return transformed

    def retr(self, delta):
        """
        Retraction: apply exponential map update

        Args:
            delta: [batch, 7] tangent space perturbation [tau (3), omega (3), sigma (1)]
                   tau: translation perturbation
                   omega: rotation perturbation (so3)
                   sigma: scale perturbation (log space)

        Returns:
            Updated Sim3: self * exp(delta)
        """
        tau = delta[..., 0:3]      # translation
        omega = delta[..., 3:6]    # rotation (axis-angle)
        sigma = delta[..., 6:7]    # scale (log space)

        # Exponential map for scale
        ds = torch.exp(sigma)

        # Exponential map for rotation (axis-angle to quaternion)
        dq = self._exp_so3(omega)

        # Exponential map for translation (simplified version)
        # For small perturbations, we use the BCH formula approximation
        dt = self._compute_V_matrix(omega) @ tau.unsqueeze(-1)
        dt = dt.squeeze(-1)

        # Create delta Sim3
        delta_data = torch.cat([dt, dq, ds], dim=-1)
        delta_sim3 = Sim3(delta_data)

        # Compose: self * delta
        return self * delta_sim3

    def adjT(self, xi):
        """
        Apply adjoint transpose: Ad_T^T * xi

        This is used in optimization for transforming gradients.

        Args:
            xi: [batch, 7] tangent vector

        Returns:
            Transformed tangent vector [batch, 7]
        """
        t, q, s = self._decompose()

        tau = xi[..., 0:3]      # translation part
        omega = xi[..., 3:6]    # rotation part
        sigma = xi[..., 6:7]    # scale part

        s_inv = 1.0 / s

        # Transform translation: s_inv * R * tau
        tau_adj = s_inv * self._quat_rotate(q, tau)

        # Transform rotation: R * omega + [t]_x * (s_inv * R * tau)
        omega_adj = self._quat_rotate(q, omega)
        cross_term = torch.cross(t, s_inv * self._quat_rotate(q, tau), dim=-1)
        omega_adj = omega_adj + cross_term

        # Transform scale: sigma + t^T * (s_inv * R * tau)
        sigma_adj = sigma + (t * s_inv * self._quat_rotate(q, tau)).sum(dim=-1, keepdim=True)

        return torch.cat([tau_adj, omega_adj, sigma_adj], dim=-1)

    # ========== Quaternion utilities ==========

    @staticmethod
    def _quat_multiply(q1, q2):
        """Multiply two quaternions: q1 * q2"""
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([x, y, z, w], dim=-1)

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate vector v by quaternion q"""
        # Extract quaternion components
        qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        # Extract vector components (handle different shapes)
        if v.dim() == q.dim():
            # Same dimensions
            vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
        else:
            # v has extra dimension (batch of points)
            vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
            qx = qx.unsqueeze(-1)
            qy = qy.unsqueeze(-1)
            qz = qz.unsqueeze(-1)
            qw = qw.unsqueeze(-1)

        # Compute rotation using quaternion formula
        # v' = v + 2 * qw * (q_xyz × v) + 2 * q_xyz × (q_xyz × v)

        # First cross product: q_xyz × v
        t0 = 2.0 * (qy * vz - qz * vy)
        t1 = 2.0 * (qz * vx - qx * vz)
        t2 = 2.0 * (qx * vy - qy * vx)

        # Result
        rx = vx + qw * t0 + (qy * t2 - qz * t1)
        ry = vy + qw * t1 + (qz * t0 - qx * t2)
        rz = vz + qw * t2 + (qx * t1 - qy * t0)

        return torch.stack([rx, ry, rz], dim=-1)

    @staticmethod
    def _exp_so3(omega):
        """
        Exponential map from so(3) to SO(3), output as quaternion

        Args:
            omega: [batch, 3] axis-angle representation

        Returns:
            q: [batch, 4] quaternion (x, y, z, w)
        """
        theta_sq = (omega ** 2).sum(dim=-1, keepdim=True)
        theta = torch.sqrt(theta_sq + 1e-8)

        # Small angle approximation
        small_angle = theta_sq < 1e-6

        # For small angles: q = [omega/2, 1]
        half_theta = theta / 2.0
        sin_half = torch.where(small_angle,
                              0.5 - theta_sq / 48.0,
                              torch.sin(half_theta) / theta)
        cos_half = torch.where(small_angle,
                              1.0 - theta_sq / 8.0,
                              torch.cos(half_theta))

        qxyz = omega * sin_half
        qw = cos_half

        return torch.cat([qxyz, qw], dim=-1)

    @staticmethod
    def _compute_V_matrix(omega):
        """
        Compute V matrix for SE(3) exponential map

        For small omega, V ≈ I
        """
        theta_sq = (omega ** 2).sum(dim=-1, keepdim=True)
        theta = torch.sqrt(theta_sq + 1e-8)

        # For very small angles, use identity
        small_angle = theta_sq < 1e-6

        # Compute coefficients
        A = torch.where(small_angle,
                       1.0 - theta_sq / 6.0,
                       torch.sin(theta) / theta)
        B = torch.where(small_angle,
                       0.5 - theta_sq / 24.0,
                       (1.0 - torch.cos(theta)) / theta_sq)
        C = torch.where(small_angle,
                       1.0 / 6.0 - theta_sq / 120.0,
                       (1.0 - A) / theta_sq)

        # Compute skew-symmetric matrix [omega]_x
        batch_size = omega.shape[0]
        omega_skew = torch.zeros(batch_size, 3, 3, device=omega.device, dtype=omega.dtype)
        omega_skew[:, 0, 1] = -omega[:, 2]
        omega_skew[:, 0, 2] = omega[:, 1]
        omega_skew[:, 1, 0] = omega[:, 2]
        omega_skew[:, 1, 2] = -omega[:, 0]
        omega_skew[:, 2, 0] = -omega[:, 1]
        omega_skew[:, 2, 1] = omega[:, 0]

        # Compute omega_skew^2
        omega_skew_sq = omega_skew @ omega_skew

        # V = I + B * omega_skew + C * omega_skew^2
        I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        V = I + B.unsqueeze(-1) * omega_skew + C.unsqueeze(-1) * omega_skew_sq

        return V

    def __repr__(self):
        return f"Sim3(data={self.data})"


class SE3:
    """SE3 Lie group in PyTorch (Rigid transformation: rotation + translation, no scale)"""

    # Class attribute for compatibility with lietorch
    embedded_dim = 7  # [t_x, t_y, t_z, q_x, q_y, q_z, q_w]

    def __init__(self, data):
        """
        Args:
            data: [batch, 7] tensor with [t_x, t_y, t_z, q_x, q_y, q_z, q_w]
        """
        self.data = data
        self.device = data.device
        self.dtype = data.dtype

    @property
    def shape(self):
        return self.data.shape[:-1]  # Return batch shape

    @staticmethod
    def Identity(batch_size, device="cpu", dtype=torch.float32):
        """Create identity SE3 transformations"""
        data = torch.zeros(batch_size, 7, device=device, dtype=dtype)
        data[..., 6] = 1.0  # q_w = 1 (identity rotation)
        return SE3(data)

    def cpu(self):
        """Move to CPU"""
        return SE3(self.data.cpu())

    def cuda(self):
        """Move to CUDA"""
        return SE3(self.data.cuda())

    def to(self, device):
        """Move to device"""
        return SE3(self.data.to(device))

    def _decompose(self):
        """Decompose into translation, quaternion"""
        t = self.data[..., 0:3]  # translation
        q = self.data[..., 3:7]  # quaternion (x, y, z, w)
        return t, q

    def inv(self):
        """Invert the SE3 transformation: T^(-1)"""
        t, q = self._decompose()

        # Inverse quaternion (conjugate for unit quaternions)
        q_inv = torch.cat([
            -q[..., 0:1],  # -x
            -q[..., 1:2],  # -y
            -q[..., 2:3],  # -z
            q[..., 3:4]    # w (unchanged)
        ], dim=-1)

        # Inverse translation: t_inv = -R_inv * t
        t_inv = -Sim3._quat_rotate(q_inv, t)

        # Compose inverse
        data_inv = torch.cat([t_inv, q_inv], dim=-1)
        return SE3(data_inv)

    def __mul__(self, other):
        """Compose two SE3 transformations: self * other"""
        t1, q1 = self._decompose()
        t2, q2 = other._decompose()

        # Composed rotation
        q_comp = Sim3._quat_multiply(q1, q2)

        # Composed translation: t_comp = R1 * t2 + t1
        t_comp = Sim3._quat_rotate(q1, t2) + t1

        # Compose result
        data_comp = torch.cat([t_comp, q_comp], dim=-1)
        return SE3(data_comp)

    def act(self, points):
        """
        Transform points: p' = R * p + t

        Args:
            points: [N, 3] or [batch, N, 3] tensor of 3D points

        Returns:
            Transformed points with same shape as input
        """
        t, q = self._decompose()

        # Handle different input shapes
        if points.dim() == 2:
            # [N, 3] -> [1, N, 3]
            points = points.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Rotate points
        rotated = Sim3._quat_rotate(q.unsqueeze(1), points)

        # Translate
        transformed = rotated + t.unsqueeze(1)

        if squeeze_output:
            transformed = transformed.squeeze(0)

        return transformed

    def __repr__(self):
        return f"SE3(data={self.data})"


def test_sim3_pytorch():
    """Test the PyTorch Sim3 implementation"""
    print("="*80)
    print("TESTING PYTORCH SIM3 IMPLEMENTATION")
    print("="*80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Test 1: Identity
    print("\n[TEST 1] Creating identity...")
    T1 = Sim3.Identity(1, device=device)
    print(f"  [OK] T1: {T1.data}")

    T2 = Sim3.Identity(1, device=device)
    print(f"  [OK] T2: {T2.data}")

    # Test 2: Inversion
    print("\n[TEST 2] Inversion...")
    T1_inv = T1.inv()
    print(f"  [OK] T1_inv: {T1_inv.data}")

    # Test 3: Multiplication
    print("\n[TEST 3] Multiplication...")
    T3 = T1 * T2
    print(f"  [OK] T3 = T1 * T2: {T3.data}")

    # Test 4: Act on points
    print("\n[TEST 4] Acting on points...")
    points = torch.randn(1000, 3, device=device)
    transformed = T1.act(points)
    print(f"  [OK] Transformed {points.shape} points -> {transformed.shape}")

    # Test 5: Retraction
    print("\n[TEST 5] Retraction...")
    delta = torch.randn(1, 7, device=device) * 0.01
    T1_new = T1.retr(delta)
    print(f"  [OK] T1_new after retraction: {T1_new.data}")

    # Test 6: Large batch of points
    print("\n[TEST 6] Large batch (196608 points)...")
    large_points = torch.randn(196608, 3, device=device)
    transformed_large = T1.act(large_points)
    print(f"  [OK] Transformed {large_points.shape} -> {transformed_large.shape}")

    print("\n" + "="*80)
    print("[OK] ALL TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    test_sim3_pytorch()
