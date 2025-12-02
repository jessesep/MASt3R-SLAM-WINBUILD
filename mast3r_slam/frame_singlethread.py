"""Non-shared versions of Keyframes and States for single-thread mode on Windows"""
import threading
import torch
import mast3r_slam.lietorch_compat as lietorch  # PyTorch-based Sim3 for Windows
from mast3r_slam.frame import Frame, Mode
from mast3r_slam.config import config


class SingleThreadKeyframes:
    """Non-shared version of Keyframes for single-thread mode"""
    def __init__(self, h, w, buffer=512, dtype=torch.float32, device="cuda"):
        self.lock = threading.RLock()
        self.n_size = 0

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # No .share_memory_() calls - regular tensors
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int)
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype)
        self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype)
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int)
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int)
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype)
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype)
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype)
        self.N = torch.zeros(buffer, device=device, dtype=torch.int)
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int)
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype)
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long)
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool)
        self.K = torch.zeros(3, 3, device=device, dtype=dtype)

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            kf = Frame(
                frame_id=int(self.dataset_idx[idx]),
                img=self.img[idx],
                img_shape=self.img_shape[idx],
                img_true_shape=self.img_true_shape[idx],
                uimg=self.uimg[idx],
                T_WC=lietorch.Sim3(self.T_WC[idx]),
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if config["use_calib"]:
                kf.K = self.K
            return kf


    def __setitem__(self, idx, frame):
        """Update an existing keyframe"""
        with self.lock:
            if idx < 0 or idx >= self.n_size:
                raise IndexError(f"Index {idx} out of range [0, {self.n_size})")
            self.dataset_idx[idx] = frame.frame_id
            self.img[idx] = frame.img
            self.uimg[idx] = frame.uimg
            self.img_shape[idx] = frame.img_shape
            self.img_true_shape[idx] = frame.img_true_shape
            self.T_WC[idx] = frame.T_WC.data
            self.X[idx] = frame.X_canon
            self.C[idx] = frame.C
            self.N[idx] = frame.N
            self.N_updates[idx] = frame.N_updates
            if frame.feat is not None:
                self.feat[idx] = frame.feat
            if frame.pos is not None:
                self.pos[idx] = frame.pos
            self.is_dirty[idx] = True
    def __len__(self):
        with self.lock:
            return self.n_size

    def append(self, frame):
        with self.lock:
            idx = self.n_size
            if idx >= self.buffer:
                raise Exception("Keyframe buffer overflow")
            self.dataset_idx[idx] = frame.frame_id
            self.img[idx] = frame.img
            self.uimg[idx] = frame.uimg
            self.img_shape[idx] = frame.img_shape
            self.img_true_shape[idx] = frame.img_true_shape
            self.T_WC[idx] = frame.T_WC.data
            self.X[idx] = frame.X_canon
            self.C[idx] = frame.C
            self.N[idx] = frame.N
            self.N_updates[idx] = frame.N_updates
            if frame.feat is not None:
                self.feat[idx] = frame.feat
            if frame.pos is not None:
                self.pos[idx] = frame.pos
            self.is_dirty[idx] = True
            self.n_size += 1

    def pop_last(self):
        with self.lock:
            if self.n_size > 0:
                self.n_size -= 1

    def set_intrinsics(self, K):
        with self.lock:
            self.K = K

    def last_keyframe(self):
        with self.lock:
            if self.n_size > 0:
                return self[self.n_size - 1]
            return None

    def update_T_WCs(self, T_WCs, idx):
        with self.lock:
            self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx


class SingleThreadStates:
    """Non-shared version of States for single-thread mode"""
    def __init__(self, h, w, dtype=torch.float32, device="cuda"):
        self.lock = threading.RLock()
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        # Use regular Python types instead of Manager types
        self._mode = Mode.INIT
        self._is_paused = False
        self._frame = None
        self.global_optimizer_tasks = []
        self.reloc_sem = 0

        # Regular tensors (no sharing)
        self.edges_ii = []
        self.edges_jj = []

    def get_mode(self):
        with self.lock:
            return self._mode

    def set_mode(self, mode):
        with self.lock:
            self._mode = mode

    def is_paused(self):
        with self.lock:
            return self._is_paused

    def pause(self):
        with self.lock:
            self._is_paused = True

    def unpause(self):
        with self.lock:
            self._is_paused = False

    def get_frame(self):
        with self.lock:
            return self._frame

    def set_frame(self, frame):
        with self.lock:
            self._frame = frame

    def queue_global_optimization(self, idx):
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            self.reloc_sem += 1

    def dequeue_reloc(self):
        with self.lock:
            if self.reloc_sem > 0:
                self.reloc_sem -= 1

    def get_reloc_sem(self):
        """Get reloc_sem value (for compatibility with SharedStates which uses Value)"""
        with self.lock:
            return self.reloc_sem
