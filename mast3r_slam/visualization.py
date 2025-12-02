import dataclasses
import weakref
from pathlib import Path

import imgui
import mast3r_slam.lietorch_compat as lietorch  # PyTorch-based Sim3 for Windows
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from mast3r_slam.frame import Mode
from mast3r_slam.geometry import get_pixel_coords
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from mast3r_slam.config import load_config, config, set_global_config


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


class Window(WindowEvents):
    title = "MASt3R-SLAM"
    window_size = (1960, 1080)

    def __init__(self, states, keyframes, main2viz, viz2main, **kwargs):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        self.clear = hex2rgba("#1E2326", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg()
        self.keyframes = keyframes
        self.states = states

        self.show_all = True
        self.show_keyframe_edges = True
        self.culling = True
        self.follow_cam = True

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = True
        self.show_curr_pointmap = True
        self.show_axis = True

        # WINDOWS FIX: Limit visualization load to prevent freezing
        # Note: These settings don't significantly impact FPS (SLAM is the bottleneck)
        # but they help keep the visualization responsive and reduce GPU memory
        self.max_keyframes_render = 10  # Only render N most recent keyframes (10 is good balance)
        self.point_skip = 1  # Render every Nth point (1=all, 2=half, 4=quarter)

        self.textures = dict()
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_img, self.kf_img = Image(), Image()
        self.curr_img_np, self.kf_img_np = None, None

        self.main2viz = main2viz
        self.viz2main = viz2main

    def render(self, t: float, frametime: float):
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        curr_frame = self.states.get_frame()
        if curr_frame is not None:
            h, w = curr_frame.img_shape.flatten()
            self.frustums.make_frustum(h, w)

            self.curr_img_np = curr_frame.uimg.numpy()
            self.curr_img.write(self.curr_img_np)

            # WINDOWS FIX: Validate pose before using it for camera
            cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
            T_WC_matrix = cam_T_WC.matrix().numpy().astype(dtype=np.float32)

            # Check for NaN/inf values that would break camera
            if np.isfinite(T_WC_matrix).all():
                if self.follow_cam:
                    T_WC = T_WC_matrix @ translation_matrix(np.array([0, 0, -2], dtype=np.float32))
                    self.camera.follow_cam(np.linalg.inv(T_WC))
                else:
                    self.camera.unfollow_cam()
                self.frustums.add(
                    cam_T_WC,
                    scale=self.frustum_scale,
                    color=[0, 1, 0, 1],
                    thickness=self.line_thickness * self.scale,
                )
            else:
                # Invalid pose - don't update camera, keep last valid position
                pass

        # WINDOWS FIX: Use non-blocking lock to prevent visualization thread starvation
        # If we can't get the lock immediately, use cached values and skip updates
        if not hasattr(self, '_cached_N_keyframes'):
            self._cached_N_keyframes = 0
            self._cached_dirty_idx = []

        lock_acquired = self.keyframes.lock.acquire(blocking=False)
        if lock_acquired:
            try:
                N_keyframes = len(self.keyframes)
                dirty_idx = self.keyframes.get_dirty_idx()
                self._cached_N_keyframes = N_keyframes
                self._cached_dirty_idx = list(dirty_idx) if hasattr(dirty_idx, '__iter__') else []
            finally:
                self.keyframes.lock.release()
        else:
            # Use cached values - don't block visualization
            N_keyframes = self._cached_N_keyframes
            dirty_idx = []  # Don't process dirty idx if we couldn't get lock

        # Initialize frustum geometry from first keyframe if not already done
        if N_keyframes > 0 and self.frustums.frustum is None:
            if self.keyframes.lock.acquire(blocking=False):
                try:
                    first_kf = self.keyframes[0]
                    h, w = first_kf.img_shape.flatten()
                    self.frustums.make_frustum(h, w)
                finally:
                    self.keyframes.lock.release()

        for kf_idx in dirty_idx:
            try:
                keyframe = self.keyframes[kf_idx]
                h, w = keyframe.img_shape.flatten()
                X = self.frame_X(keyframe)

                # WINDOWS FIX: Validate point cloud data in dirty updates
                if not np.isfinite(X).all():
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)

                if keyframe.frame_id not in self.textures:
                    ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                    ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                    itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                    self.textures[keyframe.frame_id] = ptex, ctex, itex
                    ptex, ctex, itex = self.textures[keyframe.frame_id]
                    itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())

                ptex, ctex, itex = self.textures[keyframe.frame_id]
                ptex.write(X.tobytes())
                ctex.write(C.tobytes())
            except Exception as e:
                # WINDOWS FIX: Don't crash on texture update failure
                print(f"[VIZ] Texture update failed for keyframe {kf_idx}: {e}")

        # WINDOWS FIX: Only render most recent keyframes to reduce load
        start_kf = max(0, N_keyframes - self.max_keyframes_render)
        for kf_idx in range(start_kf, N_keyframes):
            try:
                keyframe = self.keyframes[kf_idx]
                h, w = keyframe.img_shape.flatten()
                if kf_idx == N_keyframes - 1:
                    self.kf_img_np = keyframe.uimg.numpy()
                    self.kf_img.write(self.kf_img_np)

                # WINDOWS FIX: Validate keyframe pose before rendering
                kf_T_WC = keyframe.T_WC.cpu()
                kf_T_WC_se3 = as_SE3(kf_T_WC)
                kf_matrix = kf_T_WC_se3.matrix().numpy()

                # Skip keyframes with invalid poses (NaN/inf)
                if not np.isfinite(kf_matrix).all():
                    continue

                color = [1, 0, 0, 1]
                if self.show_keyframe:
                    self.frustums.add(
                        kf_T_WC_se3,
                        scale=self.frustum_scale,
                        color=color,
                        thickness=self.line_thickness * self.scale,
                    )

                # WINDOWS FIX: Check if texture exists before rendering
                if keyframe.frame_id not in self.textures:
                    # Texture not created yet - create it now
                    X = self.frame_X(keyframe)

                    # WINDOWS FIX: Validate point cloud data
                    if not np.isfinite(X).all():
                        # Replace NaN/inf with zeros to prevent rendering issues
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                    C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)
                    ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                    ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                    itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                    self.textures[keyframe.frame_id] = ptex, ctex, itex
                    itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())
                    ptex.write(X.tobytes())
                    ctex.write(C.tobytes())

                ptex, ctex, itex = self.textures[keyframe.frame_id]
                if self.show_all:
                    self.render_pointmap(kf_T_WC, w, h, ptex, ctex, itex, skip=self.point_skip)
            except Exception as e:
                # WINDOWS FIX: Don't crash on keyframe render failure
                pass  # Silent skip to avoid flooding console

        if self.show_keyframe_edges:
            # WINDOWS FIX: Use non-blocking lock acquisition
            ii = torch.tensor([], dtype=torch.long)
            jj = torch.tensor([], dtype=torch.long)
            T_WCi = None
            T_WCj = None
            if self.states.lock.acquire(blocking=False):
                try:
                    ii = torch.tensor(list(self.states.edges_ii), dtype=torch.long)
                    jj = torch.tensor(list(self.states.edges_jj), dtype=torch.long)
                    if ii.numel() > 0 and jj.numel() > 0:
                        T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                        T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
                finally:
                    self.states.lock.release()
            if ii.numel() > 0 and jj.numel() > 0 and T_WCi is not None:
                # Handle both batched [N, 4, 4] and single [4, 4] cases
                mat_i = T_WCi.matrix()
                mat_j = T_WCj.matrix()

                if mat_i.dim() == 3:  # Batched
                    t_WCi = mat_i[:, :3, 3].cpu().numpy()
                else:  # Single [4, 4]
                    t_WCi = mat_i[:3, 3].unsqueeze(0).cpu().numpy()

                if mat_j.dim() == 3:  # Batched
                    t_WCj = mat_j[:, :3, 3].cpu().numpy()
                else:  # Single [4, 4]
                    t_WCj = mat_j[:3, 3].unsqueeze(0).cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )
        if self.show_curr_pointmap and self.states.get_mode() != Mode.INIT and curr_frame is not None:
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()
            h, w = curr_frame.img_shape.flatten()
            X = self.frame_X(curr_frame)

            # WINDOWS FIX: Validate current frame point cloud data
            if not np.isfinite(X).all():
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            C = curr_frame.C.cpu().numpy().astype(np.float32)
            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            itex.write(depth2rgb(X[..., -1], colormap="turbo"))
            self.render_pointmap(
                curr_frame.T_WC.cpu(),
                w,
                h,
                ptex,
                ctex,
                itex,
                use_img=True,
                depth_bias=self.depth_bias,
            )

        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self.render_ui()

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)

        imgui.spacing()
        shader_options = [
            "surfelmap.glsl",
            "trianglemap.glsl",
        ]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )

        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i

        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()

        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()
        imgui.separator()
        imgui.text("Performance (Windows)")

        # Point skip slider (1=all, 2=half, 4=quarter, 8=eighth)
        _, self.point_skip = imgui.slider_int(
            "point_skip", self.point_skip, 1, 8
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("1=all points, 2=half, 4=quarter, 8=eighth")

        # Max keyframes to render
        _, self.max_keyframes_render = imgui.slider_int(
            "max_kf_render", self.max_keyframes_render, 1, 50
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Max keyframes to render (reduces GPU load)")

        imgui.spacing()

        gui_size = imgui.get_content_region_available()
        scale = gui_size[0] / self.curr_img.texture.size[0]
        scale = min(self.scale, scale)
        size = (
            self.curr_img.texture.size[0] * scale,
            self.curr_img.texture.size[1] * scale,
        )
        image_with_text(self.kf_img, size, "kf", same_line=False)
        image_with_text(self.curr_img, size, "curr", same_line=False)

        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0, skip=1):
        w, h = int(w), int(h)
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = np.ascontiguousarray(T_WC.matrix().numpy().astype(np.float32).T)

        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        # WINDOWS FIX: Render fewer points to reduce GPU load
        # Skip renders every Nth point (1=all, 2=half, 4=quarter)
        num_vertices = (w * h) // skip
        vao.render(mode=moderngl.POINTS, vertices=num_vertices)
        vao.release()

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)


def run_visualization(cfg, states, keyframes, main2viz, viz2main) -> None:
    set_global_config(cfg)

    config_cls = Window
    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=config_cls.title,
        size=config_cls.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = config_cls(
        states=states,
        keyframes=keyframes,
        main2viz=main2viz,
        viz2main=viz2main,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
    )
    # Avoid the event assigning in the property setter for now
    # We want the even assigning to happen in WindowConfig.__init__
    # so users are free to assign them in their own __init__.
    window._config = weakref.ref(window_config)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    # WINDOWS FIX: Track render loop health
    render_count = 0
    last_error_time = 0
    error_count = 0

    while not window.is_closing:
        try:
            current_time, delta = timer.next_frame()

            if window_config.clear_color is not None:
                window.clear(*window_config.clear_color)

            # Always bind the window framebuffer before calling render
            window.use()

            window.render(current_time, delta)
            if not window.is_closing:
                window.swap_buffers()

            # WINDOWS FIX: Heartbeat every 100 frames
            render_count += 1
            if render_count % 100 == 0:
                import sys
                sys.stdout.flush()  # Ensure output is visible

        except KeyError as e:
            # WINDOWS FIX: Handle missing texture gracefully
            error_count += 1
            if current_time - last_error_time > 1.0:  # Rate limit error messages
                print(f"[VIZ WARNING] KeyError in render loop: {e} (errors: {error_count})")
                last_error_time = current_time
            # Continue rendering - don't crash

        except Exception as e:
            # WINDOWS FIX: Catch ALL exceptions to prevent silent thread death
            error_count += 1
            if current_time - last_error_time > 1.0:  # Rate limit error messages
                import traceback
                print(f"[VIZ ERROR] Exception in render loop: {type(e).__name__}: {e}")
                traceback.print_exc()
                last_error_time = current_time
            # Continue rendering - don't crash

    state = window_config.state
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)
