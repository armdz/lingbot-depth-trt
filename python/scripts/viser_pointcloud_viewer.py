"""Viser point cloud display helpers for the RealSense live demo."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float

    @classmethod
    def from_realsense(cls, intrinsics: Any) -> "CameraIntrinsics":
        return cls(
            width=int(intrinsics.width),
            height=int(intrinsics.height),
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            ppx=float(intrinsics.ppx),
            ppy=float(intrinsics.ppy),
        )


@dataclass(frozen=True)
class CameraState:
    wxyz: np.ndarray
    position: np.ndarray
    look_at: np.ndarray
    up_direction: np.ndarray
    fov: float
    near: float
    far: float

    @classmethod
    def from_camera(cls, camera: Any) -> "CameraState":
        return cls(
            wxyz=np.asarray(camera.wxyz, dtype=np.float64).copy(),
            position=np.asarray(camera.position, dtype=np.float64).copy(),
            look_at=np.asarray(camera.look_at, dtype=np.float64).copy(),
            up_direction=np.asarray(camera.up_direction, dtype=np.float64).copy(),
            fov=float(camera.fov),
            near=float(camera.near),
            far=float(camera.far),
        )


@dataclass(frozen=True)
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray
    pixels: np.ndarray


@dataclass(frozen=True)
class PixelInspection:
    pixel: tuple[int, int]
    point: np.ndarray
    distance_m: float


class DepthPointCloudProjector:
    """Vectorized RealSense depth deprojection for the live viser viewer."""

    def __init__(self, intrinsics: CameraIntrinsics, stride: int):
        self.intrinsics = intrinsics
        self.stride = max(1, int(stride))
        rows, cols = np.mgrid[
            0 : intrinsics.height : self.stride,
            0 : intrinsics.width : self.stride,
        ].astype(np.float32)
        self.cols = cols.reshape(-1)
        self.rows = rows.reshape(-1)

    def project(
        self,
        depth_m: np.ndarray,
        color_bgr: np.ndarray,
        min_depth_m: float,
        max_depth_m: float,
    ) -> PointCloudData:
        if depth_m.shape[:2] != (self.intrinsics.height, self.intrinsics.width):
            raise RuntimeError(
                f"Point cloud projector expects {self.intrinsics.width}x{self.intrinsics.height}, "
                f"got {depth_m.shape[1]}x{depth_m.shape[0]}"
            )

        depth = depth_m[:: self.stride, :: self.stride].reshape(-1)
        rgb = np.ascontiguousarray(color_bgr[:: self.stride, :: self.stride, ::-1]).reshape(-1, 3)
        valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
        if not valid.any():
            return PointCloudData(
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                pixels=np.zeros((0, 2), dtype=np.int32),
            )

        z = depth[valid].astype(np.float32, copy=False)
        pixels = np.column_stack((self.cols[valid], self.rows[valid])).astype(np.int32, copy=False)
        points = self.deproject_pixels(pixels[:, 0].astype(np.float32), pixels[:, 1].astype(np.float32), z)
        return PointCloudData(points=points, colors=rgb[valid].astype(np.uint8, copy=False), pixels=pixels)

    def deproject_pixels(self, cols: np.ndarray, rows: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        z = depth_m.astype(np.float32, copy=False)
        x = (cols - self.intrinsics.ppx) * z / self.intrinsics.fx
        y_down = (rows - self.intrinsics.ppy) * z / self.intrinsics.fy

        points = np.empty((z.shape[0], 3), dtype=np.float32)
        points[:, 0] = x
        points[:, 1] = z
        points[:, 2] = -y_down
        return points

    def deproject_pixel(self, u: int, v: int, depth_m: float) -> np.ndarray:
        return self.deproject_pixels(
            np.array([float(u)], dtype=np.float32),
            np.array([float(v)], dtype=np.float32),
            np.array([depth_m], dtype=np.float32),
        )[0]


def display_host(host: str) -> str:
    return "localhost" if host in ("0.0.0.0", "::") else host


def apply_camera_state(client: Any, state: CameraState) -> None:
    with client.atomic():
        client.camera.wxyz = state.wxyz
        client.camera.position = state.position
        client.camera.look_at = state.look_at
        client.camera.up_direction = state.up_direction
        client.camera.fov = state.fov
        client.camera.near = state.near
        client.camera.far = state.far
    client.flush()


def make_dashboard_handler(raw_port: int, refined_port: int) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LingBot-Depth point cloud comparison</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #111318;
      color: #e8edf2;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      height: 34px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      align-items: center;
      background: #171a20;
      border-bottom: 1px solid #2a2f38;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0;
    }}
    header div {{
      padding: 0 12px;
    }}
    main {{
      height: calc(100vh - 34px);
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 1px;
      background: #2a2f38;
    }}
    iframe {{
      display: block;
      width: 100%;
      height: 100%;
      border: 0;
      background: #111318;
    }}
    @media (max-width: 900px) {{
      body {{
        overflow: auto;
      }}
      header, main {{
        grid-template-columns: 1fr;
      }}
      main {{
        height: calc(200vh - 34px);
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div>raw depth</div>
    <div>refined depth</div>
  </header>
  <main>
    <iframe id="raw" title="raw depth"></iframe>
    <iframe id="refined" title="refined depth"></iframe>
  </main>
  <script>
    const host = window.location.hostname || "localhost";
    const protocol = window.location.protocol || "http:";
    document.getElementById("raw").src = `${{protocol}}//${{host}}:{raw_port}`;
    document.getElementById("refined").src = `${{protocol}}//${{host}}:{refined_port}`;
  </script>
</body>
</html>
""".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


class ViserComparisonDashboard:
    def __init__(self, host: str, port: int, raw_port: int, refined_port: int):
        handler = make_dashboard_handler(raw_port, refined_port)
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def get_host(self) -> str:
        return str(self.httpd.server_address[0])

    def get_port(self) -> int:
        return int(self.httpd.server_address[1])

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=1.0)


class ViserPointCloudPane:
    EMPTY_POINTS = np.zeros((0, 3), dtype=np.float32)
    EMPTY_COLORS = np.zeros((0, 3), dtype=np.uint8)

    def __init__(
        self,
        args: argparse.Namespace,
        intrinsics: CameraIntrinsics,
        name: str,
        label: str,
        port: int,
    ):
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError("viser is required for --show-viser; install it with `uv pip install viser`.") from exc

        self.name = name
        self.label_text = label
        self.projector = DepthPointCloudProjector(intrinsics, args.pointcloud_stride)
        self.server = viser.ViserServer(host=args.viser_host, port=port, label=label)
        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (0.0, -1.3, 0.55)
        self.server.initial_camera.look_at = (0.0, 0.85, 0.0)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.far = 10.0
        self.server.gui.configure_theme(control_layout="collapsible", control_width="medium", dark_mode=True)

        self.data_lock = threading.Lock()
        self.current_points = self.EMPTY_POINTS
        self.current_pixels = np.zeros((0, 2), dtype=np.int32)
        self.latest_depth_m: np.ndarray | None = None
        self.latest_min_depth_m = float(args.pointcloud_min_depth)
        self.latest_max_depth_m = float(args.pointcloud_max_depth)

        self.axis = self.server.scene.add_frame(f"/{name}/camera", axes_length=0.12, axes_radius=0.006)
        self.label = self.server.scene.add_label(
            f"/{name}/label",
            label,
            position=(0.0, -0.08, 0.42),
            anchor="center-center",
        )
        self.cloud = self.server.scene.add_point_cloud(
            f"/{name}/points",
            self.EMPTY_POINTS,
            self.EMPTY_COLORS,
            point_size=args.pointcloud_point_size,
            point_shape="rounded",
            point_shading="flat",
            precision=args.pointcloud_precision,
        )
        self.selection_marker = self.server.scene.add_icosphere(
            f"/{name}/selection_marker",
            radius=0.012,
            color=(255, 230, 0),
            position=(0.0, 0.0, 0.0),
            visible=False,
        )
        self.selection_label = self.server.scene.add_label(
            f"/{name}/selection_label",
            "",
            position=(0.0, 0.0, 0.0),
            visible=False,
            anchor="bottom-center",
        )

        with self.server.gui.add_folder("Point cloud"):
            self.freeze = self.server.gui.add_checkbox("Freeze", initial_value=False)
            self.min_depth = self.server.gui.add_slider(
                "Min depth m",
                min=0.01,
                max=1.0,
                step=0.01,
                initial_value=args.pointcloud_min_depth,
            )
            self.max_depth = self.server.gui.add_slider(
                "Max depth m",
                min=0.1,
                max=6.0,
                step=0.05,
                initial_value=args.pointcloud_max_depth,
            )
            self.point_size = self.server.gui.add_slider(
                "Point size",
                min=0.001,
                max=0.03,
                step=0.001,
                initial_value=args.pointcloud_point_size,
            )
            self.update_hz = self.server.gui.add_slider(
                "Update Hz",
                min=1.0,
                max=30.0,
                step=1.0,
                initial_value=args.pointcloud_update_hz,
            )
            self.status = self.server.gui.add_text("Status", initial_value="waiting for frames", disabled=True)
            self.axis_size = self.server.gui.add_slider(
                "Axis size",
                min=0.25,
                max=5.0,
                step=0.05,
                initial_value=1.0,
            )
            self.selection_info = self.server.gui.add_text(
                "Selection",
                initial_value="click a point",
                multiline=True,
                disabled=True,
            )

        @self.axis_size.on_update
        def _(_) -> None:
            self.axis.scale = float(self.axis_size.value)

        @self.server.on_client_connect
        def _(client: Any) -> None:
            client.camera.far = 10.0

        self.last_update_s = 0.0

    def stop(self) -> None:
        self.server.stop()

    def update(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        frame_index: int,
        inference_ms: float,
    ) -> None:
        if self.freeze.value or not self.server.get_clients():
            return

        update_hz = max(1.0, float(self.update_hz.value))
        now = time.perf_counter()
        if now - self.last_update_s < 1.0 / update_hz:
            return
        self.last_update_s = now

        max_depth = max(0.02, float(self.max_depth.value))
        min_depth = min(max(0.01, float(self.min_depth.value)), max_depth - 0.001)
        cloud = self.projector.project(depth_m, color_bgr, min_depth, max_depth)
        with self.data_lock:
            self.current_points = cloud.points
            self.current_pixels = cloud.pixels
            self.latest_depth_m = depth_m.copy()
            self.latest_min_depth_m = min_depth
            self.latest_max_depth_m = max_depth

        with self.server.atomic():
            self.axis.scale = float(self.axis_size.value)
            self.cloud.point_size = float(self.point_size.value)
            self.cloud.points = cloud.points
            self.cloud.colors = cloud.colors
            self.status.value = f"frame={frame_index} points={cloud.points.shape[0]} infer={inference_ms:.1f}ms"
        self.server.flush()

    def nearest_pixel_on_ray(
        self,
        ray_origin: tuple[float, float, float],
        ray_direction: tuple[float, float, float],
        max_distance_m: float = 0.04,
    ) -> tuple[int, int] | None:
        with self.data_lock:
            points = self.current_points.copy()
            pixels = self.current_pixels.copy()
        if points.shape[0] == 0:
            return None

        origin = np.asarray(ray_origin, dtype=np.float32)
        direction = np.asarray(ray_direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            return None
        direction /= norm

        offsets = points - origin[None, :]
        ray_t = offsets @ direction
        in_front = ray_t > 0.0
        if not in_front.any():
            return None

        closest = origin[None, :] + ray_t[:, None] * direction[None, :]
        dist2 = np.einsum("ij,ij->i", points - closest, points - closest)
        dist2[~in_front] = np.inf
        index = int(np.argmin(dist2))
        if not np.isfinite(dist2[index]) or dist2[index] > max_distance_m * max_distance_m:
            return None
        u, v = pixels[index]
        return int(u), int(v)

    def inspect_pixel(self, pixel: tuple[int, int]) -> PixelInspection | None:
        u, v = pixel
        with self.data_lock:
            depth_m = None if self.latest_depth_m is None else self.latest_depth_m.copy()
            min_depth_m = self.latest_min_depth_m
            max_depth_m = self.latest_max_depth_m
        if depth_m is None:
            return None
        if u < 0 or v < 0 or u >= depth_m.shape[1] or v >= depth_m.shape[0]:
            return None

        z = float(depth_m[v, u])
        if not np.isfinite(z) or z < min_depth_m or z > max_depth_m:
            return None

        point = self.projector.deproject_pixel(u, v, z)
        return PixelInspection(pixel=(u, v), point=point, distance_m=float(np.linalg.norm(point)))

    def show_pixel_selection(self, pixel: tuple[int, int], source_name: str) -> None:
        inspection = self.inspect_pixel(pixel)
        if inspection is None:
            with self.server.atomic():
                self.selection_marker.visible = False
                self.selection_label.visible = False
                self.selection_info.value = (
                    f"source={source_name}\n"
                    f"pixel=({pixel[0]}, {pixel[1]})\n"
                    "depth=invalid or out of range"
                )
            self.server.flush()
            return

        point = inspection.point
        distance_m = inspection.distance_m
        with self.server.atomic():
            self.selection_marker.position = point
            self.selection_marker.visible = True
            self.selection_label.position = point + np.array((0.0, 0.0, 0.035), dtype=np.float32)
            self.selection_label.text = f"x={point[0]:.3f} y={point[1]:.3f} z={point[2]:.3f} d={distance_m:.3f}m"
            self.selection_label.visible = True
            self.selection_info.value = (
                f"source={source_name}\n"
                f"pixel=({inspection.pixel[0]}, {inspection.pixel[1]})\n"
                f"x={point[0]:.4f} m\n"
                f"y={point[1]:.4f} m\n"
                f"z={point[2]:.4f} m\n"
                f"distance={distance_m:.4f} m"
            )
        self.server.flush()


class ViserPointCloudViewer:
    def __init__(self, args: argparse.Namespace, intrinsics: CameraIntrinsics):
        self.args = args
        self.mode = str(args.viser_mode)
        self.panes: list[ViserPointCloudPane] = []
        self.raw_pane: ViserPointCloudPane | None = None
        self.refined_pane: ViserPointCloudPane | None = None
        self.dashboard: ViserComparisonDashboard | None = None
        self._camera_lock = threading.Lock()
        self._syncing_camera = False
        self._last_camera_state: CameraState | None = None
        self._axis_lock = threading.Lock()
        self._syncing_axis_size = False

        if self.mode == "both":
            raw_port = args.viser_port + 1
            refined_port = args.viser_port + 2
            self.raw_pane = ViserPointCloudPane(args, intrinsics, "raw", "raw depth", raw_port)
            self.refined_pane = ViserPointCloudPane(args, intrinsics, "refined", "refined depth", refined_port)
            self.panes = [self.raw_pane, self.refined_pane]
            self._install_camera_sync()
            self._install_selection_sync()
            self._install_axis_size_sync()
            self.dashboard = ViserComparisonDashboard(
                args.viser_host,
                args.viser_port,
                self.raw_pane.server.get_port(),
                self.refined_pane.server.get_port(),
            )
            host = display_host(self.dashboard.get_host())
            pane_host = display_host(args.viser_host)
            print(f"Viser point cloud dashboard: http://{host}:{self.dashboard.get_port()}")
            print(f"  raw depth: http://{pane_host}:{self.raw_pane.server.get_port()}")
            print(f"  refined depth: http://{pane_host}:{self.refined_pane.server.get_port()}")
        else:
            label = "raw depth" if self.mode == "raw" else "refined depth"
            pane = ViserPointCloudPane(args, intrinsics, self.mode, label, args.viser_port)
            self.panes = [pane]
            if self.mode == "raw":
                self.raw_pane = pane
            else:
                self.refined_pane = pane
            self._install_selection_sync()
            self._install_axis_size_sync()
            host = display_host(pane.server.get_host())
            print(f"Viser point cloud viewer: http://{host}:{pane.server.get_port()}")

    def _install_camera_sync(self) -> None:
        for pane in self.panes:
            self._install_camera_sync_for_pane(pane)

    def _install_camera_sync_for_pane(self, source_pane: ViserPointCloudPane) -> None:
        @source_pane.server.on_client_connect
        def _(client: Any) -> None:
            with self._camera_lock:
                last_state = self._last_camera_state
            if last_state is not None:
                apply_camera_state(client, last_state)

            @client.camera.on_update
            def _(camera: Any) -> None:
                self._sync_camera_from(source_pane, CameraState.from_camera(camera))

    def _install_selection_sync(self) -> None:
        for pane in self.panes:
            self._install_selection_sync_for_pane(pane)

    def _install_selection_sync_for_pane(self, source_pane: ViserPointCloudPane) -> None:
        @source_pane.server.scene.on_click()
        def _(event: Any) -> None:
            pixel = source_pane.nearest_pixel_on_ray(event.ray_origin, event.ray_direction)
            if pixel is None:
                source_pane.selection_info.value = "no point near click"
                source_pane.selection_marker.visible = False
                source_pane.selection_label.visible = False
                source_pane.server.flush()
                return
            self._show_selection_from(source_pane, pixel)

    def _show_selection_from(self, source_pane: ViserPointCloudPane, pixel: tuple[int, int]) -> None:
        for pane in self.panes:
            pane.show_pixel_selection(pixel, source_pane.name)

    def _install_axis_size_sync(self) -> None:
        for pane in self.panes:
            self._install_axis_size_sync_for_pane(pane)

    def _install_axis_size_sync_for_pane(self, source_pane: ViserPointCloudPane) -> None:
        @source_pane.axis_size.on_update
        def _(_) -> None:
            self._sync_axis_size_from(source_pane, float(source_pane.axis_size.value))

    def _sync_axis_size_from(self, source_pane: ViserPointCloudPane, value: float) -> None:
        with self._axis_lock:
            if self._syncing_axis_size:
                return
            self._syncing_axis_size = True

        try:
            for pane in self.panes:
                if pane is source_pane:
                    continue
                pane.axis_size.value = value
                pane.axis.scale = value
                pane.server.flush()
        finally:
            with self._axis_lock:
                self._syncing_axis_size = False

    def _sync_camera_from(self, source_pane: ViserPointCloudPane, state: CameraState) -> None:
        with self._camera_lock:
            if self._syncing_camera:
                return
            self._syncing_camera = True
            self._last_camera_state = state

        try:
            for pane in self.panes:
                if pane is source_pane:
                    continue
                for client in pane.server.get_clients().values():
                    apply_camera_state(client, state)
        finally:
            with self._camera_lock:
                self._syncing_camera = False

    def stop(self) -> None:
        if self.dashboard is not None:
            self.dashboard.stop()
        for pane in self.panes:
            pane.stop()

    def update(
        self,
        color_bgr: np.ndarray,
        raw_depth_m: np.ndarray,
        refined_depth_m: np.ndarray,
        frame_index: int,
        inference_ms: float,
    ) -> None:
        if self.mode == "both":
            assert self.raw_pane is not None
            assert self.refined_pane is not None
            self.raw_pane.update(color_bgr, raw_depth_m, frame_index, inference_ms)
            self.refined_pane.update(color_bgr, refined_depth_m, frame_index, inference_ms)
        elif self.mode == "raw":
            assert self.raw_pane is not None
            self.raw_pane.update(color_bgr, raw_depth_m, frame_index, inference_ms)
        else:
            assert self.refined_pane is not None
            self.refined_pane.update(color_bgr, refined_depth_m, frame_index, inference_ms)
