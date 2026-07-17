import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, Slider


def list_edf_files(data_dir):
    data_dir = Path(data_dir)
    patterns = ("*.edf", "*.EDF", "*.edf.gz", "*.EDF.gz")
    files = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No EDF files found in {data_dir}")
    return files


def read_edf_image(path):
    try:
        import fabio
    except ImportError as exc:
        raise ImportError(
            "Reading EDF files requires fabio. Install/use an ESRF Python environment with fabio available."
        ) from exc

    return fabio.open(str(path)).data.astype(np.float32)


def stretch_vertical(image, angle_deg=18.0, order=1):
    factor = 1.0 / math.tan(math.radians(angle_deg))
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise ImportError("Vertical stretching requires scipy.ndimage.zoom.") from exc

    return zoom(image, zoom=(factor, 1.0), order=order).astype(np.float32)


def load_edf_stack(data_dir, angle_deg=18.0, apply_stretch=True, max_frames=None):
    files = list_edf_files(data_dir)
    if max_frames is not None:
        files = files[:max_frames]

    frames = []
    for path in files:
        img = read_edf_image(path)
        if apply_stretch:
            img = stretch_vertical(img, angle_deg=angle_deg)
        frames.append(img)

    stack = np.stack(frames, axis=0).astype(np.float32)
    return stack, files


def prepare_display_image(image, log_scale=False, transpose=False):
    image = np.asarray(image, dtype=np.float32)
    if log_scale:
        image = np.log1p(np.clip(image, a_min=0.0, a_max=None))
    if transpose:
        image = image.T
    return image


def display_image(image, low=1.0, high=99.5, log_scale=False, transpose=False):
    image = prepare_display_image(image, log_scale=log_scale, transpose=transpose)
    vmin, vmax = np.percentile(image, [low, high])
    return np.clip(image, vmin, vmax), vmin, vmax


class RockingCurveROIViewer:
    def __init__(
        self,
        stack,
        files=None,
        initial_roi=None,
        cmap="viridis",
        log_scale=True,
        transpose=True,
    ):
        self.stack = np.asarray(stack, dtype=np.float32)
        self.files = files
        self.cmap = cmap
        self.log_scale = log_scale
        self.transpose = transpose
        self.frame_index = 0
        self.roi = initial_roi

        self.fig = None
        self.ax_img = None
        self.ax_curve = None
        self.slider = None
        self.widget_slider = None
        self.selector = None
        self.image_artist = None
        self.roi_patch = None
        self.frame_line = None
        self.curve_line = None

    @property
    def n_frames(self):
        return self.stack.shape[0]

    def get_roi(self):
        return self.roi

    def get_roi_stack(self):
        if self.roi is None:
            raise ValueError("No ROI selected yet. Drag a rectangle on the image first.")
        x0, y0, x1, y1 = self.roi
        return self.stack[:, y0:y1, x0:x1]

    def rocking_curve(self, roi=None):
        if roi is None:
            roi = self.roi
        if roi is None:
            return self.stack.sum(axis=(1, 2))
        x0, y0, x1, y1 = roi
        return self.stack[:, y0:y1, x0:x1].sum(axis=(1, 2))

    def _on_select(self, eclick, erelease):
        dx0, dx1 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
        dy0, dy1 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])

        height, width = self.stack.shape[1:]
        if self.transpose:
            x0, x1 = dy0, dy1
            y0, y1 = dx0, dx1
        else:
            x0, x1 = dx0, dx1
            y0, y1 = dy0, dy1

        x0 = max(0, min(width - 1, x0))
        x1 = max(1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(1, min(height, y1))

        if x1 <= x0 or y1 <= y0:
            return

        self.roi = (x0, y0, x1, y1)
        self._update_curve()
        self.fig.canvas.draw_idle()

    def _on_slider(self, value):
        self.frame_index = int(value)
        img, vmin, vmax = display_image(
            self.stack[self.frame_index],
            log_scale=self.log_scale,
            transpose=self.transpose,
        )
        self.image_artist.set_data(img)
        self.image_artist.set_clim(vmin, vmax)
        if self.frame_line is not None:
            self.frame_line.set_xdata([self.frame_index, self.frame_index])
        self.fig.canvas.draw_idle()

    def _update_curve(self):
        curve = self.rocking_curve()
        self.curve_line.set_ydata(curve)
        self.ax_curve.relim()
        self.ax_curve.autoscale_view()

    def show(self, slider_outside=True):
        img, vmin, vmax = display_image(
            self.stack[self.frame_index],
            log_scale=self.log_scale,
            transpose=self.transpose,
        )

        self.fig, (self.ax_img, self.ax_curve) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            gridspec_kw={"width_ratios": [1.2, 1.0]},
        )
        plt.subplots_adjust(bottom=0.08 if slider_outside else 0.18)

        self.image_artist = self.ax_img.imshow(img, cmap=self.cmap, vmin=vmin, vmax=vmax)
        self.ax_img.set_title("Drag rectangle to select ROI")
        self.ax_img.set_axis_off()

        curve = self.rocking_curve()
        x = np.arange(self.n_frames)
        (self.curve_line,) = self.ax_curve.plot(x, curve, color="tab:blue", linewidth=2)
        self.frame_line = self.ax_curve.axvline(self.frame_index, color="tab:red", linestyle="--")
        self.ax_curve.set_xlabel("Rocking frame")
        self.ax_curve.set_ylabel("Integrated intensity")
        self.ax_curve.grid(True, alpha=0.3)

        if slider_outside:
            try:
                import ipywidgets as widgets
                from IPython.display import display
            except ImportError as exc:
                raise ImportError(
                    "slider_outside=True requires ipywidgets. Use slider_outside=False "
                    "or install ipywidgets in the notebook environment."
                ) from exc

            self.widget_slider = widgets.IntSlider(
                value=self.frame_index,
                min=0,
                max=self.n_frames - 1,
                step=1,
                description="Frame",
                continuous_update=True,
                layout=widgets.Layout(width="80%"),
            )

            def on_widget_slider_change(change):
                if change["name"] == "value":
                    self._on_slider(change["new"])

            self.widget_slider.observe(on_widget_slider_change, names="value")
        else:
            slider_ax = self.fig.add_axes([0.15, 0.06, 0.7, 0.035])
            self.slider = Slider(
                slider_ax,
                "Frame",
                valmin=0,
                valmax=self.n_frames - 1,
                valinit=self.frame_index,
                valstep=1,
            )
            self.slider.on_changed(self._on_slider)

        self.selector = RectangleSelector(
            self.ax_img,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        plt.show()
        if slider_outside:
            display(self.widget_slider)
        return self


def make_viewer_for_frame(stack, files=None, frame_index=3, **kwargs):
    viewer = RockingCurveROIViewer(stack, files=files, **kwargs)
    viewer.frame_index = int(frame_index)
    return viewer


class LayerFrameROISelector:
    def __init__(
        self,
        previews,
        cmap="viridis",
        log_scale=True,
        transpose=True,
        crop_center_shape=(1000, 1000),
        display_percentiles=(1.0, 99.5),
    ):
        self.previews = previews
        self.cmap = cmap
        self.log_scale = log_scale
        self.transpose = transpose
        self.crop_center_shape = crop_center_shape
        self.display_percentiles = display_percentiles
        self.layer_index = 0
        self.rois = {}

        self.fig = None
        self.ax = None
        self.image_artist = None
        self.selector = None
        self.widget_slider = None
        self.roi_patch = None

    def _crop_bounds(self):
        height, width = self.previews[self.layer_index]["image"].shape
        if self.crop_center_shape is None:
            return 0, 0, width, height

        crop_h, crop_w = self.crop_center_shape
        crop_h = min(int(crop_h), height)
        crop_w = min(int(crop_w), width)
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
        return x0, y0, x0 + crop_w, y0 + crop_h

    def _display_current(self):
        x0, y0, x1, y1 = self._crop_bounds()
        low, high = self.display_percentiles
        image, vmin, vmax = display_image(
            self.previews[self.layer_index]["image"][y0:y1, x0:x1],
            low=low,
            high=high,
            log_scale=self.log_scale,
            transpose=self.transpose,
        )
        return image, vmin, vmax

    def _current_layer_name(self):
        return self.previews[self.layer_index]["layer_dir"].name

    def _display_roi_from_original_roi(self, roi):
        x0, y0, x1, y1 = roi
        cx0, cy0, _, _ = self._crop_bounds()
        x0 -= cx0
        x1 -= cx0
        y0 -= cy0
        y1 -= cy0
        if self.transpose:
            return (y0, x0, y1, x1)
        return (x0, y0, x1, y1)

    def _draw_saved_roi(self):
        if self.roi_patch is not None:
            self.roi_patch.remove()
            self.roi_patch = None

        layer_name = self._current_layer_name()
        if layer_name not in self.rois:
            return

        dx0, dy0, dx1, dy1 = self._display_roi_from_original_roi(self.rois[layer_name])
        self.roi_patch = Rectangle(
            (dx0, dy0),
            dx1 - dx0,
            dy1 - dy0,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        self.ax.add_patch(self.roi_patch)

    def _on_select(self, eclick, erelease):
        dx0, dx1 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
        dy0, dy1 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])

        height, width = self.previews[self.layer_index]["image"].shape
        if self.transpose:
            x0, x1 = dy0, dy1
            y0, y1 = dx0, dx1
        else:
            x0, x1 = dx0, dx1
            y0, y1 = dy0, dy1

        cx0, cy0, _, _ = self._crop_bounds()
        x0 += cx0
        x1 += cx0
        y0 += cy0
        y1 += cy0

        x0 = max(0, min(width - 1, x0))
        x1 = max(1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(1, min(height, y1))

        if x1 <= x0 or y1 <= y0:
            return

        layer_name = self._current_layer_name()
        self.rois[layer_name] = (x0, y0, x1, y1)
        print(f"{layer_name}: ROI {self.rois[layer_name]}")
        self._draw_saved_roi()
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, change):
        if change["name"] != "value":
            return

        self.layer_index = int(change["new"])
        image, vmin, vmax = self._display_current()
        self.image_artist.set_data(image)
        self.image_artist.set_clim(vmin, vmax)
        self.ax.set_title(f"{self._current_layer_name()} - select ROI on 4th image")
        self._draw_saved_roi()
        self.fig.canvas.draw_idle()

    def show(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError("LayerFrameROISelector requires ipywidgets.") from exc

        image, vmin, vmax = self._display_current()
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.image_artist = self.ax.imshow(image, cmap=self.cmap, vmin=vmin, vmax=vmax)
        self.ax.set_title(f"{self._current_layer_name()} - select ROI on 4th image")
        self.ax.set_axis_off()

        self.selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        self.widget_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.previews) - 1,
            step=1,
            description="Layer",
            continuous_update=False,
            layout=widgets.Layout(width="80%"),
        )
        self.widget_slider.observe(self._on_slider_change, names="value")

        plt.show()
        display(self.widget_slider)
        return self

    def get_rois(self):
        return dict(self.rois)


def list_layer_dirs(root_dir, pattern="Z_local_rocking_2x_*"):
    root_dir = Path(root_dir)
    dirs = sorted([p for p in root_dir.glob(pattern) if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No layer directories matching {pattern!r} in {root_dir}")
    return dirs


def load_layer_frame_preview(
    root_dir,
    frame_index=4,
    pattern="Z_local_rocking_2x_*",
    angle_deg=18.0,
    apply_stretch=True,
):
    previews = []
    for layer_dir in list_layer_dirs(root_dir, pattern=pattern):
        files = list_edf_files(layer_dir)
        if frame_index >= len(files):
            continue
        image = read_edf_image(files[frame_index])
        if apply_stretch:
            image = stretch_vertical(image, angle_deg=angle_deg)
        previews.append(
            {
                "layer_dir": layer_dir,
                "file": files[frame_index],
                "image": image,
            }
        )
    return previews


def plot_layer_frame_previews(
    previews,
    log_scale=True,
    transpose=True,
    cmap="viridis",
    columns=4,
    output_path=None,
):
    if not previews:
        raise ValueError("No previews to plot.")

    rows = int(math.ceil(len(previews) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    axes = axes.ravel()

    for ax, preview in zip(axes, previews):
        image, vmin, vmax = display_image(
            preview["image"],
            log_scale=log_scale,
            transpose=transpose,
        )
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(preview["layer_dir"].name)
        ax.set_axis_off()

    for ax in axes[len(previews):]:
        ax.set_axis_off()

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def integrate_roi_from_frame(roi_stack, start_frame=3, target_shape=None):
    roi_stack = np.asarray(roi_stack, dtype=np.float32)
    start_frame = int(start_frame)
    if start_frame < 0 or start_frame >= roi_stack.shape[0]:
        raise ValueError(f"start_frame={start_frame} is outside stack with {roi_stack.shape[0]} frames.")

    frame_indices = np.arange(start_frame, roi_stack.shape[0])
    integrated = roi_stack[frame_indices].sum(axis=0)
    integrated = normalize_nn_image(resize_image(integrated, target_shape))

    return {
        "integrated_image": integrated,
        "rocking_curve": roi_stack.sum(axis=(1, 2)).astype(np.float32),
        "frame_indices": frame_indices,
    }


def layer_number_from_name(layer_name):
    matches = re.findall(r"\d+", str(layer_name))
    return matches[-1] if matches else str(layer_name)


def corrected_roi_stack_from_background(
    stack,
    roi,
    start_frame=3,
    clip_negative=True,
    background_mode="outside_roi",
):
    stack = np.asarray(stack, dtype=np.float32)
    x0, y0, x1, y1 = roi
    start_frame = int(start_frame)
    if start_frame < 0 or start_frame >= stack.shape[0]:
        raise ValueError(f"start_frame={start_frame} is outside stack with {stack.shape[0]} frames.")

    if background_mode not in {"outside_roi", "full_image"}:
        raise ValueError("background_mode must be 'outside_roi' or 'full_image'.")

    corrected_frames = []
    background_levels = []

    for frame in stack[start_frame:]:
        if background_mode == "outside_roi":
            background_mask = np.ones(frame.shape, dtype=bool)
            background_mask[y0:y1, x0:x1] = False
            background_pixels = frame[background_mask]
        else:
            background_pixels = frame.ravel()

        background_level = float(background_pixels.mean() + background_pixels.std())
        roi_frame = frame[y0:y1, x0:x1] - background_level
        if clip_negative:
            roi_frame = np.clip(roi_frame, a_min=0.0, a_max=None)
        corrected_frames.append(roi_frame.astype(np.float32))
        background_levels.append(background_level)

    return np.stack(corrected_frames, axis=0), np.asarray(background_levels, dtype=np.float32)


def integrate_roi_from_frame_with_background(
    stack,
    roi,
    start_frame=3,
    target_shape=None,
    clip_negative=True,
    background_mode="outside_roi",
):
    corrected_stack, background_levels = corrected_roi_stack_from_background(
        stack,
        roi,
        start_frame=start_frame,
        clip_negative=clip_negative,
        background_mode=background_mode,
    )
    frame_indices = np.arange(start_frame, stack.shape[0])
    integrated = corrected_stack.sum(axis=0)
    integrated = normalize_nn_image(resize_image(integrated, target_shape))

    return {
        "integrated_image": integrated,
        "corrected_roi_stack": corrected_stack.astype(np.float32),
        "corrected_rocking_curve": corrected_stack.sum(axis=(1, 2)).astype(np.float32),
        "background_levels": background_levels,
        "frame_indices": frame_indices,
    }


def save_layer_roi_integrated_from_frame(
    viewer,
    output_dir,
    layer_name,
    start_frame=3,
    target_shape=(224, 224),
    save_roi_stack=False,
):
    output_dir = Path(output_dir)
    layer_output_dir = output_dir / layer_name
    layer_output_dir.mkdir(parents=True, exist_ok=True)

    roi = viewer.get_roi()
    roi_stack = viewer.get_roi_stack()
    outputs = integrate_roi_from_frame(
        roi_stack,
        start_frame=start_frame,
        target_shape=target_shape,
    )

    np.save(layer_output_dir / "integrated_from_4th_frame.npy", outputs["integrated_image"])
    np.save(layer_output_dir / "rocking_curve.npy", outputs["rocking_curve"])

    if save_roi_stack:
        np.save(layer_output_dir / "roi_stack.npy", roi_stack.astype(np.float32))

    metadata = {
        "layer_name": layer_name,
        "roi_xyxy_original_stack": list(map(int, roi)),
        "start_frame_zero_based": int(start_frame),
        "start_frame_one_based": int(start_frame + 1),
        "integrated_frame_indices": outputs["frame_indices"].astype(int).tolist(),
        "target_shape": list(target_shape) if target_shape is not None else None,
        "roi_stack_shape": list(map(int, roi_stack.shape)),
        "output_image_shape": list(map(int, outputs["integrated_image"].shape)),
    }

    with (layer_output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return outputs, metadata


def save_layer_integrated_from_roi(
    stack,
    roi,
    output_dir,
    layer_name,
    start_frame=3,
    target_shape=(510, 170),
    background_subtract=True,
    background_mode="outside_roi",
    clip_negative=True,
    save_corrected_stack=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_number = layer_number_from_name(layer_name)
    prefix = f"layer_{layer_number}"

    if background_subtract:
        outputs = integrate_roi_from_frame_with_background(
            stack,
            roi,
            start_frame=start_frame,
            target_shape=target_shape,
            clip_negative=clip_negative,
            background_mode=background_mode,
        )
    else:
        roi_stack = stack[:, roi[1]:roi[3], roi[0]:roi[2]]
        outputs = integrate_roi_from_frame(
            roi_stack,
            start_frame=start_frame,
            target_shape=target_shape,
        )
        outputs["background_levels"] = np.asarray([], dtype=np.float32)

    np.save(output_dir / f"{prefix}_integrated_from_4th_frame.npy", outputs["integrated_image"])
    np.save(output_dir / f"{prefix}_background_levels.npy", outputs["background_levels"])
    curve_key = "corrected_rocking_curve" if background_subtract else "rocking_curve"
    np.save(output_dir / f"{prefix}_rocking_curve.npy", outputs[curve_key])

    if save_corrected_stack and background_subtract:
        np.save(output_dir / f"{prefix}_corrected_roi_stack.npy", outputs["corrected_roi_stack"])

    metadata = {
        "layer_name": str(layer_name),
        "layer_number": str(layer_number),
        "roi_xyxy_original_stack": list(map(int, roi)),
        "start_frame_zero_based": int(start_frame),
        "start_frame_one_based": int(start_frame + 1),
        "integrated_frame_indices": outputs["frame_indices"].astype(int).tolist(),
        "target_shape": list(target_shape) if target_shape is not None else None,
        "output_image_shape": list(map(int, outputs["integrated_image"].shape)),
        "background_subtract": bool(background_subtract),
        "background_mode": background_mode,
        "background_definition": (
            "outside ROI per frame: mean + std"
            if background_mode == "outside_roi"
            else "full image per frame: mean + std"
        ),
        "clip_negative": bool(clip_negative),
    }

    with (output_dir / f"{prefix}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return outputs, metadata


def save_all_layer_integrated_images(records, output_dir, filename="all_layers_integrated_from_4th_frame.npy"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [record["outputs"]["integrated_image"] for record in records]
    layer_names = [record["layer_name"] for record in records]
    stack = np.stack(images, axis=0).astype(np.float32)

    np.save(output_dir / filename, stack)
    with (output_dir / "all_layers_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"layer_names": layer_names, "stack_shape": list(map(int, stack.shape))}, f, indent=2)

    return stack


def resize_image(image, target_shape):
    if target_shape is None:
        return image.astype(np.float32)

    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise ImportError("Resizing requires scipy.ndimage.zoom.") from exc

    target_h, target_w = target_shape
    zoom_y = target_h / image.shape[0]
    zoom_x = target_w / image.shape[1]
    return zoom(image, zoom=(zoom_y, zoom_x), order=1).astype(np.float32)


def normalize_nn_image(image):
    image = np.asarray(image, dtype=np.float32)
    image = image - image.min()
    max_value = image.max()
    if max_value > 0:
        image = image / max_value
    return image.astype(np.float32)


def subtract_full_image_mean(stack, clip_negative=True):
    stack = np.asarray(stack, dtype=np.float32)
    corrected = np.empty_like(stack, dtype=np.float32)
    background_levels = np.empty(stack.shape[0], dtype=np.float32)

    for i, frame in enumerate(stack):
        background_level = float(frame.mean())
        corrected_frame = frame - background_level
        if clip_negative:
            corrected_frame = np.clip(corrected_frame, a_min=0.0, a_max=None)
        corrected[i] = corrected_frame.astype(np.float32)
        background_levels[i] = background_level

    return corrected, background_levels


def subtract_full_image_mean_std(stack, clip_negative=True):
    stack = np.asarray(stack, dtype=np.float32)
    corrected = np.empty_like(stack, dtype=np.float32)
    background_levels = np.empty(stack.shape[0], dtype=np.float32)

    for i, frame in enumerate(stack):
        background_level = float(frame.mean() + frame.std())
        corrected_frame = frame - background_level
        if clip_negative:
            corrected_frame = np.clip(corrected_frame, a_min=0.0, a_max=None)
        corrected[i] = corrected_frame.astype(np.float32)
        background_levels[i] = background_level

    return corrected, background_levels


def integrate_weak_beam_tails(roi_stack, left_fraction=0.20, right_fraction=0.20, target_shape=None):
    roi_stack = np.asarray(roi_stack, dtype=np.float32)
    n_frames = roi_stack.shape[0]
    n_left = max(1, int(round(left_fraction * n_frames)))
    n_right = max(1, int(round(right_fraction * n_frames)))

    left_indices = np.arange(0, n_left)
    right_indices = np.arange(n_frames - n_right, n_frames)

    left_wb = roi_stack[left_indices].sum(axis=0)
    right_wb = roi_stack[right_indices].sum(axis=0)
    full_integrated = roi_stack.sum(axis=0)
    curve = roi_stack.sum(axis=(1, 2))

    left_wb = normalize_nn_image(resize_image(left_wb, target_shape))
    right_wb = normalize_nn_image(resize_image(right_wb, target_shape))
    full_integrated = normalize_nn_image(resize_image(full_integrated, target_shape))

    return {
        "left_weak_beam": left_wb,
        "right_weak_beam": right_wb,
        "weak_beam_pair": np.stack([left_wb, right_wb], axis=0),
        "full_integrated_roi": full_integrated,
        "rocking_curve": curve.astype(np.float32),
        "left_indices": left_indices,
        "right_indices": right_indices,
    }


def save_roi_weak_beam(
    viewer,
    output_dir,
    target_shape=(224, 224),
    left_fraction=0.20,
    right_fraction=0.20,
    save_roi_stack=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi = viewer.get_roi()
    roi_stack = viewer.get_roi_stack()
    outputs = integrate_weak_beam_tails(
        roi_stack,
        left_fraction=left_fraction,
        right_fraction=right_fraction,
        target_shape=target_shape,
    )

    np.save(output_dir / "weak_beam_left.npy", outputs["left_weak_beam"])
    np.save(output_dir / "weak_beam_right.npy", outputs["right_weak_beam"])
    np.save(output_dir / "weak_beam_pair.npy", outputs["weak_beam_pair"])
    np.save(output_dir / "full_integrated_roi.npy", outputs["full_integrated_roi"])
    np.save(output_dir / "rocking_curve.npy", outputs["rocking_curve"])

    if save_roi_stack:
        np.save(output_dir / "roi_stack.npy", roi_stack.astype(np.float32))

    metadata = {
        "roi_xyxy": list(map(int, roi)),
        "target_shape": list(target_shape) if target_shape is not None else None,
        "left_fraction": float(left_fraction),
        "right_fraction": float(right_fraction),
        "left_indices": outputs["left_indices"].astype(int).tolist(),
        "right_indices": outputs["right_indices"].astype(int).tolist(),
        "n_frames": int(roi_stack.shape[0]),
        "roi_stack_shape": list(map(int, roi_stack.shape)),
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return outputs, metadata
