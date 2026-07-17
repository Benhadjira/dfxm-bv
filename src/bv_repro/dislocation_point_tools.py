import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def load_integrated_layer_images(
    input_dir="edf_layer_background_corrected_integrated",
    pattern="layer_*_integrated_from_4th_frame.npy",
):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        combined = input_dir / "all_layers_integrated_nn_input.npy"
        metadata = input_dir / "all_layers_integrated_nn_input_metadata.json"
        if not combined.exists():
            raise FileNotFoundError(f"No layer images found in {input_dir}")
        images = np.load(combined).astype(np.float32)
        if metadata.exists():
            layer_names = json.loads(metadata.read_text(encoding="utf-8")).get("layer_names", [])
        else:
            layer_names = [f"layer_{i:02d}" for i in range(images.shape[0])]
        return images, layer_names

    images = [np.load(path).astype(np.float32) for path in files]
    layer_names = [path.stem.replace("_integrated_from_4th_frame", "") for path in files]
    return np.stack(images, axis=0), layer_names


def prepare_display_image(image, log_scale=False, transpose=False):
    image = np.asarray(image, dtype=np.float32)
    if log_scale:
        image = np.log1p(np.clip(image, a_min=0.0, a_max=None))
    if transpose:
        image = image.T
    return image


class LayerPointSelector:
    def __init__(
        self,
        images,
        layer_names=None,
        cmap="viridis",
        log_scale=False,
        transpose=False,
        crop_center_shape=None,
    ):
        self.images = np.asarray(images, dtype=np.float32)
        self.layer_names = layer_names or [f"layer_{i:02d}" for i in range(self.images.shape[0])]
        self.cmap = cmap
        self.log_scale = log_scale
        self.transpose = transpose
        self.crop_center_shape = crop_center_shape
        self.layer_index = 0
        self.points = {}

        self.fig = None
        self.ax = None
        self.image_artist = None
        self.point_artist = None
        self.widget_slider = None

    def _crop_bounds(self):
        height, width = self.images[self.layer_index].shape
        if self.crop_center_shape is None:
            return 0, 0, width, height

        crop_h, crop_w = self.crop_center_shape
        crop_h = min(int(crop_h), height)
        crop_w = min(int(crop_w), width)
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
        return x0, y0, x0 + crop_w, y0 + crop_h

    def _layer_name(self):
        return self.layer_names[self.layer_index]

    def _display_current(self):
        x0, y0, x1, y1 = self._crop_bounds()
        image = self.images[self.layer_index, y0:y1, x0:x1]
        image = prepare_display_image(
            image,
            log_scale=self.log_scale,
            transpose=self.transpose,
        )
        vmin, vmax = np.percentile(image, [1.0, 99.5])
        return np.clip(image, vmin, vmax), vmin, vmax

    def _display_point_from_original(self, point):
        x, y = point
        x0, y0, _, _ = self._crop_bounds()
        x = x - x0
        y = y - y0
        if self.transpose:
            return y, x
        return x, y

    def _original_point_from_display(self, x_display, y_display):
        if self.transpose:
            x, y = float(y_display), float(x_display)
        else:
            x, y = float(x_display), float(y_display)
        x0, y0, _, _ = self._crop_bounds()
        return x + x0, y + y0

    def _draw_point(self):
        if self.point_artist is not None:
            self.point_artist.remove()
            self.point_artist = None

        layer_name = self._layer_name()
        if layer_name not in self.points:
            self.fig.canvas.draw_idle()
            return

        x_display, y_display = self._display_point_from_original(self.points[layer_name])
        self.point_artist = Circle(
            (x_display, y_display),
            radius=5,
            fill=False,
            color="red",
            linewidth=2,
        )
        self.ax.add_patch(self.point_artist)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x, y = self._original_point_from_display(event.xdata, event.ydata)
        height, width = self.images[self.layer_index].shape
        x = max(0.0, min(width - 1.0, x))
        y = max(0.0, min(height - 1.0, y))

        layer_name = self._layer_name()
        self.points[layer_name] = (x, y)
        print(f"{layer_name}: x={x:.2f}, y={y:.2f}")
        self._draw_point()

    def _on_slider_change(self, change):
        if change["name"] != "value":
            return

        self.layer_index = int(change["new"])
        image, vmin, vmax = self._display_current()
        self.image_artist.set_data(image)
        self.image_artist.set_clim(vmin, vmax)
        self.ax.set_title(f"{self._layer_name()} - click dislocation point")
        self._draw_point()

    def show(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError("LayerPointSelector requires ipywidgets.") from exc

        image, vmin, vmax = self._display_current()
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.image_artist = self.ax.imshow(image, cmap=self.cmap, vmin=vmin, vmax=vmax)
        self.ax.set_title(f"{self._layer_name()} - click dislocation point")
        self.ax.set_axis_off()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.widget_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.images.shape[0] - 1,
            step=1,
            description="Layer",
            continuous_update=False,
            layout=widgets.Layout(width="80%"),
        )
        self.widget_slider.observe(self._on_slider_change, names="value")

        plt.show()
        display(self.widget_slider)
        return self

    def get_points(self):
        return dict(self.points)


def points_to_xyz(
    points,
    layer_names,
    pixel_size_x_um=1.0,
    pixel_size_y_um=1.0,
    layer_step_z_um=1.0,
    origin_x_um=0.0,
    origin_y_um=0.0,
    origin_z_um=0.0,
):
    rows = []
    for layer_index, layer_name in enumerate(layer_names):
        if layer_name not in points:
            continue
        x_px, y_px = points[layer_name]
        rows.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "x_px": float(x_px),
                "y_px": float(y_px),
                "x_um": origin_x_um + float(x_px) * pixel_size_x_um,
                "y_um": origin_y_um + float(y_px) * pixel_size_y_um,
                "z_um": origin_z_um + layer_index * layer_step_z_um,
            }
        )
    return rows


def save_points(rows, output_dir="dislocation_3d_points"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dislocation_points_3d.json"
    csv_path = output_dir / "dislocation_points_3d.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = ["layer_index", "layer_name", "x_px", "y_px", "x_um", "y_um", "z_um"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def plot_dislocation_3d(
    rows,
    output_path="dislocation_trajectory_3d.png",
    title="3D Dislocation Line",
    xlim=None,
    ylim=None,
    zlim=None,
    equal_box_aspect=True,
):
    if not rows:
        raise ValueError("No points to plot.")

    x = np.asarray([row["x_um"] for row in rows], dtype=float)
    y = np.asarray([row["y_um"] for row in rows], dtype=float)
    z = np.asarray([row["z_um"] for row in rows], dtype=float)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="red", linewidth=2)
    ax.scatter(x, y, z, color="blue", s=40, depthshade=True)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("lab x in [um]")
    ax.set_ylabel("lab y in [um]")
    ax.set_zlabel("lab z in [um]")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zlim is not None:
        ax.set_zlim(*zlim)
    if equal_box_aspect:
        x_range = (xlim[1] - xlim[0]) if xlim is not None else np.ptp(x)
        y_range = (ylim[1] - ylim[0]) if ylim is not None else np.ptp(y)
        z_range = (zlim[1] - zlim[0]) if zlim is not None else np.ptp(z)
        ax.set_box_aspect((max(x_range, 1e-9), max(y_range, 1e-9), max(z_range, 1e-9)))
    ax.grid(True, alpha=0.35)
    ax.view_init(elev=22, azim=-55)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
