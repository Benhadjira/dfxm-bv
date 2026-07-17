import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


def prepare_display_image(image, log_scale=True, transpose=True):
    image = np.asarray(image, dtype=np.float32)
    if log_scale:
        image = np.log1p(np.clip(image, a_min=0.0, a_max=None))
    if transpose:
        image = image.T
    return image


class LayerBBoxSelector:
    def __init__(
        self,
        images,
        layer_names,
        crop_center_shape=(1000, 1000),
        cmap="viridis",
        log_scale=True,
        transpose=True,
    ):
        self.images = np.asarray(images, dtype=np.float32)
        self.layer_names = list(layer_names)
        self.crop_center_shape = crop_center_shape
        self.cmap = cmap
        self.log_scale = log_scale
        self.transpose = transpose
        self.layer_index = 0
        self.boxes = {}

        self.fig = None
        self.ax = None
        self.image_artist = None
        self.box_patch = None
        self.selector = None
        self.widget_slider = None

    def _layer_name(self):
        return self.layer_names[self.layer_index]

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

    def _display_box_from_original(self, box):
        x0, y0, x1, y1 = box
        cx0, cy0, _, _ = self._crop_bounds()
        x0 -= cx0
        x1 -= cx0
        y0 -= cy0
        y1 -= cy0
        if self.transpose:
            return y0, x0, y1, x1
        return x0, y0, x1, y1

    def _original_box_from_display(self, dx0, dy0, dx1, dy1):
        if self.transpose:
            x0, x1 = dy0, dy1
            y0, y1 = dx0, dx1
        else:
            x0, x1 = dx0, dx1
            y0, y1 = dy0, dy1

        cx0, cy0, _, _ = self._crop_bounds()
        return x0 + cx0, y0 + cy0, x1 + cx0, y1 + cy0

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

    def _draw_box(self):
        if self.box_patch is not None:
            self.box_patch.remove()
            self.box_patch = None

        layer_name = self._layer_name()
        if layer_name not in self.boxes:
            self.fig.canvas.draw_idle()
            return

        dx0, dy0, dx1, dy1 = self._display_box_from_original(self.boxes[layer_name])
        self.box_patch = Rectangle(
            (dx0, dy0),
            dx1 - dx0,
            dy1 - dy0,
            fill=False,
            edgecolor="red",
            linewidth=2.5,
        )
        self.ax.add_patch(self.box_patch)
        self.fig.canvas.draw_idle()

    def _on_select(self, eclick, erelease):
        dx0, dx1 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
        dy0, dy1 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])
        x0, y0, x1, y1 = self._original_box_from_display(dx0, dy0, dx1, dy1)

        height, width = self.images[self.layer_index].shape
        x0 = max(0, min(width - 1, x0))
        x1 = max(1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(1, min(height, y1))

        if x1 <= x0 or y1 <= y0:
            return

        layer_name = self._layer_name()
        self.boxes[layer_name] = (int(x0), int(y0), int(x1), int(y1))
        print(f"{layer_name}: bbox {self.boxes[layer_name]}")
        self._draw_box()

    def _on_slider_change(self, change):
        if change["name"] != "value":
            return
        self.layer_index = int(change["new"])
        image, vmin, vmax = self._display_current()
        self.image_artist.set_data(image)
        self.image_artist.set_clim(vmin, vmax)
        self.ax.set_title(f"{self._layer_name()} - draw bounding box")
        self._draw_box()

    def show(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError("LayerBBoxSelector requires ipywidgets.") from exc

        image, vmin, vmax = self._display_current()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.image_artist = self.ax.imshow(image, cmap=self.cmap, vmin=vmin, vmax=vmax)
        self.ax.set_title(f"{self._layer_name()} - draw bounding box")
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

    def get_boxes(self):
        return dict(self.boxes)


def save_bboxes(boxes, layer_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer_index, layer_name in enumerate(layer_names):
        if layer_name not in boxes:
            continue
        x0, y0, x1, y1 = boxes[layer_name]
        rows.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            }
        )

    path = output_dir / "segmented_bounding_boxes.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path, rows


def make_bbox_video(
    images,
    layer_names,
    boxes,
    output_path,
    crop_center_shape=(1000, 1000),
    cmap="viridis",
    log_scale=True,
    transpose=True,
    fps=2,
):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("Video export requires imageio.") from exc

    images = np.asarray(images, dtype=np.float32)
    frames = []

    for layer_index, layer_name in enumerate(layer_names):
        image = images[layer_index]
        height, width = image.shape
        if crop_center_shape is None:
            cx0, cy0, cx1, cy1 = 0, 0, width, height
        else:
            crop_h, crop_w = crop_center_shape
            crop_h = min(int(crop_h), height)
            crop_w = min(int(crop_w), width)
            cy0 = max(0, (height - crop_h) // 2)
            cx0 = max(0, (width - crop_w) // 2)
            cx1, cy1 = cx0 + crop_w, cy0 + crop_h

        crop = image[cy0:cy1, cx0:cx1]
        display_img = prepare_display_image(crop, log_scale=log_scale, transpose=transpose)
        vmin, vmax = np.percentile(display_img, [1.0, 99.5])

        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.imshow(np.clip(display_img, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(layer_name)
        ax.set_axis_off()

        if layer_name in boxes:
            x0, y0, x1, y1 = boxes[layer_name]
            x0 -= cx0
            x1 -= cx0
            y0 -= cy0
            y1 -= cy0
            if transpose:
                dx0, dy0, dx1, dy1 = y0, x0, y1, x1
            else:
                dx0, dy0, dx1, dy1 = x0, y0, x1, y1
            ax.add_patch(
                Rectangle(
                    (dx0, dy0),
                    dx1 - dx0,
                    dy1 - dy0,
                    fill=False,
                    edgecolor="red",
                    linewidth=3,
                )
            )

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path
