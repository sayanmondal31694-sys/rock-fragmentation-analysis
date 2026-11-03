"""
Interactive Lens Calibration - Tkinter version (no extra installs needed)
============================================================================
Avoids the ipympl/pip install headache entirely by using Tkinter, which
ships with Python already. This opens a separate desktop window (not inside
the notebook) for each photo.

CONTROLS
--------
- Mouse wheel: zoom in/out, centered on your cursor
- Right-click + drag: pan around the image
- Left-click: mark a point. Click twice (the two edges of the lens) to
  measure it. After 2 clicks, it auto-saves and loads the next photo.
- Press 's': skip this photo (lens not found) and move to the next
- Press 'z': undo your last click if you misclicked
- Press 'b': go BACK to the previous photo and redo it (erases its saved
  entry so you can re-measure it)
- Press 'q' or close the window: stop early (safe to resume later --
  progress is saved after every photo)

USAGE (run from a notebook cell or terminal)
----------------------------------------------
    !python calibrate_lens_tk.py --input-dir Photos --output-dir results

Produces results/calibration.csv with columns: filename, lens_diameter_px
"""

import argparse
import glob
import os
import sys
import tkinter as tk

import pandas as pd
from PIL import Image, ImageTk


def get_already_done(calib_path):
    if not os.path.exists(calib_path):
        return set()
    try:
        return set(pd.read_csv(calib_path)["filename"].astype(str))
    except Exception:
        return set()


def append_row(calib_path, filename, diameter_px):
    file_exists = os.path.exists(calib_path)
    row = {"filename": filename, "lens_diameter_px": diameter_px}
    pd.DataFrame([row]).to_csv(calib_path, mode="a", header=not file_exists, index=False)


class CalibrationApp:
    def __init__(self, image_paths, calib_path):
        self.image_paths = image_paths
        self.calib_path = calib_path
        self.idx = 0

        self.root = tk.Tk()
        self.root.title("Lens Calibration")
        self.canvas_w, self.canvas_h = 1100, 800
        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack()
        self.status = tk.Label(self.root, text="", anchor="w")
        self.status.pack(fill="x")

        self.canvas.bind("<MouseWheel>", self.on_zoom)      # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)        # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_zoom)        # Linux scroll down
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_move)
        self.root.bind("<Key>", self.on_key)

        self.load_image()
        self.root.mainloop()

    def load_image(self):
        if self.idx >= len(self.image_paths):
            self.status.config(text="All done!")
            self.root.after(1500, self.root.destroy)
            return

        path = self.image_paths[self.idx]
        self.fname = os.path.basename(path)
        self.pil_img = Image.open(path).convert("RGB")
        self.img_w, self.img_h = self.pil_img.size

        # Fit whole image in canvas initially
        self.zoom = min(self.canvas_w / self.img_w, self.canvas_h / self.img_h, 1.0)
        self.view_cx = self.img_w / 2   # center of view, in original image coords
        self.view_cy = self.img_h / 2
        self.points = []  # clicked points, in ORIGINAL image coordinates

        self.redraw()

    def visible_box(self):
        """Return (left, top, right, bottom) of the visible region in original image coords."""
        half_w = (self.canvas_w / self.zoom) / 2
        half_h = (self.canvas_h / self.zoom) / 2
        left = max(0, self.view_cx - half_w)
        top = max(0, self.view_cy - half_h)
        right = min(self.img_w, self.view_cx + half_w)
        bottom = min(self.img_h, self.view_cy + half_h)
        return left, top, right, bottom

    def redraw(self):
        left, top, right, bottom = self.visible_box()
        crop = self.pil_img.crop((int(left), int(top), int(right), int(bottom)))
        disp_w = max(1, int((right - left) * self.zoom))
        disp_h = max(1, int((bottom - top) * self.zoom))
        crop = crop.resize((disp_w, disp_h), Image.BILINEAR)

        self.tk_img = ImageTk.PhotoImage(crop)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self._view_left, self._view_top = left, top

        # Draw marked points
        for (px, py) in self.points:
            sx = (px - left) * self.zoom
            sy = (py - top) * self.zoom
            r = 5
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, outline="lime", width=2)

        n = len(self.image_paths)
        self.status.config(
            text=f"[{self.idx+1}/{n}] {self.fname}  |  zoom={self.zoom:.1f}x  |  "
                 f"clicks={len(self.points)}/2  |  wheel=zoom, right-drag=pan, "
                 f"left-click=mark, 's'=skip, 'z'=undo, 'q'=quit"
        )

    def canvas_to_image_coords(self, event_x, event_y):
        x = self._view_left + event_x / self.zoom
        y = self._view_top + event_y / self.zoom
        return x, y

    def on_zoom(self, event):
        # Determine zoom direction across platforms
        if hasattr(event, "delta") and event.delta != 0:
            factor = 1.2 if event.delta > 0 else 1 / 1.2
        elif getattr(event, "num", None) == 4:
            factor = 1.2
        elif getattr(event, "num", None) == 5:
            factor = 1 / 1.2
        else:
            return

        # Zoom centered on cursor: keep the image point under the cursor fixed
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        self.zoom = max(0.05, min(self.zoom * factor, 40.0))
        self.view_cx = img_x - (event.x - self.canvas_w / 2) / self.zoom
        self.view_cy = img_y - (event.y - self.canvas_h / 2) / self.zoom
        self.redraw()

    def on_pan_start(self, event):
        self._pan_start = (event.x, event.y, self.view_cx, self.view_cy)

    def on_pan_move(self, event):
        sx, sy, cx0, cy0 = self._pan_start
        dx = (event.x - sx) / self.zoom
        dy = (event.y - sy) / self.zoom
        self.view_cx = cx0 - dx
        self.view_cy = cy0 - dy
        self.redraw()

    def on_click(self, event):
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        self.points.append((img_x, img_y))
        self.redraw()
        if len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            diameter_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            print(f"{self.fname}: {diameter_px:.1f}px")
            append_row(self.calib_path, self.fname, diameter_px)
            self.idx += 1
            self.load_image()

    def on_key(self, event):
        if event.char == "s":
            print(f"{self.fname}: SKIPPED (not found)")
            append_row(self.calib_path, self.fname, "")
            self.idx += 1
            self.load_image()
        elif event.char == "z":
            if self.points:
                self.points.pop()
                self.redraw()
        elif event.char == "b":
            self.go_back()
        elif event.char == "q":
            self.root.destroy()

    def go_back(self):
        if self.idx == 0:
            print("Already at the first photo -- nothing to go back to.")
            return
        self.idx -= 1
        prev_fname = os.path.basename(self.image_paths[self.idx])
        # Remove that photo's saved row so it can be redone
        if os.path.exists(self.calib_path):
            df = pd.read_csv(self.calib_path)
            df = df[df["filename"] != prev_fname]
            df.to_csv(self.calib_path, index=False)
        print(f"Back to {prev_fname} -- redo it.")
        self.load_image()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    calib_path = os.path.join(args.output_dir, "calibration.csv")

    raw = (
        glob.glob(os.path.join(args.input_dir, "*.jpg"))
        + glob.glob(os.path.join(args.input_dir, "*.jpeg"))
        + glob.glob(os.path.join(args.input_dir, "*.png"))
    )
    seen = {}
    for p in raw:
        seen[os.path.normcase(os.path.abspath(p))] = p
    image_paths = sorted(seen.values())

    already_done = get_already_done(calib_path)
    remaining = [p for p in image_paths if os.path.basename(p) not in already_done]

    print(f"{len(image_paths)} photos total, {len(already_done)} already calibrated, "
          f"{len(remaining)} remaining.")

    if not remaining:
        print("Nothing left to calibrate.")
        sys.exit(0)

    CalibrationApp(remaining, calib_path)
    print(f"\nSession ended. Calibration saved to {calib_path}")


if __name__ == "__main__":
    main()
