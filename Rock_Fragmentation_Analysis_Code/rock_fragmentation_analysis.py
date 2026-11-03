"""
Rock Fragmentation Analysis (v4 - small dark/blue lens marker, color-based detection)
========================================================================================
Your scale reference is a 72mm camera lens placed in each photo, but because
shots are taken from a distance to capture the whole rock pile, the lens only
covers a small number of pixels -- generic circle-shape detection (Hough
Circles) is unreliable at that scale. This version detects it by COLOR
instead: the lens appears dark blue/black/gray, which stands out against the
reddish-brown iron ore rock in every photo.

DETECTION STRATEGY
-------------------
1. Convert to float BGR. Compute a "blueness" score = B - R for every pixel.
   Rock/soil is reddish (R > B), so it scores negative/low. The dark
   blue-black lens has B similar to or greater than R, so it scores higher.
2. Combine with a brightness (darkness) constraint, since the lens is a dark
   object -- this rules out bright sky/highlights.
3. Threshold to get a binary mask, clean up noise, find contours.
4. Filter contours by area range and circularity (how close to a circle),
   since the lens is round -- this rejects irregular dark rock shadows/cracks.
5. Pick the most circular match within the plausible size range as the lens.
6. mm_per_pixel = 72mm / detected_diameter_px for that photo.

TUNING
-------
The exact color/size will vary a bit with lighting and camera distance, so
key thresholds are exposed as CLI flags (see --help). ALWAYS check
results/debug/*_marker.jpg after a first small test run (try ~5-10 photos)
before committing to all 90 -- the debug image draws a green circle on
whatever it detected as the lens, so you can confirm it's correct or adjust
the flags.

USAGE
-----
    python rock_fragmentation_analysis.py \\
        --input-dir Photos --output-dir results --lens-diameter-mm 72

    # If detection looks wrong in debug images, tune e.g.:
    python rock_fragmentation_analysis.py \\
        --input-dir Photos --output-dir results \\
        --min-marker-radius-px 3 --max-marker-radius-px 15 \\
        --blueness-threshold 5 --max-brightness 130

    # Resume / re-run: safe to interrupt anytime, safe to re-run same
    # command (already-processed photos are skipped automatically).

    # Rebuild only the graph from what's processed so far:
    python rock_fragmentation_analysis.py --output-dir results --plot-only
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LENS_DIAMETER_MM = 72.0
DEFAULT_MAX_DIM = 1600          # resize long edge before processing (speed)
MIN_FRAGMENT_AREA_PX = 150

# --- Marker (lens) detection defaults ---
# These assume the lens looks dark blue/black/gray and covers roughly
# 3-15 px radius after resizing to --max-dim. Tune via CLI if debug images
# show misses or false positives.
DEFAULT_BLUENESS_THRESHOLD = 5     # min (B - R) to count as "not reddish rock"
DEFAULT_MAX_BRIGHTNESS = 140       # max mean pixel value (0-255) - must be dark
DEFAULT_MIN_MARKER_RADIUS_PX = 3
DEFAULT_MAX_MARKER_RADIUS_PX = 18
DEFAULT_MIN_CIRCULARITY = 0.55     # 1.0 = perfect circle; lower = more permissive


# --------------------------------------------------------------------------
# Marker (lens) detection by color + shape
# --------------------------------------------------------------------------
def detect_lens_marker(img_bgr, blueness_thresh, max_brightness,
                        min_r, max_r, min_circularity):
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    blueness = b - r
    brightness = (b.astype(np.float32) + g.astype(np.float32) + r.astype(np.float32)) / 3.0

    mask = ((blueness >= blueness_thresh) & (brightness <= max_brightness)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_circularity = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_r ** 2 or area > np.pi * max_r ** 2:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue
        if circularity > best_circularity:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            best = (int(cx), int(cy), radius)
            best_circularity = circularity

    return best  # (cx, cy, radius_px) or None


def save_marker_debug_image(img_bgr, marker, out_path):
    dbg = img_bgr.copy()
    if marker is not None:
        cx, cy, r = marker
        cv2.circle(dbg, (cx, cy), int(round(r)), (0, 255, 0), 2)
        cv2.circle(dbg, (cx, cy), 2, (0, 0, 255), -1)
    cv2.imwrite(out_path, dbg)


# --------------------------------------------------------------------------
# Segmentation (rock fragments), excluding the marker region
# --------------------------------------------------------------------------
def segment_fragments(gray_img, marker):
    h, w = gray_img.shape
    marker_mask = np.zeros((h, w), dtype=np.uint8)
    if marker is not None:
        cx, cy, r = marker
        cv2.circle(marker_mask, (cx, cy), int(r * 1.5) + 3, 255, -1)

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh[marker_mask == 255] = 0

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_img, markers)

    flat = markers.copy()
    flat[flat < 0] = 0
    counts = np.bincount(flat.ravel())
    areas = [int(c) for c in counts[2:] if c >= MIN_FRAGMENT_AREA_PX]
    return areas


# --------------------------------------------------------------------------
# Fragmentation graph
# --------------------------------------------------------------------------
def rosin_rammler(x, xc, n):
    return 100.0 * (1.0 - np.exp(-((x / xc) ** n)))


def plot_fragmentation_curve(sizes_mm, out_path):
    sizes_mm = np.sort(np.array(sizes_mm))
    n = len(sizes_mm)
    cum_pct_passing = np.arange(1, n + 1) / n * 100.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes_mm, cum_pct_passing, "o-", ms=3, label="Measured")

    try:
        popt, _ = curve_fit(
            rosin_rammler, sizes_mm, cum_pct_passing,
            p0=[np.median(sizes_mm), 1.5], maxfev=10000
        )
        xc, n_fit = popt
        x_smooth = np.linspace(sizes_mm.min(), sizes_mm.max(), 300)
        ax.plot(x_smooth, rosin_rammler(x_smooth, xc, n_fit), "r--",
                label=f"Rosin-Rammler fit (xc={xc:.1f}mm, n={n_fit:.2f})")
    except Exception as e:
        print(f"  (Rosin-Rammler fit skipped: {e})")

    ax.set_xscale("log")
    ax.set_xlabel("Fragment size (mm, equivalent diameter)")
    ax.set_ylabel("Cumulative % passing")
    ax.set_title("Rock Fragmentation Distribution")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Incremental I/O helpers (safe to interrupt / resume)
# --------------------------------------------------------------------------
def append_row_csv(path, row_dict):
    file_exists = os.path.exists(path)
    pd.DataFrame([row_dict]).to_csv(path, mode="a", header=not file_exists, index=False)


def append_rows_csv(path, rows):
    if not rows:
        return
    file_exists = os.path.exists(path)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not file_exists, index=False)


def get_already_processed_filenames(per_image_csv_path):
    if not os.path.exists(per_image_csv_path):
        return set()
    try:
        return set(pd.read_csv(per_image_csv_path)["filename"].astype(str))
    except Exception:
        return set()


def resize_for_processing(img_bgr, max_dim):
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img_bgr
    scale = max_dim / float(longest)
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rock fragmentation analysis via small color-coded lens marker")
    parser.add_argument("--input-dir", default=None, help="Folder containing the rock photos (not needed with --plot-only)")
    parser.add_argument("--output-dir", required=True, help="Where results are/will be written")
    parser.add_argument("--lens-diameter-mm", type=float, default=DEFAULT_LENS_DIAMETER_MM)
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM,
                         help="Resize long edge to this many pixels before processing")
    parser.add_argument("--calibration-csv", default=None,
                         help="Optional CSV with columns filename,lens_diameter_px to override auto-detection "
                              "for specific photos")
    parser.add_argument("--blueness-threshold", type=float, default=DEFAULT_BLUENESS_THRESHOLD,
                         help="Min (Blue - Red) to count as marker-colored, not rock-colored. Lower = more permissive.")
    parser.add_argument("--max-brightness", type=float, default=DEFAULT_MAX_BRIGHTNESS,
                         help="Max average brightness (0-255) for marker pixels (it's dark). Raise if marker is missed.")
    parser.add_argument("--min-marker-radius-px", type=float, default=DEFAULT_MIN_MARKER_RADIUS_PX)
    parser.add_argument("--max-marker-radius-px", type=float, default=DEFAULT_MAX_MARKER_RADIUS_PX,
                         help="Raise this if the lens appears larger than expected in your photos.")
    parser.add_argument("--min-circularity", type=float, default=DEFAULT_MIN_CIRCULARITY,
                         help="0-1, how round the blob must be. Lower if real detections are being rejected.")
    parser.add_argument("--plot-only", action="store_true",
                         help="Skip processing; just rebuild the graph from existing fragment size CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    calibration_csv_path = os.path.join(args.output_dir, "calibration.csv")
    per_image_csv_path = os.path.join(args.output_dir, "per_image_summary.csv")
    fragments_csv_path = os.path.join(args.output_dir, "all_fragment_sizes_mm.csv")
    graph_path = os.path.join(args.output_dir, "fragmentation_graph.png")

    if args.plot_only:
        if not os.path.exists(fragments_csv_path):
            print(f"No {fragments_csv_path} found yet -- nothing to plot.")
            sys.exit(1)
        sizes = pd.read_csv(fragments_csv_path)["fragment_size_mm"].tolist()
        plot_fragmentation_curve(sizes, graph_path)
        print(f"Rebuilt graph from {len(sizes)} fragments -> {graph_path}")
        return

    if not args.input_dir:
        print("--input-dir is required unless using --plot-only")
        sys.exit(1)

    # NOTE: on Windows/macOS, filenames are case-insensitive, so searching
    # for both "*.jpg" and "*.JPG" etc. matches the SAME files twice. We
    # dedupe by normalized path below so each real photo is only counted once
    # regardless of platform.
    raw_matches = (
        glob.glob(os.path.join(args.input_dir, "*.jpg"))
        + glob.glob(os.path.join(args.input_dir, "*.jpeg"))
        + glob.glob(os.path.join(args.input_dir, "*.png"))
        + glob.glob(os.path.join(args.input_dir, "*.JPG"))
        + glob.glob(os.path.join(args.input_dir, "*.JPEG"))
        + glob.glob(os.path.join(args.input_dir, "*.PNG"))
    )
    seen = {}
    for p in raw_matches:
        seen[os.path.normcase(os.path.abspath(p))] = p
    image_paths = sorted(seen.values())
    if not image_paths:
        print(f"No images found in {args.input_dir}")
        sys.exit(1)

    overrides = {}
    skip_set = set()
    already_in_calib_file = set()
    if args.calibration_csv and os.path.exists(args.calibration_csv):
        df = pd.read_csv(args.calibration_csv)
        for _, row in df.iterrows():
            already_in_calib_file.add(row["filename"])
            val = row.get("lens_diameter_px", None)
            if pd.notna(val) and val != "":
                overrides[row["filename"]] = float(val)  # last matching row wins
            else:
                skip_set.add(row["filename"])

    already_done = get_already_processed_filenames(per_image_csv_path)
    remaining = [p for p in image_paths if os.path.basename(p) not in already_done]

    print(f"Found {len(image_paths)} images total. "
          f"{len(already_done)} already processed (resuming). "
          f"{len(remaining)} left to do.")
    print("Safe to interrupt anytime (Jupyter: Interrupt button) -- progress saved after every photo. "
          "Re-run the same command to resume.\n")

    processed_count = 0
    missed_count = 0
    t_start = time.time()

    try:
        for path in remaining:
            fname = os.path.basename(path)
            t0 = time.time()

            img_bgr_full = cv2.imread(path)
            if img_bgr_full is None:
                print(f"  Skipping unreadable file: {fname}")
                continue

            img_bgr = resize_for_processing(img_bgr_full, args.max_dim)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            if fname in overrides:
                lens_diameter_px = overrides[fname]
                marker = None
                print(f"{fname}: using manual override diameter = {lens_diameter_px:.1f}px")
                # NOTE: do NOT re-append this to calibration_csv_path -- it's
                # already recorded in the source calibration file. Re-writing
                # it here was the bug that kept resurrecting old, already-
                # corrected values as duplicate rows on every re-run.
            elif fname in skip_set:
                print(f"{fname}: marked as skip in calibration CSV -- excluding (not auto-detecting)")
                missed_count += 1
                continue
            else:
                marker = detect_lens_marker(
                    img_bgr,
                    blueness_thresh=args.blueness_threshold,
                    max_brightness=args.max_brightness,
                    min_r=args.min_marker_radius_px,
                    max_r=args.max_marker_radius_px,
                    min_circularity=args.min_circularity,
                )
                save_marker_debug_image(img_bgr, marker, os.path.join(debug_dir, f"{fname}_marker.jpg"))

                if marker is None:
                    print(f"{fname}: marker NOT detected -- skipping "
                          f"(check debug image; consider tuning thresholds or adding a manual "
                          f"row to calibration.csv)")
                    append_row_csv(calibration_csv_path, {"filename": fname, "lens_diameter_px": ""})
                    missed_count += 1
                    continue

                lens_diameter_px = marker[2] * 2

            append_row_csv(calibration_csv_path, {"filename": fname, "lens_diameter_px": lens_diameter_px})
            mm_per_px = args.lens_diameter_mm / lens_diameter_px

            fragment_areas_px = segment_fragments(gray, marker)
            fragment_diams_mm = [(2.0 * (a / np.pi) ** 0.5) * mm_per_px for a in fragment_areas_px]

            append_rows_csv(fragments_csv_path, [
                {"filename": fname, "fragment_size_mm": d} for d in fragment_diams_mm
            ])
            append_row_csv(per_image_csv_path, {
                "filename": fname,
                "mm_per_pixel": mm_per_px,
                "lens_diameter_px": lens_diameter_px,
                "num_fragments": len(fragment_diams_mm),
                "mean_fragment_size_mm": float(np.mean(fragment_diams_mm)) if fragment_diams_mm else 0.0,
                "max_fragment_size_mm": float(np.max(fragment_diams_mm)) if fragment_diams_mm else 0.0,
            })

            processed_count += 1
            dt = time.time() - t0
            print(f"{fname}: lens={lens_diameter_px:.1f}px, {len(fragment_diams_mm)} fragments, {dt:.1f}s "
                  f"[{processed_count}/{len(remaining)} this run]")

    except KeyboardInterrupt:
        print(f"\nInterrupted by user. {processed_count} photo(s) processed and saved this run "
              f"(plus {len(already_done)} from before, {missed_count} missed detections logged). "
              f"Re-run the same command to resume, or add --plot-only to see the graph so far.")
        return

    elapsed = time.time() - t_start
    print(f"\nFinished this run: {processed_count} processed, {missed_count} missed detections, "
          f"in {elapsed/60:.1f} min.")

    if os.path.exists(fragments_csv_path):
        sizes = pd.read_csv(fragments_csv_path)["fragment_size_mm"].tolist()
        if sizes:
            plot_fragmentation_curve(sizes, graph_path)
            print(f"{len(sizes)} total fragments measured across all processed photos.")
            print(f"Graph written to: {graph_path}")
        else:
            print("No fragments measured yet -- check debug/ images and calibration.")
    else:
        print("No fragments measured yet -- check debug/ images and calibration.")


if __name__ == "__main__":
    main()
