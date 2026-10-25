"""HSV tissue filtering, gamma correction and auditable WSI patch extraction."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import tempfile
import os
import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import config


def create_tissue_mask(thumbnail):
    hsv = cv2.cvtColor(np.asarray(thumbnail)[..., :3], cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 20) & (hsv[..., 2] > 40)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)


def dynamic_stride(tissue_density):
    return int(config.PATCH_SIZE * (.4 if tissue_density > .7 else .6 if tissue_density > .3 else .8))


def mask_coverage(mask, x, y, extent, dimensions):
    width, height = dimensions
    a, b = int(x / width * mask.shape[1]), int(y / height * mask.shape[0])
    c = max(a + 1, math.ceil((x + extent) / width * mask.shape[1]))
    d = max(b + 1, math.ceil((y + extent) / height * mask.shape[0]))
    return float(mask[b:d, a:c].mean())


def tcga_specs(slide, objective_power=None):
    objective = objective_power or slide.properties.get("openslide.objective-power")
    if objective is None:
        raise ValueError("Missing objective power; supply verified --objective-power, never guess magnification")
    objective = float(objective)
    if objective < 20:
        raise ValueError("Source objective power is below the required 20x resolution")
    result = []
    for mag in (20., 10., 5.):
        downsample = objective / mag
        level = slide.get_best_level_for_downsample(downsample)
        native = float(slide.level_downsamples[level])
        result.append((level, downsample, max(224, round(224 * downsample / native)), mag))
    return result


def extract(slide, output, cohort="tcga", objective_power=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    thumb = np.asarray(slide.get_thumbnail((512, 512)).convert("RGB"))
    mask = create_tissue_mask(thumb)
    if not mask.any():
        raise ValueError("No tissue found")
    specs = (tcga_specs(slide, objective_power) if cohort == "tcga" else
             [(level, float(slide.level_downsamples[level]), 224, None)
              for level in (1, 0, 2) if level < slide.level_count])
    fd, temporary = tempfile.mkstemp(prefix=output.stem + "_", suffix=".partial.h5", dir=output.parent)
    os.close(fd)
    count, seen = 0, set()
    try:
        with h5py.File(temporary, "w") as handle:
            pixels = handle.create_dataset("patches", (0, 224, 224, 3), maxshape=(None, 224, 224, 3),
                                           chunks=(1, 224, 224, 3), dtype="uint8", compression="lzf")
            coords = handle.create_dataset("coordinates", (0, 4), maxshape=(None, 4), dtype="float64")
            handle.attrs["pixel_encoding"] = "rgb_uint8_gamma_0.9"
            handle.attrs["cohort"] = cohort
            handle.attrs["specifications"] = json.dumps(specs)
            handle.attrs["source_dimensions"] = slide.dimensions
            handle.attrs["mpp_x"] = slide.properties.get("openslide.mpp-x", "unknown")
            width, height = slide.dimensions
            for cutoff in ((.10, .05) if cohort == "tcga" else (.10,)):
                for level, ds, source_size, magnification in specs:
                    extent = round(source_size * slide.level_downsamples[level])
                    stride = max(1, round(dynamic_stride(mask.mean()) * ds))
                    for y in range(0, height - extent + 1, stride):
                        for x in range(0, width - extent + 1, stride):
                            key = (level, x, y, source_size)
                            threshold = .05 if cohort == "cgci" and count >= 100 else cutoff
                            if key in seen or mask_coverage(mask, x, y, extent, slide.dimensions) < threshold:
                                continue
                            raw = np.asarray(slide.read_region((x, y), level, (source_size, source_size)).convert("RGB"))
                            if cohort == "cgci" and raw.mean() > 230:
                                continue
                            if source_size != 224:
                                raw = np.asarray(Image.fromarray(raw).resize((224, 224), Image.Resampling.BILINEAR))
                            raw = np.rint(255 * np.power(raw.astype(np.float32) / 255, .9)).astype(np.uint8)
                            pixels.resize(count + 1, axis=0)
                            coords.resize(count + 1, axis=0)
                            pixels[count], coords[count] = raw, (x, y, level, ds)
                            seen.add(key)
                            count += 1
                    if cohort == "cgci" and count >= 100:
                        break
                if count >= 100:
                    break
            if count < 100:
                raise ValueError(f"Only {count} tissue patches; case requires QC review")
            handle.attrs["patch_count"] = count
            handle.attrs["normalisation"] = "ImageNet normalisation applied once by dataset loader"
        os.replace(temporary, output)
    except Exception:
        if Path(temporary).exists():
            Path(temporary).unlink()
        raise
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV: sample_id,wsi_path; one verified diagnostic WSI per patient")
    parser.add_argument("--cohort", choices=["tcga", "cgci"], default="tcga")
    parser.add_argument("--output-dir", default=str(Path(config.DATA_DIR) / "svs"))
    parser.add_argument("--objective-power", type=float)
    args = parser.parse_args()
    import openslide
    frame = pd.read_csv(args.manifest)
    if not {"sample_id", "wsi_path"} <= set(frame) or frame.sample_id.duplicated().any():
        raise ValueError("Require unique patient sample_id and wsi_path")
    for row in frame.itertuples():
        with openslide.OpenSlide(row.wsi_path) as slide:
            n = extract(slide, Path(args.output_dir) / f"{row.sample_id}_patches.h5",
                        args.cohort, args.objective_power)
            print(row.sample_id, n, flush=True)


if __name__ == "__main__":
    main()
