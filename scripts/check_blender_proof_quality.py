from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def _basic_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        "mean_brightness": float(np.mean(gray)),
        "std_brightness": float(np.std(gray)),
        "mean_saturation": float(np.mean(hsv[:, :, 1])),
        "bright_ratio": float(np.mean(gray > 180)),
        "dark_ratio": float(np.mean(gray < 28)),
        "edge_ratio": float(np.mean(cv2.Canny(gray, 60, 160) > 0)),
    }


def _color_mask_ratio(image: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return float(np.mean(mask > 0))


def _run_checks(debug_dir: Path, contact_sheet: Path | None) -> dict[str, object]:
    required = [
        "dancers_wide.png",
        "dancers_close.png",
        "reflection_close.png",
        "water_wide.png",
        "water_close.png",
        "clouds_wide.png",
        "clouds_close.png",
        "glass_wide.png",
        "glass_close.png",
        "orbit_angle.png",
    ]
    failures: list[str] = []
    frames: dict[str, dict[str, object]] = {}

    for name in required:
        path = debug_dir / name
        if not path.exists():
            failures.append(f"missing:{name}")
            continue
        size_kb = path.stat().st_size / 1024.0
        if size_kb < 50.0:
            failures.append(f"too_small:{name}:{size_kb:.1f}kb")
        image = _load_image(path)
        metrics = _basic_metrics(image)
        frames[name] = {
            "path": str(path),
            "size_kb": round(size_kb, 1),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        if name not in {"clouds_wide.png", "clouds_close.png"} and metrics["std_brightness"] < 20.0:
            failures.append(f"low_contrast:{name}:{metrics['std_brightness']:.2f}")

    if "clouds_close.png" in frames:
        img = _load_image(debug_dir / "clouds_close.png")
        m = _basic_metrics(img)
        sky_ratio = _color_mask_ratio(img, (85, 45, 120), (120, 190, 255))
        white_ratio = _color_mask_ratio(img, (0, 0, 190), (179, 70, 255))
        frames["clouds_close.png"]["sky_ratio"] = round(sky_ratio, 4)
        frames["clouds_close.png"]["white_ratio"] = round(white_ratio, 4)
        if m["mean_brightness"] < 95 or m["bright_ratio"] < 0.18 or white_ratio < 0.18 or m["std_brightness"] < 10 or m["dark_ratio"] > 0.35:
            failures.append("clouds_close_failed")

    if "water_close.png" in frames:
        img = _load_image(debug_dir / "water_close.png")
        blue_ratio = _color_mask_ratio(img, (90, 80, 45), (125, 255, 255))
        foam_ratio = _color_mask_ratio(img, (0, 0, 185), (179, 80, 255))
        edges = _basic_metrics(img)["edge_ratio"]
        frames["water_close.png"]["blue_ratio"] = round(blue_ratio, 4)
        frames["water_close.png"]["foam_ratio"] = round(foam_ratio, 4)
        if blue_ratio < 0.18 or foam_ratio < 0.01 or edges < 0.025:
            failures.append("water_close_failed")

    if "dancers_close.png" in frames:
        img = _load_image(debug_dir / "dancers_close.png")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        colorful = float(np.mean((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 65)))
        dark_floor = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 40))
        frames["dancers_close.png"]["colorful_ratio"] = round(colorful, 4)
        frames["dancers_close.png"]["dark_floor_ratio"] = round(dark_floor, 4)
        if colorful < 0.10 or dark_floor < 0.10:
            failures.append("dancers_close_failed")

    if "glass_close.png" in frames:
        img = _load_image(debug_dir / "glass_close.png")
        cyan_ratio = _color_mask_ratio(img, (80, 30, 90), (115, 180, 255))
        bright_ratio = _basic_metrics(img)["bright_ratio"]
        dark_ratio = _basic_metrics(img)["dark_ratio"]
        frames["glass_close.png"]["cyan_ratio"] = round(cyan_ratio, 4)
        frames["glass_close.png"]["bright_ratio_check"] = round(bright_ratio, 4)
        frames["glass_close.png"]["dark_ratio_check"] = round(dark_ratio, 4)
        if cyan_ratio < 0.15 or bright_ratio < 0.10 or dark_ratio < 0.025 or _basic_metrics(img)["edge_ratio"] < 0.01:
            failures.append("glass_close_failed")

    if contact_sheet and contact_sheet.exists():
        sheet = _load_image(contact_sheet)
        hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)
        haze_ratio = float(np.mean((hsv[:, :, 0] > 88) & (hsv[:, :, 0] < 118) & (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 70)))
        contrast = float(np.std(cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)))
        contact_metrics = {
            "path": str(contact_sheet),
            "haze_ratio": round(haze_ratio, 4),
            "gray_std": round(contrast, 4),
        }
        if haze_ratio > 0.55 or contrast < 24.0:
            failures.append("contact_sheet_haze_failed")
    else:
        contact_metrics = {"path": str(contact_sheet) if contact_sheet else "", "missing": True}

    return {
        "ok": not failures,
        "failures": failures,
        "frames": frames,
        "contact_sheet": contact_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Blender proof PNG quality gates.")
    parser.add_argument("debug_dir", type=Path)
    parser.add_argument("--contact-sheet", dest="contact_sheet", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = _run_checks(args.debug_dir, args.contact_sheet)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
