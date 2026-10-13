"""Evaluate the proposed method across one or more cover images.

Example:
    python evaluate_proposed.py cover1.png cover2.jpg --repeats 3
"""
import argparse

import numpy as np

import stego_core as sc


def _summary(values):
    if not values:
        return "n/a"
    mean = float(np.mean(values))
    if len(values) == 1:
        return f"{mean:.3f} (n=1)"
    return f"{mean:.3f} +/- {np.std(values, ddof=1):.3f} (n={len(values)})"


def main():
    parser = argparse.ArgumentParser(description="Compare sequential LSB with DIP-guided adaptive LSB.")
    parser.add_argument("covers", nargs="+", help="one or more PNG/BMP/JPEG cover images")
    parser.add_argument("--repeats", type=int, default=3, help="detector seeds per cover (default: 3)")
    parser.add_argument("--seed", type=int, default=0, help="base random seed (default: 0)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    fractions = [0.05, 0.10, 0.20, 0.40, 0.60]
    methods = ("seq", "dip")
    detector_methods = ("seq", "match", "edge", "dip")
    measures = {
        fraction: {
            method: {key: [] for key in ("psnr", "ssim", "smooth")}
            for method in methods
        }
        for fraction in fractions
    }
    detection = {fraction: {method: [] for method in detector_methods} for fraction in fractions}

    for cover_index, path in enumerate(args.covers):
        cover = sc.load_image(path)
        smooth = ~sc.dip_stages(cover)["mask"]
        full_capacity = sc.capacity_bytes(cover, "seq")
        safe_capacity = sc.secure_capacity_bytes(cover, "dip")
        print(f"{path}: shape={cover.shape}, full={full_capacity} B, textured={safe_capacity} B")

        for fraction in fractions:
            data_seed = args.seed + cover_index * 1_000_003 + int(fraction * 1000)
            data_rng = np.random.default_rng(data_seed)
            data = data_rng.integers(0, 256, max(8, int(full_capacity * fraction)), dtype=np.uint8).tobytes()
            for method in methods:
                stego, _ = sc.embed(cover, data, sc.KIND_IMAGE, method, "", rng_seed=data_seed)
                quality, _ = sc.quality(cover, stego)
                changed = stego != cover
                measures[fraction][method]["psnr"].append(quality["PSNR (dB)"])
                measures[fraction][method]["ssim"].append(quality["SSIM"])
                measures[fraction][method]["smooth"].append(
                    100.0 * float((changed & smooth).sum()) / max(1, int(changed.sum()))
                )

        for repeat in range(args.repeats):
            detector_seed = args.seed + cover_index * 1_000_003 + repeat
            result, _ = sc.detector_accuracy(cover, detector_methods, fractions, seed=detector_seed)
            if result is None:
                continue
            for index, fraction in enumerate(fractions):
                for method in detector_methods:
                    detection[fraction][method].append(result[method][index])

    print(f"\nAggregate results: {len(args.covers)} cover(s); detector repeats per cover={args.repeats}")
    print(f"{'payload':>8} | {'method':>6} | {'PSNR dB (mean +/- SD)':>25} | {'SSIM (mean +/- SD)':>22} | {'smooth changes %':>20}")
    for fraction in fractions:
        for method in methods:
            values = measures[fraction][method]
            print(f"{fraction:7.0%} | {method:>6} | {_summary(values['psnr']):>25} | "
                  f"{_summary(values['ssim']):>22} | {_summary(values['smooth']):>20}")

    print("\nSPAM/LDA detector accuracy (50% = chance; mean +/- SD across cover/seed runs)")
    print(f"{'payload':>8} | " + " | ".join(f"{method:>22}" for method in detector_methods))
    for fraction in fractions:
        print(f"{fraction:7.0%} | " + " | ".join(
            f"{_summary(detection[fraction][method]):>22}" for method in detector_methods
        ))


if __name__ == "__main__":
    main()