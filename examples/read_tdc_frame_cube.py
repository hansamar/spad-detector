"""读取 tdc_frame_cube .bin 文件示例。

用法:
    python examples/read_tdc_frame_cube.py tdc_frame_cube.bin
    python examples/read_tdc_frame_cube.py tdc_frame_cube.bin --meta tdc_frame_cube.metadata.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_tdc_frame_cube(bin_path: str, metadata_path: str | None = None) -> np.ndarray:
    bin_path = Path(bin_path)

    if metadata_path is None:
        meta_candidate = bin_path.with_suffix(".metadata.json")
        if meta_candidate.exists():
            metadata_path = str(meta_candidate)

    if metadata_path is None:
        print("警告: 未找到 metadata.json，请手动指定 shape")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    empty_val = meta.get("empty_pixel_value")
    valid_range = meta.get("valid_tdc_range")
    collision_policy = meta.get("collision_policy", "N/A")

    print(f"TDC frame cube metadata:")
    print(f"  dtype={meta['dtype']}, shape={shape}")
    print(f"  empty_pixel_value={empty_val}")
    print(f"  valid_tdc_range={valid_range}")
    print(f"  collision_policy={collision_policy}")
    print(f"  tdc_bin_width_ns={meta.get('tdc_bin_width_ns', 'N/A')}")
    print(f"  range_bin_m={meta.get('range_bin_m', 'N/A')}")
    print(f"  max_unambiguous_range_m={meta.get('max_unambiguous_range_m', 'N/A')}")

    data = np.fromfile(str(bin_path), dtype=dtype)
    cube = data.reshape(shape)

    # 统计
    if empty_val is not None:
        valid_mask = cube != empty_val
        n_valid = int(np.sum(valid_mask))
        print(f"\nTDC 数据统计:")
        print(f"  有效像素数（有检测）: {n_valid}")
        print(f"  空像素数: {int(np.sum(~valid_mask))}")
        if n_valid > 0:
            valid_bins = cube[valid_mask]
            print(f"  TDC bin 范围: [{int(np.min(valid_bins))}, {int(np.max(valid_bins))}]")
            print(f"  TDC bin 均值: {float(np.mean(valid_bins)):.1f}")

    return cube


def main():
    parser = argparse.ArgumentParser(description="读取 spad-detector tdc_frame_cube .bin 文件")
    parser.add_argument("bin", help="tdc_frame_cube.bin 文件路径")
    parser.add_argument("--meta", default=None, help="metadata.json 路径")
    args = parser.parse_args()

    read_tdc_frame_cube(args.bin, args.meta)


if __name__ == "__main__":
    main()
