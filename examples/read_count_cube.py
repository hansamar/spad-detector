"""读取 count_cube .bin 文件示例。

用法:
    python examples/read_count_cube.py output/backend_jobs/<job_id>_artifacts/counts.bin
    python examples/read_count_cube.py counts.bin --meta counts.metadata.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_count_cube(bin_path: str, metadata_path: str | None = None) -> np.ndarray:
    """从 .bin 文件读取 count_cube。

    Args:
        bin_path: .bin 文件路径
        metadata_path: 配套 .metadata.json 文件路径。
                       若为 None，尝试在同目录下查找。

    Returns:
        np.ndarray: uint16 [n_frames, roi_h, roi_w]
    """
    bin_path = Path(bin_path)

    # 自动查找 metadata
    if metadata_path is None:
        meta_candidate = bin_path.with_suffix(".metadata.json")
        if meta_candidate.exists():
            metadata_path = str(meta_candidate)

    if metadata_path is not None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["dtype"])
        n_frames, roi_h, roi_w = shape
        print(f"从 metadata 读取: dtype={meta['dtype']}, shape={shape}")
        print(f"  n_frames={n_frames}, roi_h={roi_h}, roi_w={roi_w}")
        print(f"  format={meta.get('format', 'unknown')}")
        print(f"  sample_rate_hz={meta.get('sample_rate_hz', 'N/A')}")
        print(f"  detector_preset={meta.get('detector_preset', 'N/A')}")
        print(f"  dead_time_ns={meta.get('dead_time_ns', 'N/A')}")
        print(f"  random_seed={meta.get('random_seed', 'N/A')}")
    else:
        print("警告: 未找到 metadata.json，请手动指定 shape/dtype")
        sys.exit(1)

    data = np.fromfile(str(bin_path), dtype=dtype)
    cube = data.reshape(shape)

    print(f"\ncount_cube 加载完成:")
    print(f"  总光子数: {int(np.sum(cube))}")
    print(f"  每帧均值: {float(np.mean(cube)):.2f}")
    print(f"  最大值: {int(np.max(cube))}")
    print(f"  有检测的像素比例: {float(np.mean(cube > 0)):.4f}")

    return cube


def main():
    parser = argparse.ArgumentParser(description="读取 spad-detector count_cube .bin 文件")
    parser.add_argument("bin", help="counts.bin 文件路径")
    parser.add_argument("--meta", default=None, help="metadata.json 路径（可选，默认同目录查找）")
    args = parser.parse_args()

    read_count_cube(args.bin, args.meta)


if __name__ == "__main__":
    main()
