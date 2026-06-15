"""读取 event_list .npz 文件示例。

用法:
    python examples/read_event_list.py events.npz
    python examples/read_event_list.py events.npz --meta events.metadata.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def read_event_list(npz_path: str, metadata_path: str | None = None) -> dict:
    npz_path = Path(npz_path)

    if metadata_path is None:
        meta_candidate = npz_path.with_suffix(".metadata.json")
        if meta_candidate.exists():
            metadata_path = str(meta_candidate)

    data = np.load(str(npz_path))
    event_dict = {key: data[key] for key in data.files}
    data.close()

    print(f"event_list 加载完成:")
    print(f"  总事件数: {len(event_dict['event_times_s'])}")
    print(f"  字段: {list(event_dict.keys())}")

    # 各来源统计
    if "event_source" in event_dict:
        sources = event_dict["event_source"]
        source_labels = {0: "unknown", 1: "signal", 2: "background", 3: "dark", 4: "afterpulse", 5: "crosstalk"}
        print(f"\n事件来源分布:")
        for code in sorted(np.unique(sources)):
            count = int(np.sum(sources == code))
            label = source_labels.get(int(code), f"code_{code}")
            print(f"  {label:>12}: {count:>8} ({count / len(sources) * 100:.1f}%)")

    # 验证 pixel 一致性
    if "event_pixel" in event_dict and "event_row" in event_dict and "event_col" in event_dict:
        # 从 metadata 获取 roi_w
        roi_w = None
        if metadata_path is not None:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            roi_w = meta.get("roi_w")

        if roi_w is not None:
            expected_pixels = event_dict["event_row"].astype(np.int32) * roi_w + event_dict["event_col"].astype(np.int32)
            mismatches = np.sum(expected_pixels != event_dict["event_pixel"])
            if mismatches == 0:
                print(f"\n✓ event_pixel 与 row/col 一致性检查通过 (roi_w={roi_w})")
            else:
                print(f"\n✗ event_pixel 不一致: {mismatches} 条不匹配")

    # 时间范围
    times = event_dict["event_times_s"]
    if len(times) > 0:
        print(f"\n时间范围: [{float(np.min(times)):.6f}, {float(np.max(times)):.6f}] s")

    # TDC bin 范围
    if "event_tof_bins" in event_dict and len(event_dict["event_tof_bins"]) > 0:
        bins = event_dict["event_tof_bins"]
        print(f"TDC bin 范围: [{int(np.min(bins))}, {int(np.max(bins))}]")

    if metadata_path is not None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        gen = meta.get("event_generation", "N/A")
        warning = meta.get("event_generation_warning", "")
        print(f"\n事件生成方式: {gen}")
        if warning:
            print(f"注意: {warning}")

    return event_dict


def main():
    parser = argparse.ArgumentParser(description="读取 spad-detector event_list .npz 文件")
    parser.add_argument("npz", help="events.npz 文件路径")
    parser.add_argument("--meta", default=None, help="metadata.json 路径")
    args = parser.parse_args()

    read_event_list(args.npz, args.meta)


if __name__ == "__main__":
    main()
