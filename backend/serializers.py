"""numpy 数组与自描述 Base64 字符串的编解码

格式: "dtype|shape|base64data"
例:   "float32|(20000,16,16)|AAAA...==="
"""

import base64
import numpy as np


def encode_array(arr: np.ndarray) -> str:
    """将 numpy 数组编码为自描述 Base64 字符串"""
    dtype_str = str(arr.dtype)
    shape_str = str(arr.shape)
    raw = arr.tobytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"{dtype_str}|{shape_str}|{b64}"


def decode_array(s: str) -> np.ndarray:
    """将自描述 Base64 字符串解码为 numpy 数组"""
    parts = s.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"无效的数组编码格式，期望 'dtype|shape|data'，得到: {s[:80]}...")
    dtype_str, shape_str, b64_str = parts
    dtype = np.dtype(dtype_str)
    # 解析 shape: "(20000,16,16)" -> (20000, 16, 16)
    shape = tuple(int(x) for x in shape_str.strip("()").split(",") if x.strip())
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
