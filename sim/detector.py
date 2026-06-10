import numpy as np

_torch = None


def _get_torch():
    """惰性导入 torch，仅在首次需要时加载。"""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def make_pde_map(rng: np.random.Generator, roi_h: int, roi_w: int, sigma: float = 0.05, hot_pixel_fraction: float = 0.01, hot_pixel_scale: float = 5.0) -> np.ndarray:
    """生成探测器效率(PDE)分布图，包含不均匀性和热像元

    Args:
        rng: numpy Generator
        roi_h, roi_w: ROI 尺寸
        sigma: 普通像素 PDE 相对标准差
        hot_pixel_fraction: 热像元比例
        hot_pixel_scale: 热像元 PDE 倍率

    Returns:
        pde_map: ndarray [roi_h, roi_w], float32, 均值约1.0
    """
    pde_map = 1.0 + rng.normal(0, sigma, size=(roi_h, roi_w)).astype(np.float32)
    pde_map = np.clip(pde_map, 0.5, 1.5)

    # 热像元: 按 hot_pixel_fraction 的比例随机选取像素
    n_hot = int(round(roi_h * roi_w * hot_pixel_fraction))
    if n_hot > 0:
        hot_indices = np.arange(roi_h * roi_w)
        rng.shuffle(hot_indices)
        hot_pixels = np.unravel_index(hot_indices[:n_hot], (roi_h, roi_w))
        pde_map[hot_pixels] *= hot_pixel_scale
        pde_map = np.clip(pde_map, 0.5, 1.5 * hot_pixel_scale)

    return pde_map


def make_dark_map(rng: np.random.Generator, roi_h: int, roi_w: int, base_rate_cps: float, sigma_frac: float = 0.1,
                  hot_pixel_fraction: float = 0.01, hot_pixel_scale: float = 5.0) -> np.ndarray:
    """生成每像素的暗计数率分布图

    Args:
        rng: numpy Generator
        roi_h, roi_w: ROI 尺寸
        base_rate_cps: 基础暗计数率 (cps)
        sigma_frac: 普通像素暗计数率相对标准差
        hot_pixel_fraction: 热像元比例
        hot_pixel_scale: 热像元暗计数率倍率

    Returns:
        dark_map: ndarray [roi_h, roi_w], float32, 单位 cps
    """
    dark_map = base_rate_cps * (1.0 + rng.normal(0, sigma_frac, size=(roi_h, roi_w)))
    dark_map = np.clip(dark_map, 1e-10, None).astype(np.float32)

    # 热像元
    n_hot = int(round(roi_h * roi_w * hot_pixel_fraction))
    if n_hot > 0:
        hot_indices = np.arange(roi_h * roi_w)
        rng.shuffle(hot_indices)
        hot_pixels = np.unravel_index(hot_indices[:n_hot], (roi_h, roi_w))
        dark_map[hot_pixels] *= hot_pixel_scale

    return dark_map


def apply_dead_time_rate(rate_cps, tau_s, model="nonparalyzable"):
    """将理想计数率转换为经过死时间修正后的有效计数率

    Args:
        rate_cps: 理想计数率 (cps), 可以是标量或数组
        tau_s: 死时间 (秒)
        model: "nonparalyzable" 或 "paralyzable"

    Returns:
        effective_rate: 有效计数率 (cps)
    """
    rate_cps = np.asarray(rate_cps, dtype=np.float64)
    result = np.zeros_like(rate_cps)

    if model == "nonparalyzable":
        # 非延长型: rate_eff = rate / (1 + rate * tau)
        mask = rate_cps > 0
        result[mask] = rate_cps[mask] / (1.0 + rate_cps[mask] * tau_s)
    elif model == "paralyzable":
        # 延长型: rate_eff = rate * exp(-rate * tau)
        mask = rate_cps > 0
        result[mask] = rate_cps[mask] * np.exp(-rate_cps[mask] * tau_s)
    else:
        raise ValueError(f"Unknown dead time model: {model}")

    return result


def sample_poisson_counts(mu, rng):
    """泊松采样, 自动处理 mu <= 0 的情况

    Args:
        mu: 期望值 (可以是标量或数组)
        rng: numpy Generator

    Returns:
        counts: 同形状, 非负整数
    """
    mu = np.asarray(mu, dtype=np.float64)
    result = np.zeros_like(mu, dtype=np.int64)

    mask = mu > 0
    if np.any(mask):
        # np.random.Generator.poisson 支持数组输入
        result[mask] = rng.poisson(mu[mask])

    return result


def _cuda_available():
    try:
        torch = _get_torch()
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _sample_poisson_counts_cuda(mu, seed):
    torch = _get_torch()

    mu_cpu = np.asarray(mu, dtype=np.float32)
    mu_t = torch.as_tensor(np.maximum(mu_cpu, 0.0), device="cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed) & 0xFFFFFFFF)
    samples = torch.poisson(mu_t, generator=generator)
    return samples.to(dtype=torch.int64).cpu().numpy()


def sample_poisson_counts_accelerated(mu, rng, requested_backend="auto", seed=0):
    """Sample Poisson counts and report the backend that actually executed."""

    backend = str(requested_backend or "auto").lower()
    if backend not in {"auto", "cpu", "cuda"}:
        backend = "auto"

    cuda_available = _cuda_available()
    if backend == "cuda" and not cuda_available:
        raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")

    wants_cuda = backend == "cuda" or (backend == "auto" and cuda_available)
    if wants_cuda:
        try:
            return _sample_poisson_counts_cuda(mu, seed), "cuda"
        except Exception as exc:
            if backend == "cuda":
                raise RuntimeError(f"CUDA backend requested but Poisson sampling failed: {exc}") from exc

    return sample_poisson_counts(mu, rng), "cpu"
