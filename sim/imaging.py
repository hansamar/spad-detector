import numpy as np


def psf_weights_gaussian(roi_h, roi_w, cx, cy, sigma_x, sigma_y=None):
    """高斯 PSF 权重

    Args:
        roi_h, roi_w: ROI 尺寸
        cx, cy: 光斑中心 (可以是浮点数)
        sigma_x: X 方向 sigma (像素)
        sigma_y: Y 方向 sigma (None = 与 sigma_x 相同)

    Returns:
        weights: ndarray [roi_h, roi_w], float32, 归一化和为 1
    """
    if sigma_y is None:
        sigma_y = sigma_x

    # 坐标网格: x 对应列方向 (axis=1), y 对应行方向 (axis=0)
    x = np.arange(roi_w, dtype=np.float64) - cx
    y = np.arange(roi_h, dtype=np.float64) - cy

    # 二维高斯: exp(-0.5 * (x^2/sigma_x^2 + y^2/sigma_y^2))
    gauss = np.exp(-0.5 * (x[np.newaxis, :] ** 2 / (sigma_x ** 2)
                            + y[:, np.newaxis] ** 2 / (sigma_y ** 2)))

    # 归一化
    total = np.sum(gauss)
    if total > 0:
        gauss = gauss / total

    return gauss.astype(np.float32)


def jitter_series_white(rng, n_frames, sigma):
    """白噪声抖动序列

    Args:
        rng: numpy Generator
        n_frames: 帧数
        sigma: 标准差 (像素)

    Returns:
        jx, jy: ndarray [n_frames] 各自独立
    """
    jx = rng.normal(0, sigma, size=n_frames)
    jy = rng.normal(0, sigma, size=n_frames)
    return jx, jy


def jitter_series_gauss_markov(rng, n_frames, sigma, corr_time_s, dt):
    """高斯-马尔可夫抖动序列 (时间相关)

    Args:
        rng: numpy Generator
        n_frames: 帧数
        sigma: 目标标准差 (像素)
        corr_time_s: 相关时间 (秒)
        dt: 时间步长 (秒)

    Returns:
        jx, jy: ndarray [n_frames]
    """
    alpha = np.exp(-dt / corr_time_s)
    noise_scale = np.sqrt(1.0 - alpha ** 2) * sigma

    # 初始化
    jx = np.zeros(n_frames, dtype=np.float64)
    jy = np.zeros(n_frames, dtype=np.float64)

    jx[0] = sigma * rng.standard_normal()
    jy[0] = sigma * rng.standard_normal()

    # 一阶低通滤波递推
    for k in range(1, n_frames):
        jx[k] = alpha * jx[k - 1] + noise_scale * rng.standard_normal()
        jy[k] = alpha * jy[k - 1] + noise_scale * rng.standard_normal()

    return jx, jy


def jitter_series_sinusoidal(t, amplitude, freq_hz, phase_x=0.0, phase_y=1.0):
    """正弦抖动序列

    Args:
        t: 时间数组
        amplitude: 振幅 (像素)
        freq_hz: 频率 (Hz)
        phase_x, phase_y: X/Y 方向初相 (弧度)

    Returns:
        jx, jy: ndarray [n_frames]
    """
    t = np.asarray(t, dtype=np.float64)
    jx = amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase_x)
    jy = amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase_y)
    return jx, jy


def target_centroid_series(t, center_x, center_y, drift_x, drift_y, jitter_x, jitter_y):
    """目标质心随时间变化

    Args:
        t: 时间数组
        center_x, center_y: 初始中心位置 (像素)
        drift_x, drift_y: 漂移速率 (像素/秒)
        jitter_x, jitter_y: [n_frames], 抖动偏移 (像素)

    Returns:
        cx_t, cy_t: ndarray [n_frames]
    """
    t = np.asarray(t, dtype=np.float64)
    jitter_x = np.asarray(jitter_x, dtype=np.float64)
    jitter_y = np.asarray(jitter_y, dtype=np.float64)

    cx_t = center_x + drift_x * t + jitter_x
    cy_t = center_y + drift_y * t + jitter_y
    return cx_t, cy_t


def centroid_from_off_axis_series(
    off_axis_x_urad_t,
    off_axis_y_urad_t,
    roi_w,
    roi_h,
    pixel_ifov_urad,
    center_x,
    center_y,
    drift_x=0.0,
    drift_y=0.0,
    t=None,
    jitter_x=None,
    jitter_y=None,
):
    """Map boresight angular offsets to focal-plane centroid positions."""
    off_axis_x_urad_t = np.asarray(off_axis_x_urad_t, dtype=np.float64)
    off_axis_y_urad_t = np.asarray(off_axis_y_urad_t, dtype=np.float64)
    n_frames = off_axis_x_urad_t.size
    if t is None:
        t = np.arange(n_frames, dtype=np.float64)
    else:
        t = np.asarray(t, dtype=np.float64)
    if jitter_x is None:
        jitter_x = np.zeros(n_frames, dtype=np.float64)
    else:
        jitter_x = np.asarray(jitter_x, dtype=np.float64)
    if jitter_y is None:
        jitter_y = np.zeros(n_frames, dtype=np.float64)
    else:
        jitter_y = np.asarray(jitter_y, dtype=np.float64)

    safe_ifov = max(float(pixel_ifov_urad), 1e-6)
    cx_t = center_x + off_axis_x_urad_t / safe_ifov + drift_x * t + jitter_x
    cy_t = center_y + off_axis_y_urad_t / safe_ifov + drift_y * t + jitter_y
    return cx_t, cy_t


def _orthonormal_image_plane_basis(los_unit_t):
    los_unit_t = np.asarray(los_unit_t, dtype=np.float64)
    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (los_unit_t.shape[0], 1))
    alt = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), (los_unit_t.shape[0], 1))
    basis_x = np.cross(ref, los_unit_t)
    degenerate = np.linalg.norm(basis_x, axis=-1) < 1e-8
    if np.any(degenerate):
        basis_x[degenerate] = np.cross(alt[degenerate], los_unit_t[degenerate])
    basis_x = basis_x / np.maximum(np.linalg.norm(basis_x, axis=-1, keepdims=True), 1e-12)
    basis_y = np.cross(los_unit_t, basis_x)
    basis_y = basis_y / np.maximum(np.linalg.norm(basis_y, axis=-1, keepdims=True), 1e-12)
    return basis_x, basis_y


def projected_extent_series_from_vertices(vertices_body, rotation_t, los_unit_t, range_t):
    """Project a lightweight body model to the focal plane and return axis-aligned angular extents."""
    vertices_body = np.asarray(vertices_body, dtype=np.float64)
    rotation_t = np.asarray(rotation_t, dtype=np.float64)
    los_unit_t = np.asarray(los_unit_t, dtype=np.float64)
    range_t = np.asarray(range_t, dtype=np.float64)

    verts_world = np.einsum("nij,vj->nvi", rotation_t, vertices_body)
    basis_x, basis_y = _orthonormal_image_plane_basis(los_unit_t)
    proj_x = np.einsum("nvi,ni->nv", verts_world, basis_x)
    proj_y = np.einsum("nvi,ni->nv", verts_world, basis_y)
    width_urad_t = (np.max(proj_x, axis=1) - np.min(proj_x, axis=1)) / np.maximum(range_t, 1.0) * 1e6
    height_urad_t = (np.max(proj_y, axis=1) - np.min(proj_y, axis=1)) / np.maximum(range_t, 1.0) * 1e6
    return np.maximum(width_urad_t, 1e-6), np.maximum(height_urad_t, 1e-6)


def _pixel_overlap_1d(centers_t, widths_t, n_pix):
    centers_t = np.asarray(centers_t, dtype=np.float64)
    widths_t = np.asarray(widths_t, dtype=np.float64)
    left = centers_t - widths_t / 2.0
    right = centers_t + widths_t / 2.0
    pix_left = np.arange(n_pix, dtype=np.float64) - 0.5
    pix_right = np.arange(n_pix, dtype=np.float64) + 0.5
    overlap = np.minimum(right[:, None], pix_right[None, :]) - np.maximum(left[:, None], pix_left[None, :])
    return np.clip(overlap, 0.0, None)


def _gaussian_kernel_1d(sigma):
    sigma = float(max(sigma, 0.0))
    if sigma <= 1e-8:
        return np.array([1.0], dtype=np.float64)
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _apply_separable_blur(cube, sigma_x, sigma_y):
    if sigma_x <= 1e-8 and sigma_y <= 1e-8:
        return cube
    out = np.asarray(cube, dtype=np.float64).copy()
    if sigma_x > 1e-8:
        kernel_x = _gaussian_kernel_1d(sigma_x)
        pad_x = len(kernel_x) // 2
        padded = np.pad(out, ((0, 0), (0, 0), (pad_x, pad_x)), mode="edge")
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel_x, mode="valid"), 2, padded)
    if sigma_y > 1e-8:
        kernel_y = _gaussian_kernel_1d(sigma_y)
        pad_y = len(kernel_y) // 2
        padded = np.pad(out, ((0, 0), (pad_y, pad_y), (0, 0)), mode="edge")
        out = np.apply_along_axis(lambda col: np.convolve(col, kernel_y, mode="valid"), 1, padded)
    return out


def signal_distribution_cube_projected(
    signal_total_t,
    cx_t,
    cy_t,
    roi_h,
    roi_w,
    width_px_t,
    height_px_t,
    blur_sigma_x=0.0,
    blur_sigma_y=None,
    pde_map=None,
):
    """Redistribute signal photons with a geometric footprint plus optional optical blur."""
    signal_total_t = np.asarray(signal_total_t, dtype=np.float64)
    cx_t = np.asarray(cx_t, dtype=np.float64)
    cy_t = np.asarray(cy_t, dtype=np.float64)
    width_px_t = np.asarray(width_px_t, dtype=np.float64)
    height_px_t = np.asarray(height_px_t, dtype=np.float64)
    n_frames = signal_total_t.size

    if blur_sigma_y is None:
        blur_sigma_y = blur_sigma_x

    x_overlap = _pixel_overlap_1d(cx_t, np.maximum(width_px_t, 1e-6), roi_w)
    y_overlap = _pixel_overlap_1d(cy_t, np.maximum(height_px_t, 1e-6), roi_h)
    footprint = y_overlap[:, :, None] * x_overlap[:, None, :]
    totals = np.sum(footprint, axis=(1, 2), keepdims=True)
    footprint = np.divide(
        footprint,
        np.maximum(totals, 1e-12),
        out=np.zeros_like(footprint),
        where=totals > 0,
    )
    footprint = _apply_separable_blur(footprint, blur_sigma_x, blur_sigma_y)
    totals = np.sum(footprint, axis=(1, 2), keepdims=True)
    footprint = np.divide(
        footprint,
        np.maximum(totals, 1e-12),
        out=np.zeros_like(footprint),
        where=totals > 0,
    )

    signal_cube = signal_total_t[:, None, None] * footprint
    if pde_map is not None:
        signal_cube = signal_cube * np.asarray(pde_map, dtype=np.float64)[None, :, :]

    return signal_cube


def signal_distribution_cube(signal_total_t, cx_t, cy_t, roi_h, roi_w,
                             sigma_x, sigma_y=None, pde_map=None):
    """计算每帧每像素的信号期望 (信号分布立方体)

    Args:
        signal_total_t: [n_frames], 每帧总信号光子期望
        cx_t, cy_t: [n_frames], 每帧光斑中心
        roi_h, roi_w: ROI 尺寸
        sigma_x, sigma_y: PSF sigma
        pde_map: [roi_h, roi_w] 或 None, PDE 分布

    Returns:
        signal_cube: ndarray [n_frames, roi_h, roi_w], 期望值
    """
    signal_total_t = np.asarray(signal_total_t, dtype=np.float64)
    cx_t = np.asarray(cx_t, dtype=np.float64)
    cy_t = np.asarray(cy_t, dtype=np.float64)
    n_frames = len(signal_total_t)

    if pde_map is not None:
        pde_map = np.asarray(pde_map, dtype=np.float64)

    if sigma_y is None:
        sigma_y = sigma_x

    x = np.arange(roi_w, dtype=np.float64)[None, None, :] - cx_t[:, None, None]
    y = np.arange(roi_h, dtype=np.float64)[None, :, None] - cy_t[:, None, None]
    psf = np.exp(-0.5 * (x**2 / (sigma_x**2) + y**2 / (sigma_y**2)))
    totals = np.sum(psf, axis=(1, 2), keepdims=True)
    psf = np.divide(psf, np.maximum(totals, 1e-12), out=np.zeros_like(psf), where=totals > 0)

    signal_cube = signal_total_t[:, None, None] * psf
    if pde_map is not None:
        signal_cube = signal_cube * pde_map[None, :, :]

    return signal_cube
