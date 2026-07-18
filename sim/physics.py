import numpy as np


def atmospheric_attenuation_coefficient_km(wavelength_nm, visibility_km: float = 23.0):
    """Return aerosol + molecular extinction coefficient in 1/km."""
    safe_visibility = max(float(visibility_km), 0.5)
    safe_wavelength = np.clip(np.asarray(wavelength_nm, dtype=np.float64), 350.0, 1800.0)

    if safe_visibility > 50:
        q = 1.6
    elif safe_visibility > 6:
        q = 1.3
    else:
        q = 0.585 * safe_visibility ** (1.0 / 3.0)

    aerosol = (3.912 / safe_visibility) * (safe_wavelength / 550.0) ** (-q)
    molecular = 0.006 * (safe_wavelength / 550.0) ** (-4.08)
    return aerosol + molecular


def atmospheric_transmittance(wavelength_nm, visibility_km: float, distance_m, enabled: bool = True):
    """Two-way atmospheric transmittance for an active/receiver optical path."""
    distance = np.maximum(np.asarray(distance_m, dtype=np.float64), 0.0)
    if not enabled:
        return np.ones_like(distance, dtype=np.float64)

    alpha_m = atmospheric_attenuation_coefficient_km(wavelength_nm, visibility_km) / 1000.0
    return np.exp(-2.0 * alpha_m * distance)


def aperture_area(diameter_m: float) -> float:
    """计算圆孔径接收面积"""
    return np.pi * (diameter_m / 2.0) ** 2


def haze_scatter_scale_from_visibility(visibility_km: float = 23.0) -> float:
    """Return a mild ambient-scatter multiplier from visibility."""
    safe_visibility = max(float(visibility_km), 0.5)
    return float(1.0 + min(3.0, max(0.0, 23.0 / safe_visibility - 1.0)) * 0.22)


def solar_environment_detected_rate_cps_per_pixel(
    irradiance_w_m2_nm: float,
    filter_bandwidth_nm: float,
    wavelength_nm: float,
    aperture_diameter_m: float,
    receiver_efficiency: float,
    quantum_efficiency: float,
    pixel_ifov_urad: float,
    stray_light_rejection_ratio: float,
    visibility_km: float = 23.0,
    ambient_reflectance: float = 0.18,
) -> float:
    """Solar-driven scene/airlight background entering one detector pixel.

    The model treats the non-target environment inside the pixel IFOV as a
    Lambertian ambient radiance term. Optical baffling and filtering losses are
    represented by the stray-light rejection ratio before photon detection.
    """
    solar_irradiance = max(float(irradiance_w_m2_nm), 0.0)
    bandwidth = max(float(filter_bandwidth_nm), 0.0)
    if solar_irradiance == 0.0 or bandwidth == 0.0:
        return 0.0

    pixel_ifov_rad = max(float(pixel_ifov_urad), 0.0) * 1e-6
    pixel_solid_angle_sr = pixel_ifov_rad * pixel_ifov_rad
    radiance_w_m2_sr_nm = max(float(ambient_reflectance), 0.0) * solar_irradiance / np.pi
    rejection = max(float(stray_light_rejection_ratio), 1.0)
    optical_power_w = (
        radiance_w_m2_sr_nm
        * bandwidth
        * aperture_area(aperture_diameter_m)
        * pixel_solid_angle_sr
        * haze_scatter_scale_from_visibility(visibility_km)
        / rejection
    )
    detected_rate = (
        optical_power_w
        / max(photon_energy_joule(wavelength_nm), 1e-30)
        * max(float(receiver_efficiency), 0.0)
        * max(float(quantum_efficiency), 0.0)
    )
    return max(float(detected_rate), 0.0)


def make_outage_mask(n_frames: int, outage_fraction: float, seed: int = 123):
    """生成随机的信号中断/遮挡掩码"""
    if outage_fraction <= 0:
        return np.ones(n_frames, dtype=np.float32)
    rng = np.random.default_rng(seed)
    mask = np.ones(n_frames, dtype=np.float32)

    total_out = int(n_frames * outage_fraction)
    if total_out <= 0:
        return mask

    remaining = total_out
    while remaining > 0:
        # 自适应段长: 短序列时减小最小段长
        min_seg = max(1, min(100, n_frames // 10))
        max_seg = min(2000, remaining + 1)
        if remaining <= min_seg:
            seg_len = remaining
        else:
            seg_len = int(rng.integers(min_seg, max_seg))

        start = int(rng.integers(0, max(1, n_frames - seg_len + 1)))
        mask[start:start + seg_len] = 0.0
        remaining -= seg_len
    return mask


def make_visibility_mask(
    t: np.ndarray,
    outage_fraction: float,
    seed: int = 123,
    mode: str = "random_segments",
    period_s: float = 6.0,
    on_fraction: float = 0.33,
    gap_start_fraction: float = 0.33,
    gap_end_fraction: float = 0.66,
):
    """Build structured visibility masks for interruption-focused studies."""
    n_frames = int(t.size)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    if outage_fraction <= 0 and mode == "random_segments":
        return np.ones(n_frames, dtype=np.float32)

    mode = (mode or "random_segments").lower()
    if mode == "random_segments":
        return make_outage_mask(n_frames, outage_fraction, seed=seed)

    mask = np.ones(n_frames, dtype=np.float32)
    if mode == "mid_gap":
        start = int(np.clip(round(gap_start_fraction * n_frames), 0, n_frames - 1))
        end = int(np.clip(round(gap_end_fraction * n_frames), start + 1, n_frames))
        mask[start:end] = 0.0
        return mask

    if mode == "periodic":
        safe_period = max(float(period_s), float(t[1] - t[0]) if n_frames > 1 else 1.0)
        safe_on = float(np.clip(on_fraction, 1e-3, 1.0))
        phase = np.mod(t - float(t[0]), safe_period) / safe_period
        mask[phase > safe_on] = 0.0
        return mask

    if mode == "dual_window":
        left_end = int(np.clip(round(0.25 * n_frames), 1, n_frames))
        right_start = int(np.clip(round(0.65 * n_frames), 0, n_frames - 1))
        mask[:] = 0.0
        mask[:left_end] = 1.0
        mask[right_start:] = 1.0
        return mask

    return make_outage_mask(n_frames, outage_fraction, seed=seed)


def modulation_series(t, tumbling_hz, m1, m2, m3, p1, p2, p3, slow_depth, slow_hz):
    """计算目标强度随时间的变化序列（包含快滚转谐波和慢进动包络）"""
    fast = (
        1.0
        + m1 * np.cos(2 * np.pi * tumbling_hz * t + p1)
        + m2 * np.cos(2 * np.pi * 2 * tumbling_hz * t + p2)
        + m3 * np.cos(2 * np.pi * 3 * tumbling_hz * t + p3)
    )
    fast = np.maximum(fast, 0.0)

    slow = 1.0 + slow_depth * np.cos(2 * np.pi * slow_hz * t)
    slow = np.maximum(slow, 0.0)

    return fast * slow


def photon_energy_joule(wavelength_nm: float) -> float:
    """计算特定波长下单个光子能量 (焦耳)"""
    h = 6.62607015e-34
    c = 299792458.0
    lam = wavelength_nm * 1e-9
    return h * c / lam


def target_detected_rate_cps(
    irradiance_w_m2_nm: float,
    target_area_m2: float,
    target_reflectivity: float,
    phase_function_scale: float,
    range_m: float,
    aperture_diameter_m: float,
    filter_bandwidth_nm: float,
    wavelength_nm: float,
    receiver_efficiency: float,
    quantum_efficiency: float,
):
    """简化朗伯反射模型下，整个目标到达接收系统并被探测的平均计数率 (cps)"""
    Ar = aperture_area(aperture_diameter_m)
    Egamma = photon_energy_joule(wavelength_nm)

    # 目标等效谱辐亮度 -> 接收功率
    # P_recv = (rho * E_sun / pi) * A_t * dlambda * (Ar / R^2) * phase
    P_recv = (
        (target_reflectivity * irradiance_w_m2_nm / np.pi)
        * target_area_m2
        * filter_bandwidth_nm
        * (Ar / max(range_m, 1.0) ** 2)
        * max(phase_function_scale, 0.0)
    )

    if P_recv < 0:
        P_recv = 0.0

    photon_rate = P_recv / max(Egamma, 1e-30)
    detected_rate = photon_rate * receiver_efficiency * quantum_efficiency
    return max(detected_rate, 0.0)


def effective_laser_average_power_w(
    laser_mode: str,
    laser_average_power_w: float,
    laser_pulse_energy_j: float,
    laser_repetition_frequency_hz: float,
) -> float:
    """返回发射机平均功率（CW 或脉冲模式）。

    脉冲模式下优先使用 pulse_energy × rep_rate；若 pulse_energy 为零则从
    laser_average_power_w 回退推导，避免用户同时设置两者时的静默忽略。
    """
    if str(laser_mode or "").lower() == "pulsed":
        pulse_energy = float(laser_pulse_energy_j)
        rep_rate = max(float(laser_repetition_frequency_hz), 0.0)
        if pulse_energy > 0:
            return pulse_energy * rep_rate
        return max(float(laser_average_power_w), 0.0)
    return max(float(laser_average_power_w), 0.0)


def laser_target_detected_rate_cps(
    *,
    laser_mode: str,
    laser_average_power_w: float,
    laser_pulse_energy_j: float,
    laser_repetition_frequency_hz: float,
    transmitter_divergence_mrad: float,
    target_area_m2: float,
    target_reflectivity: float,
    phase_function_scale: float,
    range_m,
    aperture_diameter_m: float,
    wavelength_nm: float,
    receiver_efficiency: float,
    quantum_efficiency: float,
):
    """Approximate detected laser-return rate for an on-axis Lambertian target.

    The transmitter divergence is a full-angle Gaussian-beam engineering
    parameter. The intercepted fraction uses the on-axis small-target
    approximation and is capped at one so received energy cannot exceed the
    transmitted laser-power budget.
    """
    power_w = effective_laser_average_power_w(
        laser_mode,
        laser_average_power_w,
        laser_pulse_energy_j,
        laser_repetition_frequency_hz,
    )
    safe_range_m = np.maximum(np.asarray(range_m, dtype=np.float64), 1.0)
    half_angle_rad = max(float(transmitter_divergence_mrad), 1e-9) * 1e-3 / 2.0
    beam_radius_m = np.maximum(safe_range_m * np.tan(half_angle_rad), 1e-9)
    beam_area_m2 = np.pi * beam_radius_m**2
    intercepted_fraction = np.clip(
        2.0 * max(float(target_area_m2), 0.0) / np.maximum(beam_area_m2, 1e-24),
        0.0,
        1.0,
    )
    received_power_w = (
        power_w
        * intercepted_fraction
        * max(float(target_reflectivity), 0.0)
        * (aperture_area(aperture_diameter_m) / (np.pi * safe_range_m**2))
        * max(float(phase_function_scale), 0.0)
    )
    detected_rate = (
        received_power_w
        / max(photon_energy_joule(wavelength_nm), 1e-30)
        * max(float(receiver_efficiency), 0.0)
        * max(float(quantum_efficiency), 0.0)
    )
    detected_rate = np.maximum(detected_rate, 0.0)
    return float(detected_rate) if detected_rate.ndim == 0 else detected_rate
