"""SPAD active-imaging reflectance module.

提供朗伯漫反射通量计算、镜面反射增益及姿态驱动的完整目标光变曲线。
所有函数均基于 numpy 向量化实现，支持广播机制。
"""

import numpy as np


def lambert_face_flux(
    face_normal,
    sun_unit,
    los_unit,
    area,
    rho,
    solar_irradiance,
    range_m,
    aperture_area_m2,
    filter_bw_nm,
    wavelength_nm,
    receiver_eff,
    quantum_eff,
    phase_function_scale=1.0,
):
    """计算单个面在朗伯反射下的探测计数率 (cps)

    物理模型:
        E_photon = h*c / lambda
        P_recv = (rho * E_sun / pi) * A_face * filter_bw
                 * (A_r / R^2) * phase_scale * cos_i * cos_o
        rate = P_recv / E_photon * eta_r * QE

    Args:
        face_normal: [..., 3] 惯性系法向量
        sun_unit: [..., 3] 太阳方向 (目标→太阳)
        los_unit: [..., 3] 视线方向 (目标→观测器)
        area: 面积 (m^2)
        rho: 漫反射率
        solar_irradiance: 太阳辐照度 (W/m^2/nm)
        range_m: [...] 距离 (米)
        aperture_area_m2: 接收面积 (m^2)
        filter_bw_nm: 滤光片带宽 (nm)
        wavelength_nm: 波长 (nm)
        receiver_eff: 接收效率
        quantum_eff: 量子效率
        phase_function_scale: 相位函数系数

    Returns:
        rate_cps: [...], 计数率 (cps), 最小为 0
    """
    # 物理常数
    h = 6.62607015e-34   # 普朗克常数 (J·s)
    c = 299792458.0       # 光速 (m/s)

    # 光子能量 (J)
    wavelength_m = wavelength_nm * 1.0e-9
    E_photon = h * c / wavelength_m

    # 入射角余弦: cos_i = dot(face_normal, sun_unit)
    cos_i = np.sum(face_normal * sun_unit, axis=-1)

    # 发射角余弦: cos_o = dot(face_normal, los_unit)
    cos_o = np.sum(face_normal * los_unit, axis=-1)

    # 受光和可见条件: 两者都必须 > 0
    visible = (cos_i > 0) & (cos_o > 0)
    cos_i = np.clip(cos_i, 0.0, None)
    cos_o = np.clip(cos_o, 0.0, None)

    # 接收功率 (W)
    P_recv = ((rho * solar_irradiance / np.pi)
              * area
              * filter_bw_nm
              * (aperture_area_m2 / range_m ** 2)
              * phase_function_scale
              * cos_i * cos_o)

    # 计数率 (cps)
    rate_cps = P_recv / E_photon * receiver_eff * quantum_eff

    # 不可见时为 0
    rate_cps = np.where(visible, rate_cps, 0.0)

    return rate_cps


def specular_gain(face_normal, sun_unit, los_unit, width_rad, gain):
    """镜面反射增益

    计算镜面反射方向 r = 2*(n·s)*n - s,
    增益 = peak_gain * exp(-angle(r, los)^2 / (2*width^2)),
    仅在 cos_i > 0 且 cos_r > 0 时有效。

    Args:
        face_normal: [..., 3] 惯性系法向量
        sun_unit: [..., 3] 太阳方向 (目标→太阳)
        los_unit: [..., 3] 视线方向 (目标→观测器)
        width_rad: 镜面反射宽度 (弧度, 高斯 sigma)
        gain: 峰值增益倍数

    Returns:
        gain_factor: [...], 增益系数, 最小为 1.0 (不叠加额外增益)
    """
    # 入射角余弦
    cos_i = np.sum(face_normal * sun_unit, axis=-1)

    # 镜面反射方向: r = 2*(n·s)*n - s
    r = 2.0 * cos_i[..., None] * face_normal - sun_unit

    # 反射方向与视线方向的夹角余弦
    cos_r = np.sum(r * los_unit, axis=-1)

    # 夹角 (弧度)
    cos_r_clipped = np.clip(cos_r, -1.0, 1.0)
    angle_r_los = np.arccos(cos_r_clipped)

    # 有效条件: 面受光 (cos_i > 0) 且反射方向朝向观测器 (cos_r > 0)
    valid = (cos_i > 0) & (cos_r > 0)

    # 高斯增益
    g = gain * np.exp(-angle_r_los ** 2 / (2.0 * width_rad ** 2))

    # 无效时增益为 1.0 (不叠加额外镜面增益)
    gain_factor = np.where(valid, g, 1.0)

    return gain_factor


def target_lightcurve_attitude_driven(
    faces,
    normals_t,
    sun_unit_t,
    los_unit_t,
    range_t,
    solar_irradiance,
    aperture_area_m2,
    filter_bw_nm,
    wavelength_nm,
    receiver_eff,
    quantum_eff,
    phase_function_scale=1.0,
    specular_width_deg=5.0,
):
    """姿态驱动的完整目标光变曲线

    对每个面计算朗伯漫反射和镜面反射贡献，求和得到总计数率。

    Args:
        faces: list of dict (来自 simple_body_model)
        normals_t: [n_frames, n_faces, 3] 各面各时刻惯性系法向量
        sun_unit_t: [n_frames, 3] 或 [3] 太阳方向
        los_unit_t: [n_frames, 3] 或 [3] 视线方向
        range_t: [n_frames] 或 scalar 距离 (米)
        solar_irradiance: 太阳辐照度 (W/m^2/nm)
        aperture_area_m2: 接收面积 (m^2)
        filter_bw_nm: 滤光片带宽 (nm)
        wavelength_nm: 波长 (nm)
        receiver_eff: 接收效率
        quantum_eff: 量子效率
        phase_function_scale: 相位函数系数
        specular_width_deg: 镜面反射宽度 (度)

    Returns:
        signal_scale_t: [n_frames], 归一化信号强度 (相对最大值)
        total_rate_cps: [n_frames], 绝对计数率 (cps)
    """
    n_frames = normals_t.shape[0]
    n_faces = len(faces)

    # 广播 sun_unit_t 和 los_unit_t 到 [n_frames, ...] 形状
    sun_unit_t = np.asarray(sun_unit_t)
    los_unit_t = np.asarray(los_unit_t)
    if sun_unit_t.ndim == 1:
        sun_unit_t = np.broadcast_to(sun_unit_t, (n_frames, 3))
    if los_unit_t.ndim == 1:
        los_unit_t = np.broadcast_to(los_unit_t, (n_frames, 3))

    range_t = np.asarray(range_t)
    if range_t.ndim == 0:
        range_t = np.broadcast_to(range_t, (n_frames,))

    specular_width_rad = np.deg2rad(specular_width_deg)

    # 初始化总计数率
    total_rate_cps = np.zeros(n_frames, dtype=np.float64)

    # 遍历每个面 (面的数量通常很少, 向量化时间轴)
    for j in range(n_faces):
        face = faces[j]
        normal_j = normals_t[:, j, :]  # [n_frames, 3]

        # 漫反射贡献
        lambert_rate = lambert_face_flux(
            face_normal=normal_j,
            sun_unit=sun_unit_t,
            los_unit=los_unit_t,
            area=face["area"],
            rho=face["rho"],
            solar_irradiance=solar_irradiance,
            range_m=range_t,
            aperture_area_m2=aperture_area_m2,
            filter_bw_nm=filter_bw_nm,
            wavelength_nm=wavelength_nm,
            receiver_eff=receiver_eff,
            quantum_eff=quantum_eff,
            phase_function_scale=phase_function_scale,
        )

        # 镜面增益
        spec_gain = specular_gain(
            face_normal=normal_j,
            sun_unit=sun_unit_t,
            los_unit=los_unit_t,
            width_rad=specular_width_rad,
            gain=face["specular"],
        )

        # 镜面贡献 = 漫反射 * (镜面增益 - 1), 加上漫反射部分
        # specular_gain 返回的是乘法因子, 有效时 > 1, 无效时 = 1
        # 总面贡献 = lambert_rate * spec_gain
        face_rate = lambert_rate * spec_gain

        total_rate_cps = total_rate_cps + face_rate

    # 归一化信号强度
    max_rate = np.max(total_rate_cps)
    if max_rate > 0:
        signal_scale_t = total_rate_cps / max_rate
    else:
        signal_scale_t = np.zeros_like(total_rate_cps)

    return signal_scale_t, total_rate_cps
