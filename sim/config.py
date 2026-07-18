from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpticalParams:
    wavelength_nm: float = 780.0
    aperture_diameter_m: float = 0.025
    receiver_efficiency: float = 0.05
    quantum_efficiency: float = 0.07
    filter_bandwidth_nm: float = 10.0
    detector_fov_urad: float = 50.0 * 3.141592653589793 / 180.0 * 1e6
    stray_light_rejection_ratio: float = 1.0
    detector_off_axis_urad_x: float = 0.0
    detector_off_axis_urad_y: float = 0.0
    atmospheric_attenuation_enabled: bool = True
    atmospheric_visibility_km: float = 23.0
    atmospheric_path_length_m: float = 6_000.0


@dataclass
class SpadParams:
    dark_count_rate_cps: float = 100.0
    dead_time_ns: float = 20.0
    timing_jitter_ns: float = 0.2 / 2.355
    pde_nonuniform_sigma: float = 0.05
    hot_pixel_fraction: float = 0.01
    hot_pixel_scale: float = 5.0
    dark_count_sigma: float = 0.1
    dead_time_model: str = "nonparalyzable"
    afterpulse_probability: float = 0.0
    crosstalk_probability: float = 0.0
    tdc_bin_width_ns: float = 0.055
    max_count_per_frame: int = 65535
    irf_fwhm_ps: float = 200.0
    max_count_rate_cps_per_pixel: float = 20e6


@dataclass
class BackgroundParams:
    scene_stray_rate_cps_per_pixel: float = 4.0
    background_noise_mode: str = "solar_environment"
    manual_signal_background_ratio: float = 10.0
    temporal_drift_depth: float = 0.05
    temporal_drift_hz: float = 0.03
    spatial_nonuniformity_sigma: float = 0.0
    gradient_x: float = 0.0
    gradient_y: float = 0.0


@dataclass
class IlluminationParams:
    mode: str = "laser_plus_solar"
    laser_mode: str = "pulsed"
    laser_average_power_w: float = 1e-6
    laser_pulse_energy_j: float = 1e-12
    laser_repetition_frequency_hz: float = 1e6
    laser_pulse_width_ns: float = 1.0
    transmitter_divergence_mrad: float = 1.0


@dataclass
class TargetParams:
    target_area_m2: float = 0.025
    target_length_m: float = 0.5
    target_width_m: float = 0.05
    target_height_m: float = 0.002
    propeller_diameter_m: float = 0.10
    target_reflectivity: float = 0.1
    propeller_reflectivity: float = 0.3
    solar_irradiance_w_m2_nm: float = 1.35
    phase_function_scale: float = 1.0
    phase_function_model: str = "lambert"
    specular_fraction: float = 0.0
    specular_width_deg: float = 5.0
    target_range_m: float = 2.0
    reference_range_m: float = 2.0
    target_radial_velocity_mps: float = 0.0
    target_radial_accel_mps2: float = 0.0
    spin_hz: float = 200.0
    precession_hz: float = 0.0
    spin_axis_elevation_deg: float = 45.0
    spin_axis_azimuth_deg: float = 0.0
    body_shape: str = "blade_strip"
    tumbling_hz: float = 200.0
    modulation_depth: float = 0.35
    harmonic2_depth: float = 0.12
    harmonic3_depth: float = 0.04
    phase1: float = 0.0
    phase2: float = 0.7
    phase3: float = 1.1
    slow_envelope_depth: float = 0.15
    slow_envelope_hz: float = 5.0
    outage_fraction: float = 0.0
    outage_seed: int = 123
    outage_mode: str = "random_segments"
    outage_period_s: float = 6.0
    outage_on_fraction: float = 0.33
    outage_gap_start_fraction: float = 0.33
    outage_gap_end_fraction: float = 0.66
    glint_probability: float = 0.0
    glint_gain: float = 2.0
    orientation_yaw_deg: float = 0.0
    orientation_pitch_deg: float = 0.0
    orientation_roll_deg: float = 0.0
    custom_shape_x: Any = None
    custom_shape_y: Any = None
    custom_shape_intensity: Any = None
    custom_shape_aspect_ratio: float = 1.0


@dataclass
class ImageParams:
    roi_w: int = 32
    roi_h: int = 32
    scene_mode: str = "centered_roi"
    center_x: float = 15.5
    center_y: float = 15.5
    initial_center_x: float = -1.0
    initial_center_y: float = -1.0
    edge_entry_mode: str = "manual"
    spot_sigma_pixels: float = 0.8
    jitter_sigma_pixels: float = 0.12
    drift_pixels_per_s_x: float = 0.0
    drift_pixels_per_s_y: float = 0.0
    psf_model: str = "projected_footprint"
    spot_sigma_x_pixels: float = 0.0
    spot_sigma_y_pixels: float = 0.0
    jitter_model: str = "white"
    jitter_corr_time_s: float = 0.01
    subpixel_truth_enabled: bool = True
    pixel_pitch_um: float = 50.0
    fill_factor: float = 0.015
    microlens_gain: float = 13.3


@dataclass
class GeometryParams:
    sun_elevation_deg: float = 30.0
    sun_azimuth_deg: float = 0.0
    sun_rate_deg_per_s: float = 0.0
    los_elevation_deg: float = 0.0
    los_azimuth_deg: float = 0.0
    los_rate_deg_per_s: float = 0.0
    phase_angle_override_deg: float = 60.0
    phase_angle_rate_deg_per_s: float = 0.0
    angular_rate_urad_s_x: float = 0.0
    angular_rate_urad_s_y: float = 0.0
    tracking_residual_sigma_urad: float = 0.0
    trajectory_name: str = ""
    reacquire_enabled: bool = False
    reacquire_start_fraction: float = 0.35
    reacquire_end_fraction: float = 0.55
    detector_position_x_m: float = 0.0
    detector_position_y_m: float = 0.0
    detector_position_z_m: float = 0.0
    detector_yaw_deg: float = 0.0
    detector_pitch_deg: float = 0.0
    external_target_x_m: Any = None
    external_target_y_m: Any = None
    external_target_z_m: Any = None
    external_yaw_deg: Any = None
    external_pitch_deg: Any = None
    external_roll_deg: Any = None
    external_spin_hz: Any = None
    external_rotation_phase_rad: Any = None
    external_propeller_spin_hz: Any = None
    external_propeller_phase_rad: Any = None
    external_range_m: Any = None
    external_sun_unit: Any = None
    external_los_unit: Any = None
    external_phase_angle_rad: Any = None
    external_off_axis_x_urad: Any = None
    external_off_axis_y_urad: Any = None
    external_in_fov: Any = None
    external_visibility: Any = None


@dataclass
class SimParams:
    observation_time_s: float = 60.0
    sample_rate_hz: float = 50.0
    seed: int = 42
    compute_backend: str = "auto"
    detector_preset: str = "pf32"
    simulation_tier: str = "physics_informed"
    simulation_mode: str = "frame"
    lightcurve_mode: str = "attitude_driven"
    summary_only: bool = False
    save_event_list: bool = False
    save_truth_series: bool = False
    save_intermediate_series: bool = False
    optical: OpticalParams = field(default_factory=OpticalParams)
    spad: SpadParams = field(default_factory=SpadParams)
    background: BackgroundParams = field(default_factory=BackgroundParams)
    illumination: IlluminationParams = field(default_factory=IlluminationParams)
    target: TargetParams = field(default_factory=TargetParams)
    image: ImageParams = field(default_factory=ImageParams)
    geometry: GeometryParams = field(default_factory=GeometryParams)
