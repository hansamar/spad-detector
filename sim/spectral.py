"""Spectral helper models for SPAD active-imaging studies.

These helpers are intentionally lightweight. They are suitable for paper-scale
trade studies, but they are not a replacement for full tabulated AM0 spectra or
device-calibrated PDE curves.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
PF32_PDP_CSV = DATA_DIR / "pf32_pdp_digitized.csv"


# First-order AM0 approximation in W m^-2 nm^-1.
# Anchored to the ASTM E490 / AM0 trend: around 1.8-1.9 near 500 nm and
# gradually decaying toward the near infrared.
AM0_WAVELENGTHS_NM = np.array(
    [350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
    dtype=np.float64,
)
AM0_IRRADIANCE_W_M2_NM = np.array(
    [1.15, 1.55, 1.80, 1.90, 1.86, 1.78, 1.66, 1.55, 1.43, 1.30, 1.16, 1.00, 0.86, 0.72, 0.58],
    dtype=np.float64,
)


def _interp_clamped(x: float | np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    return np.interp(x_arr, xp, fp, left=float(fp[0]), right=float(fp[-1]))


@lru_cache(maxsize=1)
def _load_pf32_pdp_table() -> tuple[np.ndarray, np.ndarray]:
    wavelengths: list[float] = []
    fractions: list[float] = []
    with open(PF32_PDP_CSV, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            wavelengths.append(float(row["wavelength_nm"]))
            fractions.append(float(row["pdp_fraction"]))
    return np.asarray(wavelengths, dtype=np.float64), np.asarray(fractions, dtype=np.float64)


def pf32_pdp_fraction(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """PF32 photon detection probability as a 0-1 fraction from the digitized CSV asset."""
    wavelengths_nm, pdp_fraction = _load_pf32_pdp_table()
    return _interp_clamped(wavelength_nm, wavelengths_nm, pdp_fraction)


def am0_solar_irradiance_w_m2_nm(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Approximate AM0 spectral irradiance in W m^-2 nm^-1."""
    return _interp_clamped(wavelength_nm, AM0_WAVELENGTHS_NM, AM0_IRRADIANCE_W_M2_NM)


def scene_stray_color_factor(wavelength_nm: float | np.ndarray) -> np.ndarray:
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    return np.clip(1.05 - 0.00025 * (wavelength_nm - 550.0), 0.85, 1.1)


def reference_channel_response(reference_wavelength_nm: float = 550.0, reference_bandwidth_nm: float = 50.0) -> float:
    return float(am0_solar_irradiance_w_m2_nm(reference_wavelength_nm) * pf32_pdp_fraction(reference_wavelength_nm) * reference_bandwidth_nm)


def relative_channel_response(wavelength_nm: float, bandwidth_nm: float, reference_wavelength_nm: float = 550.0, reference_bandwidth_nm: float = 50.0) -> float:
    numerator = float(am0_solar_irradiance_w_m2_nm(wavelength_nm) * pf32_pdp_fraction(wavelength_nm) * bandwidth_nm)
    denominator = max(reference_channel_response(reference_wavelength_nm, reference_bandwidth_nm), 1e-12)
    return numerator / denominator


def spectral_background_scale(
    component: str,
    wavelength_nm: float,
    bandwidth_nm: float,
    reference_wavelength_nm: float = 550.0,
    reference_bandwidth_nm: float = 50.0,
) -> float:
    """Relative color term for scene ambient stray photons."""
    del component, bandwidth_nm, reference_bandwidth_nm
    color = scene_stray_color_factor(wavelength_nm)
    ref_color = scene_stray_color_factor(reference_wavelength_nm)

    return float(color / max(float(ref_color), 1e-12))
