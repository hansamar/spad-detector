import numpy as np


def rotation_matrix_axis_angle(axis_3d, angle_rad):
    k = axis_3d / np.linalg.norm(axis_3d)
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    I = np.eye(3)
    return I * np.cos(angle_rad) + (1 - np.cos(angle_rad)) * np.outer(k, k) + np.sin(angle_rad) * K


def spin_rotation_matrix(t, spin_hz, axis_3d, phase0=0.0):
    k = axis_3d / np.linalg.norm(axis_3d)
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    I = np.eye(3)
    theta = 2.0 * np.pi * spin_hz * t + phase0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    one_minus_cos = 1.0 - cos_t
    outer_kk = np.outer(k, k)
    return (
        cos_t[:, None, None] * I[None, :, :]
        + one_minus_cos[:, None, None] * outer_kk[None, :, :]
        + sin_t[:, None, None] * K[None, :, :]
    )


def spin_rotation_matrix_series(t, spin_hz, axis_t, phase0=0.0):
    axis_t = np.asarray(axis_t, dtype=np.float64)
    norms = np.linalg.norm(axis_t, axis=-1, keepdims=True)
    k = axis_t / np.maximum(norms, 1e-12)
    theta = 2.0 * np.pi * spin_hz * np.asarray(t, dtype=np.float64) + phase0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    one_minus_cos = 1.0 - cos_t

    K = np.zeros((k.shape[0], 3, 3), dtype=np.float64)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    outer_kk = np.einsum("ni,nj->nij", k, k)
    I = np.eye(3, dtype=np.float64)[None, :, :]
    return (
        cos_t[:, None, None] * I
        + one_minus_cos[:, None, None] * outer_kk
        + sin_t[:, None, None] * K
    )


def precession_axis_series(t, spin_axis_init, precession_hz, cone_angle_deg):
    k0 = spin_axis_init / np.linalg.norm(spin_axis_init)
    ref = np.array([1.0, 0.0, 0.0]) if abs(k0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, k0) * k0
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(k0, e1)
    cone_rad = np.deg2rad(cone_angle_deg)
    phi = 2.0 * np.pi * precession_hz * t
    axis_t = (
        np.cos(cone_rad) * k0[None, :]
        + np.sin(cone_rad) * (np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :])
    )
    return axis_t / np.linalg.norm(axis_t, axis=-1, keepdims=True)


def simple_body_model(shape="plate"):
    if shape == "sphere":
        return [
            {
                "normal_body": np.array([0.0, 0.0, 1.0]),
                "area": 1.0,
                "rho": 0.3,
                "specular": 0.04,
            }
        ]

    if shape == "plate":
        return [
            {
                "normal_body": np.array([0.0, 0.0, 1.0]),
                "area": 1.0,
                "rho": 0.3,
                "specular": 0.1,
            }
        ]

    if shape == "blade_strip":
        return [
            {
                "normal_body": np.array([0.0, 0.0, 1.0]),
                "area": 1.0,
                "rho": 0.3,
                "specular": 0.18,
            }
        ]

    if shape == "drone_quad":
        normals = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        areas = [0.30, 0.12, 0.12, 0.23, 0.23]
        return [
            {
                "normal_body": n,
                "area": a,
                "rho": 0.24,
                "specular": 0.10,
            }
            for n, a in zip(normals, areas)
        ]

    raise ValueError(f"unknown body shape {shape!r}")


def face_normals_in_inertial(R_bi, faces):
    body_normals = np.array([f["normal_body"] for f in faces])
    return np.einsum("nij,fj->nfi", R_bi, body_normals)


def simple_body_vertices(shape="plate", target_area_m2=1.0):
    """Return a lightweight vertex model whose broadside projected area matches target_area_m2.

    The simulator uses the retained target area as a projected-silhouette reference rather than as
    a catalog-grade mesh parameter. These vertices therefore provide a geometry-aware footprint
    envelope for focal-plane projection without claiming a detailed CAD model.
    """

    scale = float(max(target_area_m2, 0.0)) ** 0.5

    if shape in ("sphere", "plate"):
        half = 0.5 * scale
        return np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float64,
        )

    if shape == "blade_strip":
        # A propeller/blade target is a thin, high-aspect-ratio reflector.
        # The broadside area still matches target_area_m2, but the image-plane
        # footprint remains visibly elongated instead of collapsing to a point.
        aspect = 10.0
        length = np.sqrt(max(target_area_m2, 1e-12) * aspect)
        width = np.sqrt(max(target_area_m2, 1e-12) / aspect)
        hx = length / 2.0
        hy = width / 2.0
        return np.array(
            [
                [-hx, -hy, 0.0],
                [hx, -hy, 0.0],
                [hx, hy, 0.0],
                [-hx, hy, 0.0],
            ],
            dtype=np.float64,
        )

    if shape == "drone_quad":
        # Compact quadrotor footprint: central body, four arms, and rotor disks.
        base = np.array(
            [
                [-0.42, -0.20, 0.0],
                [0.42, -0.20, 0.0],
                [0.42, 0.20, 0.0],
                [-0.42, 0.20, 0.0],
                [-0.78, -0.78, 0.0],
                [-0.54, -0.78, 0.0],
                [-0.54, -0.54, 0.0],
                [-0.78, -0.54, 0.0],
                [0.54, -0.78, 0.0],
                [0.78, -0.78, 0.0],
                [0.78, -0.54, 0.0],
                [0.54, -0.54, 0.0],
                [-0.78, 0.54, 0.0],
                [-0.54, 0.54, 0.0],
                [-0.54, 0.78, 0.0],
                [-0.78, 0.78, 0.0],
                [0.54, 0.54, 0.0],
                [0.78, 0.54, 0.0],
                [0.78, 0.78, 0.0],
                [0.54, 0.78, 0.0],
                [-0.78, 0.0, 0.0],
                [0.78, 0.0, 0.0],
                [0.0, -0.78, 0.0],
                [0.0, 0.78, 0.0],
            ],
            dtype=np.float64,
        )
        return (base * scale).astype(np.float64)

    raise ValueError(f"unknown body shape {shape!r}")
