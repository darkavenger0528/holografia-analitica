import math


def schwarzschild_radii(mass=1.0):
    rs = 1.0 * mass
    return {
        "event_horizon": rs,
        "photon_sphere": 1.5 * rs,
        "isco": 3.0 * rs,
    }


def shader_params(mass=1.0, spin=0.6, disk_inner=None, disk_outer=6.2, camera_distance=11.0, tilt_deg=4.0):
    radii = schwarzschild_radii(mass)
    return {
        "mass": float(mass),
        "spin": float(spin),
        "event_horizon": float(radii["event_horizon"]),
        "photon_sphere": float(radii["photon_sphere"]),
        "isco": float(radii["isco"]),
        "disk_inner": float(disk_inner if disk_inner is not None else radii["isco"] * 0.95),
        "disk_outer": float(disk_outer),
        "camera_distance": float(camera_distance),
        "tilt_deg": float(tilt_deg),
    }


def gravitational_deflection_approx(r, mass=1.0, photon_sphere=1.5, spin=0.0):
    far_term = (0.22 * mass) / (r + 0.5)
    ring_term = 0.4 * math.exp(-((r - photon_sphere) ** 2) / 0.12)
    spin_term = 0.08 * spin / (r + 0.4)
    return far_term + ring_term + spin_term