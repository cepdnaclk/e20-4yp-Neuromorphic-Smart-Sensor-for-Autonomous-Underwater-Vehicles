import math

R_EARTH = 6_378_137.0


def _scale_factors(alpha, mu, beta, eta):
    a1 = math.sqrt(1.0 + math.tan(mu) ** 2 * math.cos(alpha) ** 2)
    a2 = math.sqrt(1.0 + math.tan(eta) ** 2 * math.cos(beta) ** 2)
    return a1, a2


def _solve_l1_l2(alpha, beta, a1, a2, L, z):
    ca_a1 = math.cos(alpha) / a1
    cb_a2 = math.cos(beta) / a2

    numerator = L * ca_a1 - z
    denominator = ca_a1 + cb_a2

    if abs(denominator) < 1e-10:
        raise ValueError(
            f"Degenerate tether geometry: denominator≈0 "
            f"(alpha={math.degrees(alpha):.2f} deg, beta={math.degrees(beta):.2f} deg)"
        )

    l2 = numerator / denominator
    l1 = L - l2

    if l1 < 0 or l2 < 0:
        raise ValueError(
            f"Impossible geometry: l1={l1:.4f}, l2={l2:.4f}, L={L:.4f}, z={z:.4f}"
        )

    return l1, l2


def _north_projection(length, plane_angle, cross_angle):
    cos_cross = math.cos(cross_angle)
    if abs(cos_cross) < 1e-10:
        return 0.0

    denom = math.sqrt(
        math.sin(plane_angle) ** 2 +
        (math.cos(plane_angle) / cos_cross) ** 2
    )
    return length / denom


def _projections(l1, l2, alpha, mu, beta, eta, a1, a2):
    l1x = l1 / a1
    l2x = l2 / a2

    l1y = _north_projection(l1, mu, alpha)
    l2y = _north_projection(l2, eta, beta)

    return l1x, l1y, l2x, l2y


def _local_positions(alpha, mu, beta, eta, l1x, l1y, l2x, l2y):
    x_auv = l1x * math.sin(alpha) + l2x * math.sin(beta)
    y_auv = l1y * math.sin(mu) + l2y * math.sin(eta)
    z_check = l1x * math.cos(alpha) - l2x * math.cos(beta)

    x_ballast = l1x * math.sin(alpha)
    y_ballast = l1y * math.sin(mu)
    z_ballast = l1x * math.cos(alpha)

    return x_auv, y_auv, z_check, x_ballast, y_ballast, z_ballast


def local_offset_to_gps(lat_b, lon_b, dx_east, dy_north):
    lat_rad = math.radians(lat_b)

    lat_out = lat_b + math.degrees(dy_north / R_EARTH)
    lon_out = lon_b + math.degrees(dx_east / (R_EARTH * math.cos(lat_rad)))

    return lat_out, lon_out


def get_auv_position(lat_B, lon_B, alpha, mu, beta, eta, L, z):
    if L <= 0:
        raise ValueError(f"Cable length must be positive, got {L}")
    if z < 0:
        raise ValueError(f"Depth must be non-negative, got {z}")
    if z >= L:
        raise ValueError(f"Depth z={z:.4f} must be less than cable length L={L:.4f}")

    a1, a2 = _scale_factors(alpha, mu, beta, eta)
    l1, l2 = _solve_l1_l2(alpha, beta, a1, a2, L, z)
    l1x, l1y, l2x, l2y = _projections(l1, l2, alpha, mu, beta, eta, a1, a2)

    x_auv, y_auv, z_check, x_ballast, y_ballast, z_ballast = _local_positions(
        alpha, mu, beta, eta, l1x, l1y, l2x, l2y
    )

    lat_auv, lon_auv = local_offset_to_gps(lat_B, lon_B, x_auv, y_auv)
    lat_ballast, lon_ballast = local_offset_to_gps(lat_B, lon_B, x_ballast, y_ballast)

    return {
        "lat_AUV": lat_auv,
        "lon_AUV": lon_auv,
        "depth_AUV": z,
        "dx_east": x_auv,
        "dy_north": y_auv,
        "lat_B_s": lat_ballast,
        "lon_B_s": lon_ballast,
        "depth_B": z_ballast,
        "l1": l1,
        "l2": l2,
        "z_check": z_check,
        "z_error": abs(z - z_check),
    }