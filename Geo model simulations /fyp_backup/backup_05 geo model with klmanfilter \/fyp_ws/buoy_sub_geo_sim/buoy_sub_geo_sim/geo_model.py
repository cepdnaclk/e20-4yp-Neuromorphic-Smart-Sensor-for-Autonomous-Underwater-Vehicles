import math
import numpy as np

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
            f"Degenerate tether geometry: alpha={math.degrees(alpha):.2f} deg, "
            f"beta={math.degrees(beta):.2f} deg"
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


def _to_gps(lat_B, lon_B, dx_east, dy_north):
    lat_rad = math.radians(lat_B)

    lat_out = lat_B + math.degrees(dy_north / R_EARTH)
    lon_out = lon_B + math.degrees(dx_east / (R_EARTH * math.cos(lat_rad)))

    return lat_out, lon_out


def gps_to_local_xy(lat0, lon0, lat, lon):
    lat0_rad = math.radians(lat0)
    dy = math.radians(lat - lat0) * R_EARTH
    dx = math.radians(lon - lon0) * R_EARTH * math.cos(lat0_rad)
    return dx, dy


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

    lat_auv, lon_auv = _to_gps(lat_B, lon_B, x_auv, y_auv)
    lat_ballast, lon_ballast = _to_gps(lat_B, lon_B, x_ballast, y_ballast)

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


class AUVKalmanFilter:
    def __init__(self, process_noise=0.3, measurement_noise=1.5):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 100.0

        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.R = np.eye(3) * (measurement_noise ** 2)
        self.q = process_noise
        self.initialized = False

    def predict(self, dt):
        if not self.initialized:
            return

        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        q2 = self.q ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q2 * dt ** 3 / 3.0
            Q[i, i + 3] = q2 * dt ** 2 / 2.0
            Q[i + 3, i] = q2 * dt ** 2 / 2.0
            Q[i + 3, i + 3] = q2 * dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_from_geometry(self, lat_B, lon_B, alpha, mu, beta, eta, L, z):
        raw = get_auv_position(lat_B, lon_B, alpha, mu, beta, eta, L, z)

        z_meas = np.array([
            raw["dx_east"],
            raw["dy_north"],
            raw["depth_AUV"]
        ])

        if not self.initialized:
            self.x[0:3] = z_meas
            self.x[3:6] = 0.0
            self.initialized = True

            self.P = np.eye(6) * 100.0
            self.P[0, 0] = self.R[0, 0]
            self.P[1, 1] = self.R[1, 1]
            self.P[2, 2] = self.R[2, 2]
        else:
            innovation = z_meas - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)

            self.x = self.x + K @ innovation

            I = np.eye(6)
            I_KH = I - K @ self.H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        lat_f, lon_f = _to_gps(lat_B, lon_B, self.x[0], self.x[1])

        filtered = {
            "lat_AUV": lat_f,
            "lon_AUV": lon_f,
            "depth_AUV": self.x[2],
            "dx_east": self.x[0],
            "dy_north": self.x[1],
            "vx": self.x[3],
            "vy": self.x[4],
            "vz": self.x[5],
            "position_std": np.sqrt(np.diag(self.P)[:3]).tolist(),
            "velocity_std": np.sqrt(np.diag(self.P)[3:]).tolist(),
        }

        return {
            "raw": raw,
            "filtered": filtered
        }