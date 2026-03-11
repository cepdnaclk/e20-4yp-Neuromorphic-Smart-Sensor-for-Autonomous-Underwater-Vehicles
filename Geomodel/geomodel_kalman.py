"""
AUV Kalman Filter -- Smoothed position tracking
=================================================

Wraps the geometric model (geomodel_clean.get_auv_position) with a
linear Kalman filter to:
  - Smooth noisy sensor readings over time
  - Estimate AUV velocity (not directly measured)
  - Predict position between measurement updates

State vector (6D):
  x = [x_east, y_north, z_depth, vx, vy, vz]

Motion model: constant-velocity  (F is identity + dt*I in upper-right)
Measurement model: position-only (H picks out x, y, z)

Dependencies: numpy, geomodel_clean.py (same directory)
"""

import math
import numpy as np
from geomodel_clean import get_auv_position, _to_gps, R_EARTH


# ---------------------------------------------
# Kalman Filter Class
# ---------------------------------------------

class AUVKalmanFilter:
    """
    6-state Kalman filter for AUV position tracking.

    States: [x_east, y_north, z_depth, vx, vy, vz]
      - x, y, z : AUV offset from buoy (metres)
      - vx, vy, vz : AUV velocity (m/s)

    Measurements: [dx_east, dy_north, depth_AUV] from get_auv_position()
    """

    def __init__(self, process_noise=0.5, measurement_noise=2.0):
        """
        Parameters
        ----------
        process_noise : float
            Std-dev of process noise (m/s^2 acceleration uncertainty).
            Higher = filter trusts measurements more.
        measurement_noise : float
            Std-dev of measurement noise (m).
            Higher = filter trusts its prediction more.
        """
        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)

        # State covariance -- start with high uncertainty
        self.P = np.eye(6) * 100.0

        # Measurement matrix: we observe [x, y, z] only
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0  # x_east
        self.H[1, 1] = 1.0  # y_north
        self.H[2, 2] = 1.0  # z_depth

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise**2

        # Process noise intensity (will be scaled by dt in predict)
        self._q = process_noise

        # Track whether filter has been initialised with a measurement
        self._initialised = False

    def predict(self, dt):
        """
        Time-update (predict) step.

        Parameters
        ----------
        dt : float
            Time since last update (seconds).
        """
        if not self._initialised:
            return

        # State transition matrix (constant velocity)
        F = np.eye(6)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt

        # Process noise covariance (discrete white-noise acceleration model)
        # Q models random acceleration: position gets dt^3/3, cross gets dt^2/2,
        # velocity gets dt
        q = self._q**2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i]     = q * dt**3 / 3.0   # position variance
            Q[i, i+3]   = q * dt**2 / 2.0   # position-velocity cross
            Q[i+3, i]   = q * dt**2 / 2.0   # velocity-position cross
            Q[i+3, i+3] = q * dt             # velocity variance

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, lat_B, lon_B, alpha, mu, beta, eta, L, z):
        """
        Measurement-update step.

        Calls get_auv_position() to get a raw position fix, then
        fuses it with the predicted state.

        Parameters
        ----------
        (same as get_auv_position)

        Returns
        -------
        dict with:
            raw  : dict from get_auv_position (unfiltered)
            filtered : dict with smoothed position, velocity, uncertainty
        """
        # Get raw geometric fix
        raw = get_auv_position(lat_B, lon_B, alpha, mu, beta, eta, L, z)

        # Measurement vector
        z_meas = np.array([raw['dx_east'], raw['dy_north'], raw['depth_AUV']])

        if not self._initialised:
            # First measurement: initialise state directly
            self.x[0] = z_meas[0]  # x_east
            self.x[1] = z_meas[1]  # y_north
            self.x[2] = z_meas[2]  # z_depth
            self.x[3] = 0.0        # vx (unknown)
            self.x[4] = 0.0        # vy
            self.x[5] = 0.0        # vz
            self._initialised = True
            # Keep high covariance for velocity
            self.P = np.eye(6) * 100.0
            self.P[0, 0] = self.R[0, 0]
            self.P[1, 1] = self.R[1, 1]
            self.P[2, 2] = self.R[2, 2]
        else:
            # Innovation (measurement residual)
            y = z_meas - self.H @ self.x

            # Innovation covariance
            S = self.H @ self.P @ self.H.T + self.R

            # Kalman gain
            K = self.P @ self.H.T @ np.linalg.inv(S)

            # State update
            self.x = self.x + K @ y

            # Covariance update (Joseph form for numerical stability)
            I_KH = np.eye(6) - K @ self.H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Build filtered result
        filtered = self._build_result(lat_B, lon_B)

        return {
            'raw': raw,
            'filtered': filtered,
        }

    def get_state(self):
        """Return current state vector and covariance."""
        return {
            'x_east':  self.x[0],
            'y_north': self.x[1],
            'z_depth': self.x[2],
            'vx':      self.x[3],
            'vy':      self.x[4],
            'vz':      self.x[5],
            'P':       self.P.copy(),
            'position_std': np.sqrt(np.diag(self.P)[:3]),
            'velocity_std': np.sqrt(np.diag(self.P)[3:]),
        }

    def get_gps_position(self, lat_B, lon_B):
        """Convert smoothed local state to GPS coordinates."""
        return _to_gps(lat_B, lon_B, self.x[0], self.x[1])

    def _build_result(self, lat_B, lon_B):
        """Build a result dict from the current filtered state."""
        lat_f, lon_f = self.get_gps_position(lat_B, lon_B)
        state = self.get_state()
        return {
            'lat_AUV':    lat_f,
            'lon_AUV':    lon_f,
            'depth_AUV':  self.x[2],
            'dx_east':    self.x[0],
            'dy_north':   self.x[1],
            'vx':         self.x[3],
            'vy':         self.x[4],
            'vz':         self.x[5],
            'position_std': state['position_std'],
            'velocity_std': state['velocity_std'],
        }


# ---------------------------------------------
# Simulation Demo
# ---------------------------------------------

def run_kalman_demo():
    """
    Simulate an AUV moving in a circle with noisy sensors.
    Show that the Kalman filter reduces position error vs raw fixes.
    """
    import random
    random.seed(123)
    np.random.seed(123)

    LAT_B = 7.208300
    LON_B = 79.835800

    # Simulation parameters
    dt = 1.0             # 1 Hz updates
    n_steps = 60         # 60 seconds
    radius = 15.0        # circular path radius (m)
    omega = 2 * math.pi / 40.0   # angular velocity (full circle in 40s)
    base_depth = 50.0
    cable_L = 100.0

    # Noise added to angles (simulates sensor noise)
    angle_noise_std = math.radians(2.0)  # 2 degrees std

    kf = AUVKalmanFilter(process_noise=0.3, measurement_noise=3.0)

    raw_errors = []
    kf_errors = []

    print("\n" + "="*60)
    print("  Kalman Filter Demo -- AUV circular path")
    print("="*60)
    print(f"\n  Simulation: {n_steps}s, dt={dt}s, radius={radius}m\n")
    print(f"  {'Step':>4}  {'Raw err(m)':>10}  {'KF err(m)':>10}  {'KF vx(m/s)':>10}  {'KF vy(m/s)':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for step in range(n_steps):
        t = step * dt

        # True AUV position (circular path)
        true_x = radius * math.sin(omega * t)
        true_y = radius * math.cos(omega * t)
        true_z = base_depth

        # True velocity
        true_vx = radius * omega * math.cos(omega * t)
        true_vy = -radius * omega * math.sin(omega * t)

        # Compute the tether angles that would produce this position
        # (reverse-engineer from the geometry, with noise added)
        # For simplicity, use small-angle approx: sin(a) ~ a, cos(a) ~ 1
        # x_auv ~ l1*sin(alpha) + l2*sin(beta), y_auv ~ l1*sin(mu) + l2*sin(eta)
        # With most cable vertical: l1 ~ 70, l2 ~ 30 (at z=50, L=100)
        alpha_true = math.atan2(true_x * 0.7, 70.0)  # rough inverse
        mu_true    = math.atan2(true_y * 0.7, 70.0)
        beta_true  = math.atan2(true_x * 0.3, 30.0)
        eta_true   = math.atan2(true_y * 0.3, 30.0)

        # Add sensor noise
        alpha_noisy = alpha_true + random.gauss(0, angle_noise_std)
        mu_noisy    = mu_true    + random.gauss(0, angle_noise_std)
        beta_noisy  = beta_true  + random.gauss(0, angle_noise_std)
        eta_noisy   = eta_true   + random.gauss(0, angle_noise_std)

        # Clamp angles to valid range
        alpha_noisy = max(0.001, min(math.radians(80), abs(alpha_noisy))) * (1 if alpha_noisy >= 0 else -1)
        mu_noisy    = max(0.001, min(math.radians(80), abs(mu_noisy)))    * (1 if mu_noisy >= 0 else -1)
        beta_noisy  = max(0.001, min(math.radians(80), abs(beta_noisy)))  * (1 if beta_noisy >= 0 else -1)
        eta_noisy   = max(0.001, min(math.radians(80), abs(eta_noisy)))   * (1 if eta_noisy >= 0 else -1)

        # Make all angles positive (model expects positive angles)
        alpha_noisy = abs(alpha_noisy)
        mu_noisy    = abs(mu_noisy)
        beta_noisy  = abs(beta_noisy)
        eta_noisy   = abs(eta_noisy)

        try:
            # Predict + update
            if step > 0:
                kf.predict(dt)
            result = kf.update(LAT_B, LON_B, alpha_noisy, mu_noisy,
                               beta_noisy, eta_noisy, cable_L, true_z)

            # Raw position error
            raw_x = result['raw']['dx_east']
            raw_y = result['raw']['dy_north']
            raw_err = math.sqrt((raw_x - true_x)**2 + (raw_y - true_y)**2)

            # KF position error
            kf_x = result['filtered']['dx_east']
            kf_y = result['filtered']['dy_north']
            kf_err = math.sqrt((kf_x - true_x)**2 + (kf_y - true_y)**2)

            raw_errors.append(raw_err)
            kf_errors.append(kf_err)

            if step % 5 == 0:
                print(f"  {step:4d}  {raw_err:10.3f}  {kf_err:10.3f}"
                      f"  {result['filtered']['vx']:10.4f}"
                      f"  {result['filtered']['vy']:10.4f}")

        except ValueError:
            # Skip invalid geometry cases from noise
            continue

    if raw_errors and kf_errors:
        avg_raw = sum(raw_errors) / len(raw_errors)
        avg_kf  = sum(kf_errors)  / len(kf_errors)
        improvement = (1.0 - avg_kf / avg_raw) * 100

        print(f"\n  {'='*50}")
        print(f"  Average raw error:      {avg_raw:.3f} m")
        print(f"  Average filtered error: {avg_kf:.3f} m")
        print(f"  Improvement:            {improvement:.1f}%")
        print(f"  {'='*50}")
    else:
        print("\n  No valid data points collected.")


# ---------------------------------------------
# Automated Tests
# ---------------------------------------------

def run_kalman_tests():
    """Automated tests for the Kalman filter."""

    print("\n" + "="*55)
    print("  Kalman Filter Tests")
    print("="*55)

    LAT_B = 7.208300
    LON_B = 79.835800

    # --------------------------------------------------
    # Test 1: Static convergence
    # Feed the same measurement repeatedly. The filter
    # should converge to that position with shrinking
    # uncertainty.
    # --------------------------------------------------
    print("\nTest 1 -- Static convergence:")
    kf = AUVKalmanFilter(process_noise=0.1, measurement_noise=2.0)

    alpha = math.radians(10)
    mu    = math.radians(8)
    beta  = math.radians(5)
    eta   = math.radians(3)
    L, z  = 100.0, 50.0

    # Get the true position for this setup
    true_pos = get_auv_position(LAT_B, LON_B, alpha, mu, beta, eta, L, z)

    for i in range(30):
        if i > 0:
            kf.predict(1.0)
        kf.update(LAT_B, LON_B, alpha, mu, beta, eta, L, z)

    state = kf.get_state()
    pos_err = math.sqrt(
        (state['x_east']  - true_pos['dx_east'])**2 +
        (state['y_north'] - true_pos['dy_north'])**2 +
        (state['z_depth'] - true_pos['depth_AUV'])**2
    )
    vel_mag = math.sqrt(state['vx']**2 + state['vy']**2 + state['vz']**2)
    pos_std = max(state['position_std'])

    print(f"  Position error after 30 steps: {pos_err:.6f} m")
    print(f"  Velocity magnitude: {vel_mag:.6f} m/s (should be ~0)")
    print(f"  Max position std: {pos_std:.4f} m")
    assert pos_err < 0.5, f"Position error {pos_err:.3f}m too large"
    assert vel_mag < 0.1, f"Velocity {vel_mag:.3f}m/s should be near zero"
    assert pos_std < 2.0, f"Uncertainty {pos_std:.3f}m should have shrunk"
    print("  OK PASSED")

    # --------------------------------------------------
    # Test 2: Noisy measurements -- KF should smooth
    # Add noise to angles and verify KF error < raw error
    # over many steps.
    # --------------------------------------------------
    print("\nTest 2 -- Noise smoothing (KF error < raw error):")
    np.random.seed(42)
    kf2 = AUVKalmanFilter(process_noise=0.1, measurement_noise=3.0)

    raw_errors = []
    kf_errors = []

    for i in range(50):
        # Add noise to angles
        a_noisy = alpha + np.random.normal(0, math.radians(1.5))
        m_noisy = mu    + np.random.normal(0, math.radians(1.5))
        b_noisy = beta  + np.random.normal(0, math.radians(1.5))
        e_noisy = eta   + np.random.normal(0, math.radians(1.5))

        # Ensure positive angles
        a_noisy = abs(a_noisy)
        m_noisy = abs(m_noisy)
        b_noisy = abs(b_noisy)
        e_noisy = abs(e_noisy)

        try:
            if i > 0:
                kf2.predict(1.0)
            result = kf2.update(LAT_B, LON_B, a_noisy, m_noisy,
                                b_noisy, e_noisy, L, z)

            raw_err = math.sqrt(
                (result['raw']['dx_east']  - true_pos['dx_east'])**2 +
                (result['raw']['dy_north'] - true_pos['dy_north'])**2
            )
            kf_err = math.sqrt(
                (result['filtered']['dx_east']  - true_pos['dx_east'])**2 +
                (result['filtered']['dy_north'] - true_pos['dy_north'])**2
            )
            raw_errors.append(raw_err)
            kf_errors.append(kf_err)
        except ValueError:
            continue

    avg_raw = sum(raw_errors) / len(raw_errors)
    avg_kf  = sum(kf_errors) / len(kf_errors)
    print(f"  Avg raw error:  {avg_raw:.3f} m")
    print(f"  Avg KF error:   {avg_kf:.3f} m")
    assert avg_kf < avg_raw, f"KF error {avg_kf:.3f} >= raw {avg_raw:.3f}"
    print("  OK PASSED")

    # --------------------------------------------------
    # Test 3: Covariance shrinks over time
    # --------------------------------------------------
    print("\nTest 3 -- Covariance reduction:")
    kf3 = AUVKalmanFilter(process_noise=0.1, measurement_noise=2.0)

    # Initial covariance (before any update)
    initial_trace = np.trace(kf3.P)

    for i in range(20):
        if i > 0:
            kf3.predict(1.0)
        kf3.update(LAT_B, LON_B, alpha, mu, beta, eta, L, z)

    final_trace = np.trace(kf3.P)
    print(f"  Initial covariance trace: {initial_trace:.2f}")
    print(f"  Final covariance trace:   {final_trace:.4f}")
    assert final_trace < initial_trace * 0.1, \
        f"Covariance didn't shrink enough: {final_trace:.2f} vs {initial_trace:.2f}"
    print("  OK PASSED")

    # --------------------------------------------------
    # Test 4: GPS output consistency
    # --------------------------------------------------
    print("\nTest 4 -- GPS output consistency:")
    kf4 = AUVKalmanFilter(process_noise=0.1, measurement_noise=2.0)
    for i in range(10):
        if i > 0:
            kf4.predict(1.0)
        kf4.update(LAT_B, LON_B, alpha, mu, beta, eta, L, z)

    lat_kf, lon_kf = kf4.get_gps_position(LAT_B, LON_B)
    state = kf4.get_state()

    # Should be close to the raw position
    assert abs(lat_kf - true_pos['lat_AUV']) < 1e-5, "Latitude mismatch"
    assert abs(lon_kf - true_pos['lon_AUV']) < 1e-5, "Longitude mismatch"
    print(f"  KF GPS:  {lat_kf:.8f} N  {lon_kf:.8f} E")
    print(f"  Raw GPS: {true_pos['lat_AUV']:.8f} N  {true_pos['lon_AUV']:.8f} E")
    print("  OK PASSED")

    print(f"\n{'='*55}")
    print("  All Kalman filter tests passed!")
    print("="*55)


# ---------------------------------------------
# Main
# ---------------------------------------------

if __name__ == "__main__":
    run_kalman_tests()
    run_kalman_demo()
