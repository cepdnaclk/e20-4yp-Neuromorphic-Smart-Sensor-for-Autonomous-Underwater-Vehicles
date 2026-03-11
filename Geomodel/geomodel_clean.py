"""
GeoModel — AUV Position from RTK Buoy + Tether Sensors
=======================================================

Based on: Viel et al., "ROV localization based on umbilical angle measurement"

With these substituted, the equations simplify to:

  l2 = ( L·cos(α)/a1 − z ) / ( cos(α)/a1 + cos(β)/a2 )
  l1 = L − l2
  z_B = l1x · cos(α)         ← ballast depth (always > AUV depth)

INPUTS
------
  lat_B, lon_B  — buoy RTK-CORS coordinates  (decimal degrees, WGS84)
  alpha (α)     — tether angle at buoy, East-West    (radians from vertical)
  mu    (μ)     — tether angle at buoy, North-South  (radians from vertical)
  beta  (β)     — tether angle at AUV,  East-West    (radians from vertical)
  eta   (η)     — tether angle at AUV,  North-South  (radians from vertical)
  L             — total cable length paid out  (metres, from encoder)
  z             — AUV depth  (metres, positive downward, from pressure sensor)

OUTPUTS
-------
  lat_AUV, lon_AUV   — AUV GPS coordinates  (decimal degrees, WGS84)
  depth_AUV          — AUV depth  (metres)
  lat_B_s, lon_B_s   — Ballast GPS coordinates
  depth_B            — Ballast depth  (metres, always > depth_AUV)
  dx_east, dy_north  — AUV offset from buoy  (metres)
  l1, l2             — tether segment lengths  (metres)
  z_check            — depth recomputed from geometry  (should equal z)
  z_error            — |z − z_check|  in metres
"""

import math

R_EARTH = 6_378_137.0   # WGS84 mean earth radius (metres)


# ---------------------------------------------
# STEP 1 -- Scale factors 
# ---------------------------------------------

def _scale_factors(alpha, mu, beta, eta):
    """
    a1 = sqrt(1 + tan²(μ)·cos²(α))
    a2 = sqrt(1 + tan²(η)·cos²(β))

    Correct for each segment tilting in two planes simultaneously.
    """
    a1 = math.sqrt(1.0 + math.tan(mu)**2  * math.cos(alpha)**2)
    a2 = math.sqrt(1.0 + math.tan(eta)**2 * math.cos(beta)**2)
    return a1, a2


# ---------------------------------------------
# STEP 2 -- Solve for l1 and l2  
# ---------------------------------------------

def _solve_l1_l2(alpha, beta, a1, a2, L, z):
    """
    With sb=1 and l0=0 substituted into paper eq (13):

      l2 = ( L·cos(α)/a1 − z ) / ( cos(α)/a1 + cos(β)/a2 )
      l1 = L − l2
    """
    ca_a1 = math.cos(alpha) / a1
    cb_a2 = math.cos(beta)  / a2

    numerator   = L * ca_a1 - z
    denominator = ca_a1 + cb_a2

    if abs(denominator) < 1e-10:
        raise ValueError(
            f"Degenerate tether geometry — denominator is zero. "
            f"Check angles: α={math.degrees(alpha):.1f}°, β={math.degrees(beta):.1f}°"
        )

    l2 = numerator / denominator
    l1 = L - l2

    if l1 < 0 or l2 < 0:
        raise ValueError(
            f"Impossible geometry: l1={l1:.3f}m, l2={l2:.3f}m. "
            f"Depth z={z:.2f}m is not reachable with cable L={L:.2f}m "
            f"at angles α={math.degrees(alpha):.1f}°, β={math.degrees(beta):.1f}°."
        )

    return l1, l2


# STEP 3 — Segment projections 

def _projections(l1, l2, alpha, mu, beta, eta, a1, a2):
    """
    East components:
      l1x = l1 / a1
      l2x = l2 / a2

    North components (paper eq 9, 11):
      l1y = l1 / sqrt( sin²(μ) + (cos(μ)/cos(α))² )
      l2y = l2 / sqrt( sin²(η) + (cos(η)/cos(β))² )
    """
    def _north(l, mu_ang, alpha_ang):
        cos_a = math.cos(alpha_ang)
        if abs(cos_a) < 1e-10:
            return 0.0
        return l / math.sqrt(math.sin(mu_ang)**2 + (math.cos(mu_ang) / cos_a)**2)

    l1x = l1 / a1
    l2x = l2 / a2
    l1y = _north(l1, mu,  alpha)
    l2y = _north(l2, eta, beta)

    return l1x, l1y, l2x, l2y


# STEP 4 — AUV and ballast local positions

def _local_positions(alpha, mu, beta, eta, l1x, l1y, l2x, l2y):
    """
    AUV offset from buoy (paper eq 3):
      x = l1x·sin(α) + l2x·sin(β)
      y = l1y·sin(μ) + l2y·sin(η)

    Depth verification (paper eq 4, sb=1, l0=0):
      z_check = l1x·cos(α) − l2x·cos(β)

    Ballast depth (paper Appendix A.1 eq 35, sb=1, l0=0):
      z_B = l1x·cos(α)

    Ballast offset from buoy:
      x_B = l1x·sin(α)
      y_B = l1y·sin(μ)
    """
    x_auv   = l1x * math.sin(alpha) + l2x * math.sin(beta)
    y_auv   = l1y * math.sin(mu)    + l2y * math.sin(eta)
    z_check = l1x * math.cos(alpha) - l2x * math.cos(beta)

    z_B = l1x * math.cos(alpha)
    x_B = l1x * math.sin(alpha)
    y_B = l1y * math.sin(mu)

    return x_auv, y_auv, z_check, x_B, y_B, z_B


# STEP 5 — Local offset GPS coordinates

def _to_gps(lat_B, lon_B, dx_east, dy_north):
    """
    Flat-earth conversion, valid for offsets < ~10 km.

      lat = lat_B + dy_north / R_earth
      lon = lon_B + dx_east  / (R_earth · cos(lat_B))
    """
    lat_rad = math.radians(lat_B)
    lat_out = lat_B + math.degrees(dy_north / R_EARTH)
    lon_out = lon_B + math.degrees(dx_east  / (R_EARTH * math.cos(lat_rad)))
    return lat_out, lon_out


# MAIN FUNCTION

def get_auv_position(lat_B, lon_B, alpha, mu, beta, eta, L, z):
    """
    Compute AUV GPS position from RTK buoy coordinates and sensor values.

    Parameters
    ----------
    lat_B  : float  Buoy latitude           (decimal degrees, RTK-CORS)
    lon_B  : float  Buoy longitude          (decimal degrees, RTK-CORS)
    alpha  : float  Tether angle at buoy, East-West    (radians)
    mu     : float  Tether angle at buoy, North-South  (radians)
    beta   : float  Tether angle at AUV,  East-West    (radians)
    eta    : float  Tether angle at AUV,  North-South  (radians)
    L      : float  Cable length paid out  (metres, encoder)
    z      : float  AUV depth             (metres, pressure sensor)

    Returns
    -------
    dict
        lat_AUV, lon_AUV  : AUV GPS position  (decimal degrees, WGS84)
        depth_AUV         : AUV depth  (metres)
        dx_east           : AUV east  offset from buoy  (metres)
        dy_north          : AUV north offset from buoy  (metres)
        lat_B_s, lon_B_s  : Ballast GPS position  (decimal degrees)
        depth_B           : Ballast depth  (metres, always > depth_AUV)
        l1                : buoy → ballast segment  (metres)
        l2                : ballast → AUV segment   (metres)
        z_check           : depth recomputed from geometry  (metres)
        z_error           : |z − z_check|  (metres)
    """
    if L <= 0:
        raise ValueError(f"Cable length L must be positive, got {L}")
    if z < 0:
        raise ValueError(f"Depth z must be non-negative, got {z}")
    if z >= L:
        raise ValueError(
            f"Depth z={z:.2f}m must be less than cable length L={L:.2f}m"
        )

    a1, a2                          = _scale_factors(alpha, mu, beta, eta)
    l1, l2                          = _solve_l1_l2(alpha, beta, a1, a2, L, z)
    l1x, l1y, l2x, l2y             = _projections(l1, l2, alpha, mu, beta, eta, a1, a2)
    x_auv, y_auv, z_check, \
    x_B, y_B, z_B                  = _local_positions(alpha, mu, beta, eta, l1x, l1y, l2x, l2y)
    lat_AUV, lon_AUV                = _to_gps(lat_B, lon_B, x_auv, y_auv)
    lat_B_s, lon_B_s                = _to_gps(lat_B, lon_B, x_B,   y_B)

    return {
        "lat_AUV":  lat_AUV,
        "lon_AUV":  lon_AUV,
        "depth_AUV": z,
        "dx_east":  x_auv,
        "dy_north": y_auv,
        "lat_B_s":  lat_B_s,
        "lon_B_s":  lon_B_s,
        "depth_B":  z_B,
        "l1":       l1,
        "l2":       l2,
        "z_check":  z_check,
        "z_error":  abs(z - z_check),
    }


# ---------------------------------------------
# TESTS
# ---------------------------------------------

def run_tests():
    import math, random

    LAT_B = 7.208300
    LON_B = 79.835800

    print("\n" + "="*55)
    print("  GeoModel Tests")
    print("="*55)

    # Test 1 — zero angles, straight down
    r = get_auv_position(LAT_B, LON_B, 0, 0, 0, 0, 100, 40)
    print(f"\nTest 1 -- Zero angles:")
    print(f"  l1={r['l1']:.4f}m   l2={r['l2']:.4f}m")
    print(f"  Ballast depth={r['depth_B']:.4f}m   AUV depth={r['depth_AUV']:.4f}m")
    assert math.isclose(r['l1'], 70.0, rel_tol=1e-9)
    assert math.isclose(r['l2'], 30.0, rel_tol=1e-9)
    assert math.isclose(r['depth_B'], 70.0, rel_tol=1e-9)
    assert r['depth_B'] > r['depth_AUV']
    assert math.isclose(r['lat_AUV'], LAT_B, abs_tol=1e-9)  # directly below buoy
    assert math.isclose(r['lon_AUV'], LON_B, abs_tol=1e-9)
    print("  OK PASSED")

    # Test 2 — small angles (original notebook Test 2)
    r = get_auv_position(LAT_B, LON_B,
                         math.radians(10), math.radians(8),
                         math.radians(5),  math.radians(3),
                         100, 50)
    print(f"\nTest 2 -- Small angles (original notebook values):")
    print(f"  l1={r['l1']:.4f}m   l2={r['l2']:.4f}m")
    print(f"  AUV:  {r['lat_AUV']:.8f}N  {r['lon_AUV']:.8f}E  depth={r['depth_AUV']:.3f}m")
    print(f"  Ballast: depth={r['depth_B']:.4f}m")
    print(f"  z_error={r['z_error']:.2e}m")
    assert math.isclose(r['l1'], 75.8668, rel_tol=1e-4)
    assert math.isclose(r['l2'], 24.1332, rel_tol=1e-4)
    assert r['z_error'] < 1e-9
    assert r['depth_B'] > r['depth_AUV']
    print("  OK PASSED")

    # Test 3 — large angles (original notebook Test 4)
    r = get_auv_position(LAT_B, LON_B,
                         math.radians(25), math.radians(10),
                         math.radians(15), math.radians(5),
                         120, 60)
    print(f"\nTest 3 -- Large angles (original notebook values):")
    print(f"  l1={r['l1']:.4f}m   l2={r['l2']:.4f}m")
    assert math.isclose(r['l1'], 94.4842, rel_tol=1e-4)
    assert math.isclose(r['l2'], 25.5158, rel_tol=1e-4)
    assert r['depth_B'] > r['depth_AUV']
    print("  OK PASSED")

    # Test 4 — ballast always deeper (100 random cases)
    print(f"\nTest 4 -- Ballast always deeper than AUV (100 random cases):")
    random.seed(42)
    for i in range(100):
        a  = math.radians(random.uniform(0, 40))
        m  = math.radians(random.uniform(0, 40))
        b  = math.radians(random.uniform(0, 40))
        e  = math.radians(random.uniform(0, 40))
        Lv = random.uniform(10, 100)
        # z must be physically reachable: with tilted cable, max depth < L.
        # Use 0.80*L as a safe upper bound for random testing.
        zv = random.uniform(0.1, Lv * 0.80)
        r  = get_auv_position(LAT_B, LON_B, a, m, b, e, Lv, zv)
        assert r['depth_B'] > r['depth_AUV'], \
            f"Case {i}: depth_B={r['depth_B']:.3f} <= depth_AUV={r['depth_AUV']:.3f}"
    print("  OK PASSED -- ballast always deeper in all 100 cases")

    print(f"\n{'='*55}")
    print("  All tests passed!")
    print("="*55)


# ---------------------------------------------
# EXAMPLE
# ---------------------------------------------

if __name__ == "__main__":
    run_tests()

    import math
    print("\n\n-- Example: live position fix --\n")

    r = get_auv_position(
        lat_B  = 7.208300,
        lon_B  = 79.835800,
        alpha  = math.radians(10),
        mu     = math.radians(8),
        beta   = math.radians(5),
        eta    = math.radians(3),
        L      = 100.0,
        z      = 50.0,
    )

    print(f"BUOY    :  7.20830000 N   79.83580000 E   (surface)")
    print()
    print(f"BALLAST :  {r['lat_B_s']:.8f} N   {r['lon_B_s']:.8f} E"
          f"   depth = {r['depth_B']:.3f}m   l1 = {r['l1']:.4f}m")
    print()
    print(f"AUV     :  {r['lat_AUV']:.8f} N   {r['lon_AUV']:.8f} E"
          f"   depth = {r['depth_AUV']:.3f}m   l2 = {r['l2']:.4f}m")
    print()
    print(f"z_check :  {r['z_check']:.8f}m   (error = {r['z_error']:.2e}m)")
