import math


def cor(alpha, beta, mu, eta, L, z):
    a1_sq = 1 + math.tan(mu) ** 2 * math.cos(alpha) ** 2
    a2_sq = 1 + math.tan(eta) ** 2 * math.cos(beta) ** 2

    a1 = math.sqrt(a1_sq)
    a2 = math.sqrt(a2_sq)

    num_l2 = (L * math.cos(alpha) / a1) - z
    den_l2 = (math.cos(alpha) / a1) + (math.cos(beta) / a2)

    if abs(den_l2) < 1e-9:
        den_l2 = 1e-9

    l2 = num_l2 / den_l2
    l1 = L - l2

    l1x_sq = max((l1 ** 2) / a1_sq, 0.0)
    l1y_sq = max((l1 ** 2) / (math.sin(mu) ** 2 + (math.cos(mu) / math.cos(alpha)) ** 2), 0.0)

    l2x_sq = max((l2 ** 2) / a2_sq, 0.0)
    l2y_sq = max((l2 ** 2) / (math.sin(eta) ** 2 + (math.cos(eta) / math.cos(beta)) ** 2), 0.0)

    return {
        "l1": l1,
        "l2": l2,
        "l1x_sq": l1x_sq,
        "l1y_sq": l1y_sq,
        "l2x_sq": l2x_sq,
        "l2y_sq": l2y_sq
    }


def calculate_coordinates(alpha, beta, mu, eta, L, z_input, sb=1, l0=0):
    results = cor(alpha, beta, mu, eta, L, z_input)

    l1x = math.sqrt(results["l1x_sq"])
    l1y = math.sqrt(results["l1y_sq"])
    l2x = math.sqrt(results["l2x_sq"])
    l2y = math.sqrt(results["l2y_sq"])

    x = l1x * math.sin(alpha) + l2x * math.sin(beta)
    y = l1y * math.sin(mu) + l2y * math.sin(eta)

    z_check_1 = l0 + sb * l1x * math.cos(alpha) - sb * l2x * math.cos(beta)
    z_check_2 = l0 + sb * l1y * math.cos(mu) - sb * l2y * math.cos(eta)

    z_est = -(z_input)

    return {
        "x": x,
        "y": y,
        "z": z_est,
        "z_check_1": z_check_1,
        "z_check_2": z_check_2,
    }