"""Finite-domain check of the dispersion-corrected mean tracer velocity (Appendix C).

The closed-form expression (Eq. 27) keeps only the decaying spatial mode of the
moment equations. Here, the moment boundary-value problem is solved on the
finite segment with Danckwerts inlet and zero-gradient outlet conditions, once
analytically and once with scipy's collocation solver, and the velocity fitted
at the observation positions is compared with Eq. 27.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp
from scipy.stats import linregress

V_CONV = 5e-5               # m/s
LENGTH = 0.2                # m
POSITIONS = np.array([0.03, 0.06, 0.09])  # m

# name, axial dispersion (m^2/s), e12, e21, e23 (1/s)
CASES = [
    ('exchange_low', 1e-12, 1e-3, 1e-3, 1e-2),
    ('exchange_high', 1e-12, 1e-2, 1e-2, 1e-6),
    ('dispersion_low', 1e-7, 0, 0, 0),
    ('dispersion_high', 1e-6, 0, 0, 0),
    ('both_low', 1e-7, 1e-4, 1e-4, 1e-3),
    ('both_high', 1e-6, 1e-2, 1e-2, 1e-3),
]


def exchange_factors(e12, e21, e23):
    """Effective loss rate k and retardation factor h = 1 + e12 e21 / (e21 + e23)^2."""
    if e12 == 0:
        return 0.0, 1.0
    return e12 * e23 / (e21 + e23), 1 + e12 * e21 / (e21 + e23) ** 2


def zeroth_moment(x, dispersion, k):
    """Zeroth moment of the mobile tracer for a unit inlet pulse, both spatial modes."""
    s = np.sqrt(V_CONV ** 2 + 4 * dispersion * k)
    r_minus = -2 * k / (V_CONV + s)
    r_plus = (V_CONV + s) / (2 * dispersion)
    q = -r_minus / r_plus  # weight of the growing mode from the outlet condition
    shape = np.exp(r_minus * x) + q * np.exp(r_minus * LENGTH + r_plus * (x - LENGTH))
    shape_0 = 1 + q * np.exp((r_minus - r_plus) * LENGTH)
    slope_0 = r_minus + q * r_plus * np.exp((r_minus - r_plus) * LENGTH)
    return V_CONV * shape / (V_CONV * shape_0 - dispersion * slope_0)


def moments(x, dispersion, k, h):
    """Zeroth and first moment; M1 = -h dM0/dk, evaluated by complex-step differentiation."""
    eps = 1e-20
    m0 = zeroth_moment(x, dispersion, k)
    m1 = -h * np.imag(zeroth_moment(x, dispersion, k + 1j * eps)) / eps
    return m0, m1


def collocation_mean_times(dispersion, k, h):
    """Mean arrival times from scipy's BVP solver, in scaled coordinates z = x / L."""
    peclet = V_CONV * LENGTH / dispersion
    loss = k * LENGTH ** 2 / dispersion
    z = np.linspace(0, 1, 401)
    m0, m1 = moments(z * LENGTH, dispersion, k, h)
    m1 = m1 * V_CONV / LENGTH
    guess = np.vstack([m0, np.gradient(m0, z), m1, np.gradient(m1, z)])

    def rhs(z, y):
        return np.vstack([y[1], peclet * y[1] + loss * y[0],
                          y[3], peclet * y[3] + loss * y[2] - h * peclet * y[0]])

    def boundary(ya, yb):
        return np.array([ya[0] - ya[1] / peclet - 1, yb[1], ya[2] - ya[3] / peclet, yb[3]])

    solution = solve_bvp(rhs, boundary, z, guess, tol=1e-9, max_nodes=20000)
    if not solution.success:
        raise RuntimeError(solution.message)
    values = solution.sol(POSITIONS / LENGTH)
    return values[2] / values[0] * LENGTH / V_CONV


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, dispersion, e12, e21, e23 in CASES:
        k, h = exchange_factors(e12, e21, e23)
        closed_form = np.sqrt(V_CONV ** 2 + 4 * dispersion * k) / h * 60000
        m0, m1 = moments(POSITIONS, dispersion, k, h)
        finite_domain = linregress(m1 / m0, POSITIONS).slope * 60000
        row = dict(case=name, closed_form_mm_min=closed_form, finite_domain_mm_min=finite_domain,
                   difference_mm_min=finite_domain - closed_form)
        if dispersion >= 1e-7:
            collocation = linregress(collocation_mean_times(dispersion, k, h), POSITIONS).slope * 60000
            row['collocation_mm_min'] = collocation
            assert abs(collocation - finite_domain) < 1e-6
        results.append(row)
        print(json.dumps(row))

    (output_dir / 'finite_domain_validation.json').write_text(json.dumps(results, indent=2))
    print(f"maximum |difference|: {max(abs(r['difference_mm_min']) for r in results):.5f} mm/min")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    main(parser.parse_args().output_dir)
