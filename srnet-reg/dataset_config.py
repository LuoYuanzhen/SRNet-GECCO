"""Real function, map of experiments' cgpnet_result, range of interpolation data, extrapolation data"""
import torch


PI = torch.acos(torch.zeros(1)).item() * 2


# kkk
def kkk0_koza_sin(x):
    # [-1, 1] 200
    return torch.sin(x) + torch.sin(x + x**2)


def kkk1_koza_sincos(x0, x1):
    # x0, x1 [-1, 1] 200
    return 2 * torch.sin(x0) * torch.cos(x1)


def kkk2_korns_ln(x):
    # [-50, 50] 200
    return 3 + 2.13 * torch.log(torch.abs(x))


def kkk3_korns_exponent(x0, x1):
    # x0, x1 [-5.5, 0.4] 25
    return 1 / (1 + x0 ** (-4)) + 1 / (1 + x1 ** (-4))


def kkk4_keijzer_div(x0, x1, x2):
    # x0, x2:U[-1,1] x1:U[1,2] 1000
    return 30 * x0 * x1 / ((x0-10) * x2 * x2)


def kkk5_keijzer_sin(x0, x1):
    # x0, x1 [-3, 3] 20
    return x0 * x1 + torch.sin((x0 - 1) * (x1 - 1))


# feynman
def mass_velocity_effect_formula(m0, v, c):
    return m0 / torch.sqrt(1 - v**2 / c**2)


def coulombs_law_formula(q1, q2, epsilon, r):
    return q1 * q2 / (4 * PI * epsilon * r**2)


def gravitational_potential_energy(m1, m2, r1, r2, G):
    return G * m1 * m2 * (1/r2 - 1/r1)


def elastic_potential_formula(k, x):
    return 0.5 * k * x**2


def gravitational_wave_power_formula(G, c, m1, m2, r):
    return -32/5 * G**4/c**5 * ((m1*m2)**2 * (m1+m2) / r**5)


def jackson2_11(q, y, Volt, d, ep):
    temp = 4 * PI * ep
    return q / (temp * y ** 2) * (temp * Volt * d - q * d * y ** 3 / (y ** 2 - d ** 2) ** 2)


# the dataset that could draw as curve
CURVES_DATASET = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk5']

# the dataset that should draw with log(output)
LOG_RESULT_DATASET = ['kkk4', 'feynman1', 'feynman2', 'feynman4', 'feynman5']

FUNC_MAP = {
    'kkk0': kkk0_koza_sin,
    'kkk1': kkk1_koza_sincos,
    'kkk2': kkk2_korns_ln,
    'kkk3': kkk3_korns_exponent,
    'kkk4': kkk4_keijzer_div,
    'kkk5': kkk5_keijzer_sin,

    'feynman0': mass_velocity_effect_formula,
    'feynman1': coulombs_law_formula,
    'feynman2': gravitational_potential_energy,
    'feynman3': elastic_potential_formula,
    'feynman4': gravitational_wave_power_formula,
    'feynman5': jackson2_11
}

vars_map = {
    'feynman0': (['m0', 'v', 'c'], 'm'),
    'feynman1': (['q1', 'q2', 'e', 'r'], 'F'),
    'feynman2': (['G', 'm1', 'm2', 'r1', 'r2'], 'U'),
    'feynman3': (['k', 'x'], 'U'),
    'feynman4': (['G', 'c', 'm1', 'm2', 'r'], 'P'),
    'feynman5': (['q', 'y', 'V', 'd', 'e'], 'F')
}

INTER_MAP = {
    'kkk0': [(-1, 1)],
    'kkk1': [(-1, 1), (-1, 1)],
    'kkk2': [(-50, 50)],
    'kkk3': [(-5, 5), (-5, 5)],
    'kkk4': [(-1, 1), (-1, 1), (1, 2)],
    'kkk5': [(-3, 3), (-3, 3)],

    'feynman0': [(1, 5), (1, 2), (3, 10)],
    'feynman1': [(1, 5), (1, 5), (1, 5), (1, 5)],
    'feynman2': [(1, 5), (1, 5), (1, 5), (1, 5), (1, 5)],
    'feynman3': [(1, 5), (1, 5)],
    'feynman4': [(1, 2), (1, 2), (1, 5), (1, 5), (1, 2)],
    'feynman5': [(1, 5), (1, 3), (1, 5), (4, 6), (1, 5)]
}

VALID_MAP = {
    'kkk0': [(-2, 2)],
    'kkk1': [(-2, 2), (-2, 2)],
    'kkk2': [(-100, 100)],
    'kkk3': [(-10, 10), (-10, 10)],
    'kkk4': [(-2, 2), (-2, 2), (0.5, 2.5)],
    'kkk5': [(-6, 6), (-6, 6)],

    'feynman0': [(0.5, 8.5), (0.5, 2.5), (0.5, 14.5)],
    'feynman1': [(0.5, 8.5), (0.5, 8.5), (0.5, 8.5), (0.5, 8.5)],
    'feynman2': [(0.5, 8.5), (0.5, 8.5), (0.5, 8.5), (0.5, 8.5), (0.5, 8.5)],
    'feynman3': [(0.5, 8.5), (0.5, 8.5)],
    'feynman4': [(0.5, 2.5), (0.5, 2.5), (0.5, 8.5), (0.5, 8.5), (0.5, 2.5)],
    'feynman5': [(0.5, 8.5), (0.5, 4.5), (0.5, 8.5), (2, 8), (0.5, 8.5)]
}

TEST_MAP = {
    'kkk0': [(-5, 5)],
    'kkk1': [(-5, 5), (-5, 5)],
    'kkk2': [(-250, 250)],
    'kkk3': [(-25, 25), (-25, 25)],
    'kkk4': [(-5, 5), (-5, 5), (0.5, 5.5)],
    'kkk5': [(-15, 15), (-15, 15)],

    'feynman0': [(0.5, 20.5), (0.5, 5.5), (0.5, 35.5)],
    'feynman1': [(0.5, 20.5), (0.5, 20.5), (0.5, 20.5), (0.5, 20.5)],
    'feynman2': [(0.5, 20.5), (0.5, 20.5), (0.5, 20.5), (0.5, 20.5), (0.5, 20.5)],
    'feynman3': [(0.5, 20.5), (0.5, 20.5)],
    'feynman4': [(0.5, 5.5), (0.5, 5.5), (0.5, 20.5), (0.5, 20.5), (0.5, 5.5)],
    'feynman5': [(0.5, 20.5), (0.5, 10.5), (0.5, 20.5), (0.5, 10.5), (0.5, 20.5)]
}
