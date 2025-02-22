# src/constants.py

import math

HBAR = 1.0545718e-34
M_E  = 9.10938356e-31
C    = 299792458

T_C_E = HBAR / (M_E * C**2)     # Electron Compton time
E0_J  = M_E * C**2             # Electron rest energy (Joules)
JOULE_TO_KEV = 1.0 / 1.602176634e-16

def time_to_tau(t):
    return t / T_C_E

def tau_to_time(tau):
    return tau * T_C_E

def energy_to_dimensionless(e_joules):
    return e_joules / E0_J

def energy_from_dimensionless(e_dimless):
    return e_dimless * E0_J

def energy_dimless_to_keV(e_dimless):
    return energy_from_dimensionless(e_dimless) * JOULE_TO_KEV
