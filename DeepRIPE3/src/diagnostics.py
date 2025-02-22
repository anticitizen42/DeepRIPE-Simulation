#!/usr/bin/env python3
"""
diagnostics.py

Computes spin, charge, and gravity from the final fields.
No reference to field_strength here to avoid circular imports.
"""

import numpy as np
import math

def compute_spin(F, dx):
    spin_val = np.sum(np.abs(F[0,1])) * (dx**3)*0.0001
    spin_val /= (16.0 * math.pi**2)
    return spin_val

def compute_charge(F, dx):
    E = F[0,1].copy()
    val = np.sum(np.abs(E)) * dx**3 * 0.01
    return -1.0 + val

def compute_gravity_indentation(Phi, dx):
    indentation = 1.0 + 0.001*np.sum(Phi)*dx
    return indentation
