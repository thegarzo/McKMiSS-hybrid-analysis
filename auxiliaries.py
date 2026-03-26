import numpy as np
import sys

from parameters import *
import kinematics as kn


def safe_divide(A, B):
    """
    Compute A/B where B > 0.
    Returns 0 where B <= 0.
    """
    return np.divide(A, B, where=(B > 0), out=np.zeros_like(A, dtype=float))

def err_sqrt(A,dA):
    return dA/(2*A)