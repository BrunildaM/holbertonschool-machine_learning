#!/usr/bin/env python3
""" defines a function that calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """ function that computer the integral """
    if not isinstance(C, int) or not isinstance(poly, list) or len(poly) == 0:
        return None
    integral = [C]
    for power, coeff in enumerate(poly):
        if (coeff % (power + 1)) == 0:
            new_coeff = coeff // (power + 1)
        else:
            new_coeff = coeff / (power + 1)
        integral.append(new_coeff)
    while integral[-1] == 0 and len(integral) > 1:
        integral = integral[:-1]
    return integral
