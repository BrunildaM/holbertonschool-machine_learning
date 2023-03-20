#!/usr/bin/env python3
"""defines Exponential class that represents a exponential distribution"""


class Exponential:
    """a class that represents a exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(len(data)) / sum(data)

    def pdf(self, x):
        """calculates the value of the PDF for a given time period"""
        if (x < 0):
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        pdf = lambtha * (e ** (-lambtha * x))
        return pdf

    def cdf(self, x):
        """calculates the value of the CDF for a given time period"""
        if (x < 0):
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        cdf = 1 - e ** (-lambtha * x)
        return cdf
