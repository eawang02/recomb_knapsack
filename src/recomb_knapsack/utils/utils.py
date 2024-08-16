
import scipy.stats as stats
import numpy as np
from tqdm import tqdm

def generate_epsilon(ep=0.01, size=1000):
    """
    Samples from a normal distribution with mean 1 and standard deviation ep.

    Args:
        - ep: float, standard deviation of the normal distribution (default is 0.01)
        - size: int, number of samples to generate (default is 1000)

    Returns:
        - ndarray of floats, samples from the normal distribution
    """
    return stats.norm.rvs(loc=1, scale=ep, size=size)


def generate_gamma(size=1000):
    """
    Samples from a gamma distribution with shape parameter k=2.34, scale parameter theta=814.8, and shifted by -200.21.
    Parameters are tuned to fit the national precinct populations from the 2020 US Census.

    Args:
        - size: int, number of samples to generate (default is 1000)

    Returns:
        - ndarray of floats, samples from the gamma distribution
    """
    return stats.gamma.rvs(a=2.34, loc=-200.21, scale=814.80, size=size)


def generate_gamma_clean(size=1000):
    """
    Samples from a gamma distribution with shape parameter k=2.34, scale parameter theta=814.8, and shifted by -200.21.
    Parameters are tuned to fit the national precinct populations from the 2020 US Census.
    These samples are guaranteed to be at least 0 and are rounded to the nearest integer.

    Args:
        - size: int, number of samples to generate (default is 1000)

    Returns:
        - ndarray of floats, samples from the gamma distribution
    """
    return np.round(np.maximum(0, stats.gamma.rvs(a=2.34, loc=-200.21, scale=814.80, size=size)))


def validate(partitions, C):
    """
    Validates that the partitions are valid given the constraints C.

    Args:
        - partitions: list of lists, representing a partition of the input data
        - C: int tuple of length 2, constraints on the partitions

    Returns:
        - bool, whether all partitions fall within the range denoted by C
    """
    return all([C[0] <= sum(subset) <= C[1] for subset in partitions])