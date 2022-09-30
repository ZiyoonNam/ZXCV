from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy
import copy
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import detrend
from scipy import fftpack
import scipy.stats
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings(action="ignore")


def cov_xr(da1, da2, dims="time"):
     """Calculate covariance matrix between da1 and da2.  

    :param da1: 1-D (time) DataArray
    :type da1: :class:`xarray.DataArray`
    :param da2: 3-D (time, lat, lon) DataArray
    :type da2: :class:`xarray.DataArray`
    :param dims: The axis that you want to calculate, defaults to "time"
    :type dims: :class:`str`
    :return: 2-D (lat, lon) DataArray, covariance matrix.
    :rtype: :class:`xarray.DataArray`
    """
   
    return xr.dot(da1-da1.mean(dims), da2-da2.mean(dims), dims=dims) \
        / (da1.count(dims) - 1)

def cor_xr(da1, da2, dims="time"):
    """Calculate correlation coefficient matrix between da1 and da2

    :param da1: 1-D (time) DataArray
    :type da1: :class:`xarray.DataArray`
    :param da2: 3-D (time, lat, lon) DataArray
    :type da2: :class:`xarray.DataArray`
    :param dims: The axis that you want to calculate, defaults to "time"
    :type dims: :class:`str`
    :return: 2-D (lat, lon) DataArray, correlation coefficient matrix.
    :rtype: :class:`xarray.DataArray`
    """
    return cov_xr(da1, da2, dims) \
        / (da1.std(dims, ddof=1) * da2.std(dims, ddof=1))