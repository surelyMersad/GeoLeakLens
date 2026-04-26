"""Geodesic utilities for the geolocation pipeline.

Implements §11.1 (Haversine) and §11.2 (error clipping, threshold accuracy).
Pure-Python; no numpy/pandas dependency so this module is the cheapest unit
to import and to test.
"""
from __future__ import annotations

import math
from typing import Optional

# WGS-84 mean radius. Numeric value used by the standard Haversine
# implementation; switching radii changes city-pair distances by < 0.3%.
EARTH_RADIUS_KM = 6371.0088

# Earth half-circumference. §11.2: clip infinite errors here so medians and
# bootstraps stay finite. Approx π × EARTH_RADIUS_KM.
MAX_ERROR_KM = 20015.0


def is_valid_latlon(lat: Optional[float], lon: Optional[float]) -> bool:
    """True iff (lat, lon) is a finite real coordinate in valid ranges."""
    if lat is None or lon is None:
        return False
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat_f) or math.isnan(lon_f):
        return False
    if math.isinf(lat_f) or math.isinf(lon_f):
        return False
    return -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0


def haversine_km(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
) -> float:
    """Great-circle distance in km. Returns inf if any input is missing/invalid."""
    if not (is_valid_latlon(lat1, lon1) and is_valid_latlon(lat2, lon2)):
        return math.inf

    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def clip_error_km(error_km: Optional[float]) -> float:
    """Clip None / NaN / inf / negative errors into [0, MAX_ERROR_KM]."""
    if error_km is None:
        return MAX_ERROR_KM
    try:
        e = float(error_km)
    except (TypeError, ValueError):
        return MAX_ERROR_KM
    if math.isnan(e) or math.isinf(e):
        return MAX_ERROR_KM
    return min(max(e, 0.0), MAX_ERROR_KM)


def success_at(error_km: Optional[float], threshold_km: float) -> bool:
    """Threshold-accuracy success indicator. Inf / None / NaN error => False."""
    if error_km is None:
        return False
    try:
        e = float(error_km)
    except (TypeError, ValueError):
        return False
    if math.isnan(e) or math.isinf(e):
        return False
    return e <= float(threshold_km)
