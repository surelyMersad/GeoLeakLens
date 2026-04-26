import math

import pytest

from geoleaklens.data.geo_utils import (
    EARTH_RADIUS_KM,
    MAX_ERROR_KM,
    clip_error_km,
    haversine_km,
    is_valid_latlon,
    success_at,
)


CITIES = {
    "san_francisco": (37.7749, -122.4194),
    "new_york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
}


def test_haversine_zero_distance():
    lat, lon = CITIES["san_francisco"]
    assert haversine_km(lat, lon, lat, lon) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "a,b,expected_km,tol_km",
    [
        ("san_francisco", "new_york", 4129.0, 30.0),
        ("london", "new_york", 5570.0, 30.0),
        ("london", "tokyo", 9560.0, 50.0),
        ("sydney", "tokyo", 7822.0, 50.0),
    ],
)
def test_haversine_known_pairs(a, b, expected_km, tol_km):
    la1, lo1 = CITIES[a]
    la2, lo2 = CITIES[b]
    d = haversine_km(la1, lo1, la2, lo2)
    assert abs(d - expected_km) < tol_km, f"got {d:.1f} km, expected ~{expected_km} km"


def test_haversine_symmetric():
    a = CITIES["san_francisco"]
    b = CITIES["tokyo"]
    assert haversine_km(*a, *b) == pytest.approx(haversine_km(*b, *a), abs=1e-6)


def test_haversine_invalid_returns_inf():
    assert haversine_km(None, 0.0, 0.0, 0.0) == math.inf
    assert haversine_km(0.0, None, 0.0, 0.0) == math.inf
    assert haversine_km(91.0, 0.0, 0.0, 0.0) == math.inf
    assert haversine_km(0.0, 181.0, 0.0, 0.0) == math.inf
    assert haversine_km(float("nan"), 0.0, 0.0, 0.0) == math.inf
    assert haversine_km(float("inf"), 0.0, 0.0, 0.0) == math.inf


def test_is_valid_latlon():
    assert is_valid_latlon(0.0, 0.0)
    assert is_valid_latlon(90.0, 180.0)
    assert is_valid_latlon(-90.0, -180.0)
    assert not is_valid_latlon(None, 0.0)
    assert not is_valid_latlon(0.0, None)
    assert not is_valid_latlon(90.1, 0.0)
    assert not is_valid_latlon(0.0, 180.1)
    assert not is_valid_latlon(float("nan"), 0.0)
    assert not is_valid_latlon(float("inf"), 0.0)


def test_clip_error_km():
    assert clip_error_km(100.0) == 100.0
    assert clip_error_km(0.0) == 0.0
    assert clip_error_km(math.inf) == MAX_ERROR_KM
    assert clip_error_km(-math.inf) == MAX_ERROR_KM
    assert clip_error_km(float("nan")) == MAX_ERROR_KM
    assert clip_error_km(None) == MAX_ERROR_KM
    assert clip_error_km(-5.0) == 0.0
    assert clip_error_km(MAX_ERROR_KM + 1) == MAX_ERROR_KM


def test_success_at_thresholds():
    assert success_at(10.0, 25.0)
    assert success_at(25.0, 25.0)
    assert not success_at(25.1, 25.0)
    assert not success_at(math.inf, 25.0)
    assert not success_at(float("nan"), 25.0)
    assert not success_at(None, 25.0)


def test_max_error_is_half_circumference():
    # MAX_ERROR_KM should be approximately π × Earth radius (half-circumference).
    assert abs(MAX_ERROR_KM - math.pi * EARTH_RADIUS_KM) < 5.0


def test_haversine_antipode_close_to_max():
    # San Francisco -> its antipode (~south of Madagascar in the Indian Ocean)
    # should be close to the half-circumference clip.
    lat, lon = CITIES["san_francisco"]
    antipode = (-lat, lon + 180 if lon + 180 <= 180 else lon - 180)
    d = haversine_km(lat, lon, *antipode)
    assert abs(d - MAX_ERROR_KM) < 50.0
