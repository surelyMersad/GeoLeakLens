from geoleaklens.scoring.parse_predictions import parse_geolocation_response


def test_dict_input_passes_through_geoclip_shape():
    raw = {
        "lat": 37.7749,
        "lon": -122.4194,
        "confidence": 0.9,
        "raw_response": "topk=...",
    }
    out = parse_geolocation_response(raw)
    assert out["parse_success"] is True
    assert out["error"] is None
    assert out["parsed"]["lat"] == 37.7749
    assert out["parsed"]["lon"] == -122.4194
    assert out["parsed"]["confidence"] == 0.9
    assert out["parsed"]["country"] is None
    assert out["parsed"]["evidence"] == []


def test_strict_json_string_with_latitude_longitude_keys():
    raw = (
        '{"country": "France", "city": "Paris", '
        '"latitude": 48.8566, "longitude": 2.3522, '
        '"confidence": 0.7, "visual_evidence": ["eiffel tower"]}'
    )
    out = parse_geolocation_response(raw)
    assert out["parse_success"] is True
    assert out["parsed"]["lat"] == 48.8566
    assert out["parsed"]["lon"] == 2.3522
    assert out["parsed"]["country"] == "France"
    assert out["parsed"]["city"] == "Paris"
    assert out["parsed"]["evidence"] == ["eiffel tower"]


def test_regex_fallback_extracts_first_object():
    raw = 'Here is my answer: {"latitude": 51.5, "longitude": -0.12} -- done.'
    out = parse_geolocation_response(raw)
    assert out["parse_success"] is True
    assert out["parsed"]["lat"] == 51.5
    assert out["parsed"]["lon"] == -0.12


def test_invalid_response_marks_parse_failed():
    out = parse_geolocation_response("I cannot tell where this is.")
    assert out["parse_success"] is False
    assert out["parsed"]["lat"] is None
    assert out["parsed"]["lon"] is None
    assert out["parsed"]["evidence"] == []
    assert out["error"]


def test_out_of_range_lat_lon_set_to_none():
    raw = '{"latitude": 99.0, "longitude": 200.0}'
    out = parse_geolocation_response(raw)
    assert out["parse_success"] is True
    assert out["parsed"]["lat"] is None
    assert out["parsed"]["lon"] is None


def test_partial_lat_lon_set_to_none():
    raw = '{"latitude": 40.0, "longitude": null}'
    out = parse_geolocation_response(raw)
    assert out["parse_success"] is True
    assert out["parsed"]["lat"] is None
    assert out["parsed"]["lon"] is None


def test_confidence_clipped_to_unit_interval():
    raw = '{"latitude": 0.0, "longitude": 0.0, "confidence": 2.5}'
    out = parse_geolocation_response(raw)
    assert out["parsed"]["confidence"] == 1.0

    raw = '{"latitude": 0.0, "longitude": 0.0, "confidence": -0.4}'
    out = parse_geolocation_response(raw)
    assert out["parsed"]["confidence"] == 0.0


def test_evidence_handles_string_or_list():
    raw = '{"latitude": 0.0, "longitude": 0.0, "visual_evidence": "a sign"}'
    out = parse_geolocation_response(raw)
    assert out["parsed"]["evidence"] == ["a sign"]

    raw = '{"latitude": 0.0, "longitude": 0.0, "visual_evidence": ["a", "b"]}'
    out = parse_geolocation_response(raw)
    assert out["parsed"]["evidence"] == ["a", "b"]


def test_none_input():
    out = parse_geolocation_response(None)
    assert out["parse_success"] is False
    assert out["parsed"]["lat"] is None
