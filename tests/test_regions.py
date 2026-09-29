# tests/test_regions.py

import numpy as np
import pytest

from ..src.lib.regions import extract_regions


def test_returns_no_regions_when_all_values_are_below_threshold():
    anomaly_map = np.zeros((5, 5), dtype=np.float32)

    assert extract_regions(anomaly_map, threshold=0.5) == []


def test_extracts_region_properties_from_rectangle():
    anomaly_map = np.zeros((6, 7), dtype=np.float32)
    anomaly_map[1:4, 2:6] = 1.0

    regions = extract_regions(anomaly_map, threshold=0.5)

    assert len(regions) == 1
    region = regions[0]
    assert region["region_id"] == "region_1"
    assert region["bbox"] == [2, 1, 4, 3]
    assert region["area"] == pytest.approx(6.0)
    assert region["compactness"] == pytest.approx(0.5)
    assert isinstance(region["polygon"], list)
    assert len(region["polygon"]) >= 3
    assert all(len(point) == 2 for point in region["polygon"])


def test_extracts_multiple_separate_regions():
    anomaly_map = np.zeros((8, 9), dtype=np.float32)
    anomaly_map[1:3, 1:3] = 1.0
    anomaly_map[5:7, 6:8] = 1.0

    regions = extract_regions(anomaly_map, threshold=0.5)

    assert len(regions) == 2
    assert {region["region_id"] for region in regions} == {"region_1", "region_2"}
    assert {tuple(region["bbox"]) for region in regions} == {
        (1, 1, 2, 2),
        (6, 5, 2, 2),
    }


def test_threshold_value_is_included():
    anomaly_map = np.zeros((3, 3), dtype=np.float32)
    anomaly_map[1, 1] = 0.5

    regions = extract_regions(anomaly_map, threshold=0.5)

    assert len(regions) == 1
    assert regions[0]["bbox"] == [1, 1, 1, 1]
    assert regions[0]["area"] == pytest.approx(0.0)
    assert regions[0]["compactness"] == pytest.approx(0.0)