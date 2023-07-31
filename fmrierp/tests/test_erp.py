import numpy as np
import pytest

from fmrierp.erp import create_extraction_window, round_onsets


def test_round_onsets_with_positive_t_r():
    onsets = np.array([0.2, 0.5, 1.0, 1.8, 2.2])
    t_r = 0.5
    expected_result = np.array([0, 1, 2, 4, 4])

    result = round_onsets(onsets, t_r)
    assert np.array_equal(result, expected_result)


def test_round_onsets_with_zero_t_r():
    onsets = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    t_r = 0.0
    with pytest.warns(RuntimeWarning):
        round_onsets(onsets, t_r)


def test_round_onsets_with_empty_array():
    onsets = np.array([])
    t_r = 0.5
    expected_result = np.array([])

    result = round_onsets(onsets, t_r)
    assert np.array_equal(result, expected_result)


def test_create_extraction_window_with_correct_size():
    window = [2.3, 4.7]
    t_r = 0.5
    expected_result = np.array([4, 9])

    result = create_extraction_window(window, t_r)
    assert np.array_equal(result, expected_result)


def test_create_extraction_window_with_incorrect_size():
    window = [1.0, 2.0, 3.0]
    t_r = 0.5
    with pytest.raises(ValueError, match="window has to be a list of length 2"):
        create_extraction_window(window, t_r)


def test_create_extraction_window_with_zero_t_r():
    window = [2.3, 4.7]
    t_r = 0.0
    with pytest.warns(RuntimeWarning):
        create_extraction_window(window, t_r)


def test_create_extraction_window_with_negative_t_r():
    window = [2.3, 4.7]
    t_r = -0.5
    expected_result = np.array([-4, -9])

    result = create_extraction_window(window, t_r)
    assert np.array_equal(result, expected_result)
