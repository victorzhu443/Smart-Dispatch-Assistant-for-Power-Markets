"""Tests for the train/test split.

These exist because the original split called sklearn's train_test_split
without shuffle=False. On a time series that lets the model train on later
hours and be scored on earlier ones -- it sees the future, and every score it
reports is inflated. These tests fail if that regresses.

Run with:  pytest tests/ -v
"""
import importlib

import numpy as np
import pytest


phase_3_1 = importlib.import_module('forecasting_model.phase_3_1_load_features')


@pytest.fixture
def ordered_data():
    """100 samples already in time order, values encoding their position."""
    n = 100
    X = np.arange(n, dtype=float).reshape(n, 1)
    y = np.arange(n, dtype=float)
    sequences = np.tile(np.arange(n, dtype=float).reshape(n, 1), (1, 24))
    return X, y, sequences


def test_test_set_is_strictly_after_train_set(ordered_data):
    """The core property: every test sample postdates every train sample."""
    X, y, sequences = ordered_data
    X_train, X_test, y_train, y_test, _, _ = phase_3_1.create_train_test_split(
        X, y, sequences, test_size=0.2
    )

    assert y_train.max() < y_test.min(), (
        "train set overlaps or postdates test set -- the model can see the future"
    )


def test_split_preserves_order_and_does_not_shuffle(ordered_data):
    """Both halves stay in their original sequence."""
    X, y, sequences = ordered_data
    _, _, y_train, y_test, _, _ = phase_3_1.create_train_test_split(
        X, y, sequences, test_size=0.2
    )

    np.testing.assert_array_equal(y_train, np.sort(y_train))
    np.testing.assert_array_equal(y_test, np.sort(y_test))
    np.testing.assert_array_equal(y_train, np.arange(80, dtype=float))
    np.testing.assert_array_equal(y_test, np.arange(80, 100, dtype=float))


def test_split_is_deterministic(ordered_data):
    """Two calls on the same input produce identical splits."""
    X, y, sequences = ordered_data
    first = phase_3_1.create_train_test_split(X, y, sequences, test_size=0.2)
    second = phase_3_1.create_train_test_split(X, y, sequences, test_size=0.2)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_sequences_stay_aligned_with_their_targets(ordered_data):
    """A row's 24-hour sequence must still belong to that row after splitting."""
    X, y, sequences = ordered_data
    _, _, y_train, y_test, seq_train, seq_test = phase_3_1.create_train_test_split(
        X, y, sequences, test_size=0.2
    )

    # Each sequence row was built as [i] * 24, so it must match its target.
    np.testing.assert_array_equal(seq_train[:, 0], y_train)
    np.testing.assert_array_equal(seq_test[:, 0], y_test)


def test_rejects_a_split_that_would_empty_either_side(ordered_data):
    X, y, sequences = ordered_data

    with pytest.raises(ValueError):
        phase_3_1.create_train_test_split(X, y, sequences, test_size=1.0)

    with pytest.raises(ValueError):
        phase_3_1.create_train_test_split(X, y, sequences, test_size=0.0)


def test_works_without_price_sequences(ordered_data):
    X, y, _ = ordered_data
    _, _, y_train, y_test, seq_train, seq_test = phase_3_1.create_train_test_split(
        X, y, None, test_size=0.2
    )

    assert seq_train is None and seq_test is None
    assert y_train.max() < y_test.min()
