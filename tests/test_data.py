"""Checks that the train/validation split is leakage-free and the label mapping is stable."""

import pytest

from intent_bench.data import load_banking77, make_split, normalize

N_CLASSES = 77


@pytest.fixture(scope="module")
def data():
    """Load the dataset and build the split once for all tests in this file."""
    train, test, label_names = load_banking77()
    train_idx, val_idx = make_split(train)
    return train, test, label_names, train_idx, val_idx


def test_normalize():
    assert normalize("  Where IS\tmy   card?\n") == "where is my card?"


def test_no_text_overlap_between_train_and_val(data):
    train, _, _, train_idx, val_idx = data
    train_texts = set(train["norm"].iloc[train_idx])
    val_texts = set(train["norm"].iloc[val_idx])
    assert train_texts.isdisjoint(val_texts)


def test_split_covers_every_row_exactly_once(data):
    train, _, _, train_idx, val_idx = data
    assert sorted(train_idx + val_idx) == list(range(len(train)))


def test_all_classes_in_both_splits(data):
    train, _, _, train_idx, val_idx = data
    assert train["label"].iloc[train_idx].nunique() == N_CLASSES
    assert train["label"].iloc[val_idx].nunique() == N_CLASSES


def test_validation_is_about_20_percent(data):
    train, _, _, _, val_idx = data
    assert 0.18 <= len(val_idx) / len(train) <= 0.22


def test_split_is_reproducible(data):
    train, _, _, train_idx, val_idx = data
    assert make_split(train) == (train_idx, val_idx)


def test_label_mapping(data):
    train, test, label_names, _, _ = data
    assert len(label_names) == N_CLASSES
    # Names are alphabetical ignoring case: one label starts with a capital letter.
    assert label_names == sorted(label_names, key=str.lower)
    assert label_names[0] == "activate_my_card"
    assert label_names[11] == "card_arrival"
    # Irregular names that must be preserved exactly (LLM outputs are matched against them).
    assert label_names[51] == "Refund_not_showing_up"
    assert "reverted_card_payment?" in label_names
    # Known example: first training message.
    assert train["text"].iloc[0] == "I am still waiting on my card?"
    assert label_names[train["label"].iloc[0]] == "card_arrival"
    assert set(test["label"]) == set(range(N_CLASSES))
