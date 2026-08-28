import os

from covertlens.features.entropy import shannon_entropy


def test_all_zero_bytes_have_zero_entropy() -> None:
    assert shannon_entropy(bytes(1000)) == 0.0


def test_random_bytes_have_high_entropy() -> None:
    assert shannon_entropy(os.urandom(1000)) > 7.0


def test_empty_bytes_have_zero_entropy() -> None:
    assert shannon_entropy(b"") == 0.0


def test_domain_name_has_moderate_entropy() -> None:
    assert shannon_entropy(b"www.example.com") < 5.0
