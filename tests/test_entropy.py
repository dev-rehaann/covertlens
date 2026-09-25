import os

from covertlens.features.entropy import compression_ratio, shannon_entropy


def test_all_zero_bytes_have_zero_entropy() -> None:
    assert shannon_entropy(bytes(1000)) == 0.0


def test_random_bytes_have_high_entropy() -> None:
    assert shannon_entropy(os.urandom(1000)) > 7.0


def test_empty_bytes_have_zero_entropy() -> None:
    assert shannon_entropy(b"") == 0.0
    assert compression_ratio(b"") == 0.0


def test_domain_name_has_moderate_entropy() -> None:
    assert shannon_entropy(b"www.example.com") < 5.0


def test_compression_distinguishes_pattern_from_random_data() -> None:
    ramp = bytes(range(256)) * 4
    random_data = os.urandom(1000)

    assert shannon_entropy(ramp) > 7.9
    assert compression_ratio(ramp) < 0.5
    assert shannon_entropy(random_data) > 7.0
    assert compression_ratio(random_data) > 0.9
