from contextlib import redirect_stdout
from io import StringIO

import pandas as pd

from scripts.inspect_features import inspect_features


def test_reports_opposite_size_cv_directions_per_protocol() -> None:
    features = pd.DataFrame(
        [
            ("dns", 0, 0.2, 0.1, 3.0, 0.5, 10.0, 0.0, None),
            ("dns", 1, 0.1, 0.2, 4.0, 0.9, 30.0, 0.5, None),
            ("icmp", 0, 0.1, 0.1, 3.0, 0.5, None, None, 0.1),
            ("icmp", 1, 0.3, 0.2, 4.0, 0.9, None, None, 0.3),
        ],
        columns=[
            "protocol",
            "label",
            "size_cv",
            "interarrival_cv",
            "entropy_mean",
            "compression_ratio_mean",
            "mean_query_length",
            "txt_null_ratio",
            "icmp_size_cv",
        ],
    )
    output = StringIO()

    with redirect_stdout(output):
        inspect_features(features)

    text = output.getvalue()
    assert "DNS FLOWS" in text and "ICMP FLOWS" in text
    assert "size_cv: covert=0.1000, legit=0.2000, direction=↓" in text
    assert "size_cv: covert=0.3000, legit=0.1000, direction=↑" in text
    assert "compression_ratio_mean: covert=0.9000, legit=0.5000, direction=↑" in text
    assert "(two-sided expected)" in text
