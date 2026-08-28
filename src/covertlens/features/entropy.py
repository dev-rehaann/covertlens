"""Entropy features for packet payload analysis."""

from collections import Counter
from math import log2


def shannon_entropy(data: bytes) -> float:
    """Return Shannon entropy in bits per byte, in the range 0 to 8.

    The calculation is ``H(X) = -sum(p(x) * log2(p(x)))`` for the observed
    frequency ``p(x)`` of each byte value. Low entropy (~0-3) indicates
    repetitive or structured data typical of legitimate protocol fields.
    High entropy (~7-8) indicates compressed or encrypted data typical of
    tunneled/exfiltrated content. Base32/Base64 text is alphabet-limited to
    about 5/6 bits per encoded byte but can still be elevated relative to
    ordinary protocol text.
    """
    if not data:
        return 0.0

    length = len(data)
    return -sum(
        (count / length) * log2(count / length) for count in Counter(data).values()
    )
