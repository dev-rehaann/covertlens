"""Entropy features for packet payload analysis."""

import zlib
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


def compression_ratio(data: bytes) -> float:
    """Return maximum-zlib compressed size divided by original size.

    A low ratio (~0.1-0.3) indicates highly compressible, structured,
    repetitive, or patterned data, such as a deterministic ping payload ramp.
    A high ratio (~0.9-1.0) indicates incompressible data consistent with
    genuinely random, encoded, or already-compressed tunneled content. This
    complements :func:`shannon_entropy`, which cannot distinguish a repeating
    deterministic pattern from true randomness when both distribute byte
    values uniformly. Ratios can exceed 1.0 for short inputs because the zlib
    framing overhead is larger than any compression savings.
    """
    if not data:
        return 0.0
    return len(zlib.compress(data, level=9)) / len(data)
