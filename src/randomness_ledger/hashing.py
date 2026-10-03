"""Ideal random-function reference laws for the hashing toy, not SHA security claims."""

import math


def random_oracle_preimage_success(q: int, n_bits: int, m_bits: int) -> float:
    """Success against a uniform target using q distinct independent input queries.

    This averages over an ideal random function, not fixed SHA-256. A query set
    contains the target input with probability q/2**m; otherwise its q output
    matches are independent. Success means finding any matching preimage.
    """
    for value in (q, n_bits, m_bits):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("query budget and bit counts must be integers")
    if not 1 <= n_bits <= 256 or not 1 <= m_bits <= 62 or not 0 <= q <= 2**m_bits:
        raise ValueError("invalid random-oracle domain, range, or query budget")
    log_failure = q * math.log1p(-(2.0 ** -n_bits))
    collision_success = -math.expm1(log_failure)
    target_hit = q / float(2**m_bits)
    return target_hit + (1.0 - target_hit) * collision_success
