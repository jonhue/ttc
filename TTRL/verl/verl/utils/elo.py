import math
from typing import Sequence, Optional


def fit_cf_elo(
    P: Sequence[float],              # problem ratings P_i
    success_rate: Sequence[float],   # r_i in [0,1]
    attempts: Optional[Sequence[int]] = None,  # n_i (defaults to 1)
    eps: float = 1e-9,
):
    """
    Returns (R_hat, SE_R, converged).
    Uses bisection on the score equation sum_i (k_i - n_i p_i(R)) = 0.
    """
    assert len(P) == len(success_rate)
    n = [1]*len(P) if attempts is None else list(attempts)
    k = [max(0.0, min(ni, ni*ri)) for ni, ri in zip(n, success_rate)]

    a = math.log(10.0)/400.0

    def pred_success(R):
        # p_i(R) = sigmoid(a*(R - P_i))
        return [1.0/(1.0 + math.exp(a*(Pi - R))) for Pi in P]

    def score(R):
        # sum_i (k_i - n_i p_i(R))
        p = pred_success(R)
        return sum([ki - ni*pi for ki, ni, pi in zip(k, n, p)])

    # Bracket: ratings on CF live roughly ~ [800, 3600], widen a bit
    lo, hi = min(P) - 1200.0, max(P) + 1200.0
    # Ensure the root is bracketed
    while score(lo) < 0:
        lo -= 800
    while score(hi) > 0:
        hi += 800

    # Bisection
    for _ in range(80):
        mid = 0.5*(lo+hi)
        s = score(mid)
        if abs(s) < 1e-10:
            lo = hi = mid
            break
        if s > 0:
            lo = mid
        else:
            hi = mid
    R_hat = 0.5*(lo+hi)

    # SE via observed Fisher information
    p = pred_success(R_hat)
    I = (a*a) * sum(ni * pi * (1.0 - pi) for ni, pi in zip(n, p))
    SE = 1.0/math.sqrt(I + eps)

    return R_hat, SE, True
