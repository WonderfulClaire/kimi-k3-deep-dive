"""Single-head KDA state kernels, independently written in NumPy float64.

The state has shape [key_dim, value_dim].  Inputs are projected features rather
than hidden states: the neural projections, short convolution, output RMSNorm,
and output gate are outside this reference.  Alpha is a retention factor, NOT
the log-decay accepted by FLA.  See docs/foundation-models/01-kda.md for the
derivation and the explicit normalization/readout-scale conventions.

Public formula references:
https://arxiv.org/html/2607.24653v1#S2.SS1.SSS1
https://arxiv.org/html/2510.26692v1#S3
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _unit_rows(x: FloatArray) -> FloatArray:
    """Normalize nonzero rows; preserve zero rows without an arbitrary epsilon."""
    magnitude = np.max(np.abs(x), axis=-1, keepdims=True)
    scaled = np.divide(x, magnitude, out=np.zeros_like(x), where=magnitude != 0)
    norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
    return np.divide(scaled, norm, out=np.zeros_like(x), where=norm != 0)


def _prepare(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
    initial_state: ArrayLike | None,
    normalize_qk: bool,
    scale: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    q, k, v, alpha, beta = [
        np.asarray(x, dtype=np.float64) for x in (q, k, v, alpha, beta)
    ]
    if q.ndim != 2 or q.shape[1] == 0:
        raise ValueError("q must have shape [T, K] with K > 0")
    if k.shape != q.shape:
        raise ValueError("k must have the same [T, K] shape as q")
    if v.ndim != 2 or v.shape[0] != q.shape[0] or v.shape[1] == 0:
        raise ValueError("v must have shape [T, V] with V > 0")
    if alpha.shape != q.shape:
        raise ValueError("alpha must have shape [T, K], one gate per key channel")
    if beta.shape != (q.shape[0],):
        raise ValueError("beta must have shape [T], one write strength per token")
    if not all(np.all(np.isfinite(x)) for x in (q, k, v, alpha, beta)):
        raise ValueError("all KDA inputs must be finite")
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("alpha retention factors must be in [0, 1]")
    if np.any((beta < 0) | (beta > 1)):
        raise ValueError("beta write strengths must be in [0, 1]")
    if not isinstance(normalize_qk, (bool, np.bool_)):
        raise ValueError("normalize_qk must be a boolean")
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")
    state_shape = (q.shape[1], v.shape[1])
    if initial_state is None:
        state = np.zeros(state_shape, dtype=np.float64)
    else:
        state = np.array(initial_state, dtype=np.float64, copy=True)
        if state.shape != state_shape or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must be finite with shape [K, V]")
    if normalize_qk:
        q, k = _unit_rows(q), _unit_rows(k)
    return q * scale, k, v, alpha, beta, state


def kda_recurrent(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
    *,
    initial_state: ArrayLike | None = None,
    normalize_qk: bool = True,
    scale: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate decay -> predict -> delta write -> read for each token.

    q/k: [T,K], v: [T,V], alpha: [T,K], beta: [T].  Returns
    (outputs[T,V], final_state[K,V]); inputs are never modified.  q and k are
    L2-normalized by default.  Readout scale defaults to 1, matching the displayed
    recurrence; FLA's default scale is K**-0.5.  Boundary alpha/beta values 0 and 1
    are accepted for controlled interventions, beyond sigmoid-open intervals.
    """
    q, k, v, alpha, beta, state = _prepare(
        q, k, v, alpha, beta, initial_state, normalize_qk, scale
    )
    output = np.empty((len(q), v.shape[1]), dtype=np.float64)
    for t in range(len(q)):
        decayed = alpha[t, :, None] * state
        prediction = k[t] @ decayed
        state = decayed + np.outer(k[t], beta[t] * (v[t] - prediction))
        output[t] = q[t] @ state
    return output, state


def kda_chunkwise(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
    *,
    chunk_size: int = 64,
    initial_state: ArrayLike | None = None,
    normalize_qk: bool = True,
    scale: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Solve a causal lower-triangular system inside each chunk.

    This is a readable NumPy reference, not a FlashKDA speed reproduction.
    The within-chunk computation does not call the token recurrence.  It solves
    (I+L) Z = diag(beta) (V - (Gamma*K) @ incoming_state) for innovations Z,
    computes all chunk outputs, and carries the outgoing [K,V] state forward.

    Relative decay products are formed directly, with no division by cumulative
    alpha.  This supports zero gates and avoids reciprocal overflow.  The
    [C,C,K] temporary and generic np.linalg.solve favor clarity over GPU-style
    tiling; the final chunk may be shorter than chunk_size.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    q, k, v, alpha, beta, state = _prepare(
        q, k, v, alpha, beta, initial_state, normalize_qk, scale
    )
    output = np.empty((len(q), v.shape[1]), dtype=np.float64)
    for start in range(0, len(q), chunk_size):
        stop = min(start + chunk_size, len(q))
        qc, kc, vc, ac, bc = [x[start:stop] for x in (q, k, v, alpha, beta)]
        size = stop - start
        prefix = np.cumprod(ac, axis=0)

        # relative[i,j] = product(alpha[j+1:i+1]); current writes do not decay.
        relative = np.zeros((size, size, k.shape[1]), dtype=np.float64)
        for j in range(size):
            relative[j, j] = 1.0
            relative[j + 1 :, j] = np.cumprod(ac[j + 1 :], axis=0)

        interactions = np.einsum("ik,ijk,jk->ij", kc, relative, kc)
        lower = np.tril(bc[:, None] * interactions, k=-1)
        system = np.eye(size, dtype=np.float64) + lower
        rhs = bc[:, None] * (vc - (prefix * kc) @ state)
        innovations = np.linalg.solve(system, rhs)

        read_weights = np.einsum("ik,ijk,jk->ij", qc, relative, kc)
        output[start:stop] = (prefix * qc) @ state + np.tril(read_weights) @ innovations
        state = prefix[-1, :, None] * state + (relative[-1] * kc).T @ innovations
    return output, state


def k3_retention(
    logits: ArrayLike, *, log_scale: float = 0.0, g_min: float = -5.0
) -> FloatArray:
    """K3's bounded-decay mapping only: alpha = exp(g_min*sigmoid(exp(A)*z)).

    This helper does not implement K3's learned projections or output gate.
    In real arithmetic alpha lies in (exp(g_min), 1); floating-point sigmoid
    saturation may attain the endpoints.  Unlike Kimi Linear's negative-softplus
    mapping, the log retention has an explicit lower bound.
    """
    z = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(z)) or not np.isfinite(log_scale):
        raise ValueError("logits and log_scale must be finite")
    if not np.isfinite(g_min) or g_min >= 0:
        raise ValueError("g_min must be finite and negative")
    # Saturating at +/-inf in the scaled logit is safe for the sigmoid below.
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = z * np.exp(log_scale)
    scaled = np.where(z == 0, 0.0, scaled)
    sigmoid = np.exp(-np.logaddexp(0.0, -scaled))
    return np.asarray(np.exp(g_min * sigmoid), dtype=np.float64)
