"""Float64 MoE routing/dispatch and symbolic latent-width accounting.

topk_route implements sigmoid-score selection with an independent expert
bias. heuristic_balance_bias is deliberately not named Quantile Balancing.
quantile_balance_bias is an exact, small-batch threshold reference for K3
Eq. (14), without its distributed histogram implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoutePlan:
    indices: np.ndarray  # [T, k], expert IDs
    weights: np.ndarray  # [T, k], normalized ORIGINAL scores
    loads: np.ndarray  # [E], assignment counts, sum = T*k


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Stable sigmoid, including large finite positive/negative logits."""
    x = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("logits must be finite")
    return np.exp(-np.logaddexp(0.0, -x))


def _scores_and_bias(scores: np.ndarray, bias: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) <= 0:
        raise ValueError("scores must have shape [nonempty T, nonempty E]")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("scores must be finite sigmoid probabilities in [0, 1]")
    expert_bias = np.zeros(scores.shape[1], dtype=np.float64) if bias is None else np.asarray(bias, dtype=np.float64)
    if expert_bias.shape != (scores.shape[1],) or not np.all(np.isfinite(expert_bias)):
        raise ValueError("bias must be finite with shape [E]")
    return scores, expert_bias


def topk_route(scores: np.ndarray, k: int, bias: np.ndarray | None = None) -> RoutePlan:
    """Select on scores+bias, combine with raw scores normalized over Top-k.

    Stable sorting breaks selection ties by lower expert ID. Zero selected
    mass raises ValueError rather than silently inventing mixture weights.
    """
    scores, bias = _scores_and_bias(scores, bias)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= scores.shape[1]:
        raise ValueError("k must be an integer in [1, E]")
    indices = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :k]
    selected = np.take_along_axis(scores, indices, axis=-1)
    mass = np.sum(selected, axis=-1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("selected raw scores must have positive total mass")
    weights = selected / mass
    loads = np.bincount(indices.ravel(), minlength=scores.shape[1])
    return RoutePlan(indices=indices, weights=weights, loads=loads)


def route_logits(logits: np.ndarray, k: int, bias: np.ndarray | None = None) -> RoutePlan:
    return topk_route(sigmoid(logits), k, bias)


def _validate_linear_inputs(
    tokens: np.ndarray, expert_weights: np.ndarray, plan: RoutePlan
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(tokens, dtype=np.float64)
    expert_weights = np.asarray(expert_weights, dtype=np.float64)
    if tokens.ndim != 2 or expert_weights.ndim != 3:
        raise ValueError("tokens [T,D], expert_weights [E,D,Dout] required")
    if tokens.shape[1] != expert_weights.shape[1]:
        raise ValueError("expert input dimension does not match tokens")
    if plan.indices.ndim != 2 or plan.indices.shape != plan.weights.shape:
        raise ValueError("route indices and weights must both be [T,k]")
    if plan.indices.shape[0] != tokens.shape[0] or len(plan.loads) != len(expert_weights):
        raise ValueError("route token/expert counts do not match inputs")
    if not np.issubdtype(plan.indices.dtype, np.integer) or np.any(plan.indices < 0) or np.any(plan.indices >= len(expert_weights)):
        raise ValueError("route contains invalid expert IDs")
    if not np.all(np.isfinite(tokens)) or not np.all(np.isfinite(expert_weights)):
        raise ValueError("tokens and weights must be finite")
    if not np.all(np.isfinite(plan.weights)) or np.any(plan.weights < 0):
        raise ValueError("route weights must be finite and nonnegative")
    if not np.allclose(plan.weights.sum(axis=-1), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("route weights must sum to one per token")
    return tokens, expert_weights


def moe_linear_dispatch(tokens: np.ndarray, expert_weights: np.ndarray, plan: RoutePlan) -> np.ndarray:
    """Expert-major dispatch/combine with linear experts for a routing test.

    Empty experts execute no matrix multiplication. np.add.at returns each
    weighted output to its original token and accumulates its k contributions.
    """
    tokens, expert_weights = _validate_linear_inputs(tokens, expert_weights, plan)
    out = np.zeros((len(tokens), expert_weights.shape[-1]), dtype=np.float64)
    for expert in range(len(expert_weights)):
        token_ids, slots = np.nonzero(plan.indices == expert)
        if len(token_ids):
            expert_out = tokens[token_ids] @ expert_weights[expert]
            np.add.at(out, token_ids, plan.weights[token_ids, slots, None] * expert_out)
    return out


def moe_linear_reference(tokens: np.ndarray, expert_weights: np.ndarray, plan: RoutePlan) -> np.ndarray:
    """Independent token-major scalar loop reference for dispatch/combine."""
    tokens, expert_weights = _validate_linear_inputs(tokens, expert_weights, plan)
    out = np.zeros((len(tokens), expert_weights.shape[-1]), dtype=np.float64)
    for token in range(len(tokens)):
        for slot, expert in enumerate(plan.indices[token]):
            for dst in range(expert_weights.shape[-1]):
                value = 0.0
                for src in range(tokens.shape[1]):
                    value += tokens[token, src] * expert_weights[expert, src, dst]
                out[token, dst] += plan.weights[token, slot] * value
    return out


def heuristic_balance_bias(loads: np.ndarray, bias: np.ndarray | None = None, *, step_size: float = 0.01) -> np.ndarray:
    """Sign-based auxiliary-loss-free teaching baseline, NOT K3 QB."""
    loads = np.asarray(loads, dtype=np.float64)
    if loads.ndim != 1 or len(loads) == 0 or np.any(loads < 0) or not np.all(np.isfinite(loads)):
        raise ValueError("loads must be a finite nonnegative [E] array")
    if step_size < 0 or not np.isfinite(step_size):
        raise ValueError("step_size must be finite and nonnegative")
    bias = np.zeros_like(loads) if bias is None else np.asarray(bias, dtype=np.float64)
    if bias.shape != loads.shape or not np.all(np.isfinite(bias)):
        raise ValueError("bias must match [E] loads")
    updated = bias + step_size * np.sign(np.mean(loads) - loads)
    return updated - np.mean(updated)


def quantile_balance_bias(scores: np.ndarray, k: int, bias: np.ndarray | None = None) -> np.ndarray:
    """Exact order-statistic threshold reference, applied to the NEXT batch.

    Integer q=T*k/E and 1<=k<E are required. For each expert, take the
    (q+1)-th largest raw-score-minus-old-cutoff margin; negate and center.
    No-tie fixed-cutoff column counts equal q, but simultaneous rerouting is
    not promised to have perfect loads. Ties can prevent exact column counts.
    This omits distributed histogram approximation and online scheduling.
    """
    scores, bias = _scores_and_bias(scores, bias)
    tokens, experts = scores.shape
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k < experts:
        raise ValueError("quantile reference needs integer 1 <= k < E")
    if (tokens * k) % experts:
        raise ValueError("exact target T*k/E must be an integer in this reference")
    target = tokens * k // experts
    if not 1 <= target < tokens:
        raise ValueError("exact target must be between 1 and T-1")
    cutoff = np.sort(scores + bias, axis=1)[:, -(k + 1)]
    margins = scores - cutoff[:, None]
    threshold = np.sort(margins, axis=0)[::-1][target]
    updated = -threshold
    return updated - np.mean(updated)


def load_statistics(plan: RoutePlan) -> dict[str, float]:
    mean = float(np.mean(plan.loads))
    return {
        "mean": mean,
        "max_over_mean": float(np.max(plan.loads) / mean) if mean else 0.0,
        "coefficient_of_variation": float(np.std(plan.loads) / mean) if mean else 0.0,
    }


def latent_moe_cost(
    hidden_dim: int,
    latent_dim: int,
    ffn_dim: int,
    num_experts: int,
    top_k: int,
    *,
    tokens: int = 1,
    shared_experts: int = 0,
    dtype_bytes: int = 2,
) -> dict[str, int]:
    """Count a gated 3-matrix expert architecture, NOT the linear toy experts.

    Routed experts are ell -> ffn_dim -> ell; shared experts are full-width
    d -> ffn_dim -> d. Only the intermediate FFN dimension is assumed equal.
    All linear-layer biases and the independent routing-control bias state
    are excluded. Two routed-path d<->latent projections, a d*E router, and
    latent RMSNorm gain are included. Communication is logical dispatch+combine payload assuming
    every assignment is remote; metadata, alignment, topology and replication
    are excluded. Counts are formulas, not measured runtime or network bytes.
    """
    dims = (hidden_dim, latent_dim, ffn_dim, num_experts, top_k, tokens, shared_experts, dtype_bytes)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in dims):
        raise ValueError("dimensions/counts must be integers")
    if min(dims[:6]) <= 0 or shared_experts < 0 or dtype_bytes <= 0 or top_k > num_experts:
        raise ValueError("positive dimensions, 1<=top_k<=experts and nonnegative shared experts required")
    d, ell, h, e, k, t = dims[:6]
    routed_params = 3 * e * ell * h
    shared_params = 3 * shared_experts * d * h
    projections = 2 * d * ell
    router = d * e
    fixed = projections + router + ell
    dispatch_elements = t * k * ell
    return {
        "routed_expert_parameters": routed_params,
        "full_width_routed_expert_parameters": 3 * e * d * h,
        "shared_expert_parameters": shared_params,
        "projection_parameters": projections,
        "router_parameters": router,
        "norm_parameters": ell,
        "total_parameters": routed_params + shared_params + fixed,
        "active_parameters_per_token": 3 * k * ell * h + shared_params + fixed,
        "matrix_madds": t * (3 * k * ell * h + shared_params + projections + router),
        "routed_matrix_madds": t * 3 * k * ell * h,
        "shared_matrix_madds": t * shared_params,
        "projection_matrix_madds": t * projections,
        "router_matrix_madds": t * router,
        "dispatch_elements": dispatch_elements,
        "combine_elements": dispatch_elements,
        "logical_payload_bytes": 2 * dispatch_elements * dtype_bytes,
        "full_width_logical_payload_bytes": 2 * t * k * d * dtype_bytes,
    }
