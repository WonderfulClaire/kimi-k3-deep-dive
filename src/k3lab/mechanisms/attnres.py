"""Float64 attention-over-depth references, independent of a Transformer backend.

The embedding is a separate source. Layer outputs mean f_i(h_i), not the
already accumulated residual stream. Softmax is over source depth (axis 0);
RMSNorm touches keys only, while values remain in their original scale.
Full/Block compare fixed supplied outputs; they do not simulate the changes
to later Transformer sublayers that a different aggregation would cause.
"""

from __future__ import annotations

import math

import numpy as np


def rms_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Unit-gain RMSNorm over the last, feature axis; return float64."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x needs a nonempty feature axis")
    if eps < 0 or not np.isfinite(eps) or not np.all(np.isfinite(x)):
        raise ValueError("finite x and nonnegative finite eps required")
    denom = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    if np.any(denom == 0):
        raise ValueError("zero RMS requires positive eps")
    return x / denom


def depth_attention(
    sources: np.ndarray, pseudo_query: np.ndarray, *, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Mix [sources, ..., D] using one learned [D] pseudo-query.

    Returns mixed [..., D] and weights [sources, ...]. The published dot
    product has no extra 1/sqrt(D) factor. RMSNorm gain is fixed to one in this
    reference; a learned gain can be absorbed into the pseudo-query.
    """
    values = np.asarray(sources, dtype=np.float64)
    query = np.asarray(pseudo_query, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("sources must have shape [nonempty depth, ..., D]")
    if query.shape != (values.shape[-1],) or not np.all(np.isfinite(query)):
        raise ValueError("pseudo_query must be finite with shape [D]")
    keys = rms_norm(values, eps=eps)
    logits = np.einsum("s...d,d->s...", keys, query)
    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    weights = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    return np.sum(weights[..., None] * values, axis=0), weights


def _validate_sources(
    embedding: np.ndarray, layer_outputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.asarray(embedding, dtype=np.float64)
    outputs = np.asarray(layer_outputs, dtype=np.float64)
    # [] is a convenient empty-history spelling, including batched embeddings.
    if outputs.size == 0:
        outputs = np.empty((0, *embedding.shape), dtype=np.float64)
    if embedding.ndim < 1 or outputs.ndim != embedding.ndim + 1:
        raise ValueError("embedding [..., D], layer_outputs [L, ..., D] required")
    if outputs.shape[1:] != embedding.shape:
        raise ValueError("layer output shape must match embedding")
    if not np.all(np.isfinite(embedding)) or not np.all(np.isfinite(outputs)):
        raise ValueError("all source values must be finite")
    return embedding, outputs


def full_attnres(
    embedding: np.ndarray,
    layer_outputs: np.ndarray,
    pseudo_query: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Attend to embedding and every prior sublayer output separately."""
    embedding, outputs = _validate_sources(embedding, layer_outputs)
    sources = np.concatenate((embedding[None], outputs), axis=0)
    return depth_attention(sources, pseudo_query, eps=eps)


def block_sources(
    embedding: np.ndarray, layer_outputs: np.ndarray, *, block_size: int
) -> np.ndarray:
    """Build embedding + completed block sums + a nonempty partial sum.

    block_size counts individual f_i outputs in this API, not Transformer
    layers. An attention output and an MLP output each consume one slot. A
    block boundary never inserts an all-zero partial block as an extra source.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    embedding, outputs = _validate_sources(embedding, layer_outputs)
    sources = [embedding]
    for start in range(0, len(outputs), block_size):
        sources.append(np.sum(outputs[start : start + block_size], axis=0))
    return np.stack(sources, axis=0)


def block_attnres(
    embedding: np.ndarray,
    layer_outputs: np.ndarray,
    pseudo_query: np.ndarray,
    *,
    block_size: int,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Attend to block sums, with standard addition inside each block."""
    return depth_attention(
        block_sources(embedding, layer_outputs, block_size=block_size),
        pseudo_query,
        eps=eps,
    )


def attnres_storage_counts(
    num_layers: int,
    hidden_dim: int,
    *,
    block_size: int,
    batch_size: int = 1,
    seq_len: int = 1,
    dtype_bytes: int = 8,
) -> dict[str, int]:
    """Live source counts at final aggregation, not all training activations.

    num_layers follows the f_i-output convention of block_sources. A partial
    final block is one source; the embedding always adds one more source.
    """
    values = (num_layers, hidden_dim, block_size, batch_size, seq_len, dtype_bytes)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError("counts must be integers")
    if num_layers < 0 or min(values[1:]) <= 0:
        raise ValueError("nonnegative layers and positive dimensions required")
    full_sources = num_layers + 1
    block_source_count = math.ceil(num_layers / block_size) + 1
    per_source = batch_size * seq_len * hidden_dim
    return {
        "full_sources": full_sources,
        "block_sources": block_source_count,
        "full_elements": full_sources * per_source,
        "block_elements": block_source_count * per_source,
        "full_bytes": full_sources * per_source * dtype_bytes,
        "block_bytes": block_source_count * per_source * dtype_bytes,
    }
