"""Independent affine expansion and interventions for the KDA state kernel."""

import numpy as np
import pytest

from k3lab.mechanisms.kda import k3_retention, kda_chunkwise, kda_recurrent


def _data(seed=7, length=11, key_dim=3, value_dim=5):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(length, key_dim)),
        rng.normal(size=(length, key_dim)),
        rng.normal(size=(length, value_dim)),
        rng.uniform(0.15, 1.0, size=(length, key_dim)),
        rng.uniform(0.0, 1.0, size=length),
        rng.normal(size=(key_dim, value_dim)),
    )


def _dense_affine_expansion(q, k, v, alpha, beta, initial_state, scale=1.0):
    """Expand products A_t ... A_j explicitly; never use a residual update."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    k = k / np.linalg.norm(k, axis=1, keepdims=True)
    key_dim = k.shape[1]
    transitions = [
        (np.eye(key_dim) - b * np.outer(key, key)) @ np.diag(a)
        for key, a, b in zip(k, alpha, beta)
    ]
    writes = [b * np.outer(key, value) for key, value, b in zip(k, v, beta)]
    output = []
    for t in range(len(q)):
        product = np.eye(key_dim)
        state = np.zeros_like(initial_state)
        for j in range(t, -1, -1):
            state = state + product @ writes[j]
            product = product @ transitions[j]
        state = state + product @ initial_state
        output.append(scale * q[t] @ state)
    return np.array(output), state


@pytest.mark.parametrize("key_dim,value_dim", [(2, 5), (5, 2), (1, 4)])
@pytest.mark.parametrize("chunk_size", [1, 3, 4, 64])
def test_recurrence_and_chunk_match_dense_affine_expansion(key_dim, value_dim, chunk_size):
    q, k, v, alpha, beta, state0 = _data(key_dim=key_dim, value_dim=value_dim)
    expected = _dense_affine_expansion(q, k, v, alpha, beta, state0, scale=0.7)
    for algorithm, kwargs in [(kda_recurrent, {}), (kda_chunkwise, {"chunk_size": chunk_size})]:
        actual = algorithm(q, k, v, alpha, beta, initial_state=state0, scale=0.7, **kwargs)
        for x, y in zip(actual, expected):
            np.testing.assert_allclose(x, y, rtol=2e-12, atol=2e-12)
            assert x.dtype == np.float64


@pytest.mark.parametrize("algorithm", [kda_recurrent, kda_chunkwise])
def test_future_perturbation_cannot_change_past_outputs(algorithm):
    q, k, v, alpha, beta, state0 = _data(length=15)
    baseline, _ = algorithm(q, k, v, alpha, beta, initial_state=state0)
    altered = [x.copy() for x in (q, k, v, alpha, beta)]
    altered[0][8:] *= -3.0
    altered[1][8:] = np.roll(altered[1][8:], 1, axis=-1)
    altered[2][8:] += 20.0
    altered[3][8:] = 0.0
    altered[4][8:] = 1.0
    changed, _ = algorithm(*altered, initial_state=state0)
    np.testing.assert_allclose(changed[:8], baseline[:8], rtol=0, atol=2e-12)
    assert not np.allclose(changed[8:], baseline[8:])


@pytest.mark.parametrize("algorithm", [kda_recurrent, kda_chunkwise])
def test_beta_zero_means_decay_without_new_write(algorithm):
    q, k, v, alpha, beta, state0 = _data()
    beta[:] = 0.0
    outputs, final = algorithm(q, k, v, alpha, beta, initial_state=state0)
    cumulative = np.cumprod(alpha, axis=0)
    normalized_q = q / np.linalg.norm(q, axis=1, keepdims=True)
    expected = np.stack([qt @ (a[:, None] * state0) for qt, a in zip(normalized_q, cumulative)])
    np.testing.assert_allclose(outputs, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(final, cumulative[-1, :, None] * state0, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("algorithm", [kda_recurrent, kda_chunkwise])
def test_beta_one_overwrites_current_unit_key_direction(algorithm):
    _, k, v, alpha, beta, state0 = _data()
    outputs, _ = algorithm(k, k, v, alpha, np.ones_like(beta), initial_state=state0)
    np.testing.assert_allclose(outputs, v, rtol=2e-12, atol=2e-12)


def test_beta_one_preserves_decayed_orthogonal_component():
    key = np.array([[3.0, 4.0]])
    orthogonal = np.array([-0.8, 0.6])
    state0 = np.array([[2.0, 1.0, 5.0], [-1.0, 3.0, 2.0]])
    alpha = np.array([[0.25, 0.75]])
    _, final = kda_recurrent(key, key, [[4.0, -2.0, 7.0]], alpha, [1.0], initial_state=state0)
    np.testing.assert_allclose(orthogonal @ final, orthogonal @ (alpha[0, :, None] * state0))


def test_gate_is_applied_before_delta_erasure():
    q = k = np.array([[1.0, 1.0]]) / np.sqrt(2.0)
    v = np.array([[4.0, -2.0]])
    alpha, beta = np.array([[0.2, 0.9]]), np.array([0.6])
    state0 = np.array([[2.0, 1.0], [-1.0, 3.0]])
    output, final = kda_chunkwise(q, k, v, alpha, beta, initial_state=state0)
    projection = np.eye(2) - beta[0] * np.outer(k[0], k[0])
    right = projection @ np.diag(alpha[0]) @ state0 + beta[0] * np.outer(k[0], v[0])
    wrong = np.diag(alpha[0]) @ projection @ state0 + beta[0] * np.outer(k[0], v[0])
    np.testing.assert_allclose(final, right)
    np.testing.assert_allclose(output[0], q[0] @ right)
    assert not np.allclose(final, wrong)


def test_two_dimensional_hand_calculation_reads_current_write():
    q = k = np.eye(2)
    values = np.array([[2.0, 1.0], [4.0, -2.0]])
    alpha = np.array([[0.5, 0.25], [0.5, 1.0]])
    beta = np.array([0.5, 1.0])
    expected_output = np.array([[1.0, 0.5], [4.0, -2.0]])
    expected_state = np.array([[0.5, 0.25], [4.0, -2.0]])
    for algorithm in (kda_recurrent, kda_chunkwise):
        output, final = algorithm(q, k, values, alpha, beta)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(final, expected_state)


def test_chunk_state_can_resume_with_different_chunk_boundaries():
    q, k, v, alpha, beta, state0 = _data(length=19)
    full_output, full_state = kda_recurrent(q, k, v, alpha, beta, initial_state=state0)
    first_output, first_state = kda_chunkwise(q[:7], k[:7], v[:7], alpha[:7], beta[:7], initial_state=state0, chunk_size=3)
    second_output, second_state = kda_chunkwise(q[7:], k[7:], v[7:], alpha[7:], beta[7:], initial_state=first_state, chunk_size=5)
    np.testing.assert_allclose(np.concatenate([first_output, second_output]), full_output, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(second_state, full_state, rtol=2e-12, atol=2e-12)


def test_zero_retention_and_underflow_do_not_require_inverse_decay():
    q, k, v, alpha, beta, state0 = _data(length=40)
    alpha[:] = 1e-30
    alpha[5:9, 0] = 0.0
    expected = kda_recurrent(q, k, v, alpha, beta, initial_state=state0)
    actual = kda_chunkwise(q, k, v, alpha, beta, initial_state=state0, chunk_size=40)
    for x, y in zip(actual, expected):
        assert np.all(np.isfinite(x))
        np.testing.assert_allclose(x, y, rtol=2e-12, atol=2e-12)


def test_chunk_calls_a_real_unit_lower_triangular_solve(monkeypatch):
    calls = []
    original_solve = np.linalg.solve

    def checked_solve(matrix, rhs):
        np.testing.assert_array_equal(matrix, np.tril(matrix))
        np.testing.assert_array_equal(np.diag(matrix), np.ones(len(matrix)))
        result = original_solve(matrix, rhs)
        np.testing.assert_allclose(matrix @ result, rhs, rtol=1e-12, atol=1e-12)
        calls.append(matrix.shape)
        return result

    monkeypatch.setattr(np.linalg, "solve", checked_solve)
    q, k, v, alpha, beta, _ = _data(length=11)
    kda_chunkwise(q, k, v, alpha, beta, chunk_size=4)
    assert calls == [(4, 4), (4, 4), (3, 3)]


@pytest.mark.parametrize("algorithm", [kda_recurrent, kda_chunkwise])
def test_empty_input_preserves_initial_state_and_does_not_alias(algorithm):
    state0 = np.arange(6.0).reshape(2, 3)
    output, state = algorithm(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 2)), np.empty(0), initial_state=state0)
    assert output.shape == (0, 3)
    np.testing.assert_array_equal(state, state0)
    state[:] = -1
    assert state0[0, 0] == 0


@pytest.mark.parametrize("algorithm", [kda_recurrent, kda_chunkwise])
def test_zero_query_or_key_and_explicit_raw_feature_mode(algorithm):
    output, state = algorithm([[0, 0], [1, 0]], [[0, 0], [2, 0]], [[5], [3]], np.ones((2, 2)), [1, 0.5], normalize_qk=False)
    np.testing.assert_array_equal(output, [[0.0], [3.0]])
    np.testing.assert_array_equal(state, [[3.0], [0.0]])
    normalized, _ = algorithm([[0, 0], [1, 0]], [[0, 0], [2, 0]], [[5], [3]], np.ones((2, 2)), [1, 0.5])
    np.testing.assert_array_equal(normalized, [[0.0], [1.5]])


def test_inputs_are_not_mutated_and_scaling_only_changes_readout():
    args = _data()
    saved = [x.copy() for x in args]
    base, state = kda_chunkwise(*args[:5], initial_state=args[5])
    scaled, scaled_state = kda_chunkwise(*args[:5], initial_state=args[5], scale=0.25)
    np.testing.assert_allclose(scaled, base * 0.25)
    np.testing.assert_allclose(scaled_state, state)
    for x, previous in zip(args, saved):
        np.testing.assert_array_equal(x, previous)


@pytest.mark.parametrize("algorithm", [kda_recurrent, kda_chunkwise])
@pytest.mark.parametrize("invalid", ["alpha_shape", "beta_shape", "negative_alpha", "beta_above_one", "nan", "state_shape"])
def test_invalid_semantics_fail_early(algorithm, invalid):
    q, k, v, alpha, beta, state0 = _data()
    if invalid == "alpha_shape":
        alpha = alpha[:, 0]
    elif invalid == "beta_shape":
        beta = beta[:, None]
    elif invalid == "negative_alpha":
        alpha[0, 0] = -0.01
    elif invalid == "beta_above_one":
        beta[0] = 1.01
    elif invalid == "nan":
        q[0, 0] = np.nan
    else:
        state0 = state0.T
    with pytest.raises(ValueError):
        algorithm(q, k, v, alpha, beta, initial_state=state0)


@pytest.mark.parametrize("chunk_size", [0, -1, 2.5, True])
def test_chunk_size_is_a_positive_integer(chunk_size):
    args = _data()
    with pytest.raises(ValueError, match="chunk_size"):
        kda_chunkwise(*args[:5], chunk_size=chunk_size)


def test_k3_gate_bounds_direction_and_finite_extremes():
    z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    alpha = k3_retention(z)
    assert alpha.dtype == np.float64
    assert np.all(np.isfinite(alpha))
    assert np.all((alpha >= np.exp(-5.0)) & (alpha <= 1.0))
    assert np.all(np.diff(alpha) <= 0)
    assert alpha[2] == np.exp(-2.5)
    extreme = k3_retention([-1.0, 0.0, 1.0], log_scale=1000)
    np.testing.assert_allclose(extreme, [1.0, np.exp(-2.5), np.exp(-5.0)])
