# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for rollout stop-reason distribution and num-turn quantile
metrics."""

from types import SimpleNamespace

import pytest

from relax.utils.metrics.metric_utils import compute_num_turn_metrics, compute_stop_reason_metrics
from relax.utils.types import Sample


def _sample(status=Sample.Status.COMPLETED, metadata=None):
    return Sample(status=status, metadata=dict(metadata or {}))


class TestComputeNumTurnMetrics:
    def test_empty_samples_return_empty(self):
        assert compute_num_turn_metrics([]) == {}

    def test_single_sample_reports_identical_stats(self):
        metrics = compute_num_turn_metrics([_sample(metadata={"rollout_turns": 5})])

        assert set(metrics) == {
            "num_turn/mean",
            "num_turn/max",
            "num_turn/min",
            "num_turn/p50",
            "num_turn/p90",
            "num_turn/p95",
            "num_turn/p99",
        }
        assert all(value == 5 for value in metrics.values())

    def test_percentiles_use_linear_interpolation(self):
        metrics = compute_num_turn_metrics([_sample(metadata={"rollout_turns": turns}) for turns in range(1, 11)])

        assert metrics["num_turn/min"] == 1
        assert metrics["num_turn/max"] == 10
        assert metrics["num_turn/mean"] == pytest.approx(5.5)
        assert metrics["num_turn/p50"] == pytest.approx(5.5)
        assert metrics["num_turn/p90"] == pytest.approx(9.1)
        assert metrics["num_turn/p95"] == pytest.approx(9.55)
        assert metrics["num_turn/p99"] == pytest.approx(9.91)

    def test_missing_rollout_turns_defaults_to_one(self):
        metrics = compute_num_turn_metrics([_sample(metadata={})])

        assert all(value == 1 for value in metrics.values())

    def test_samples_missing_turns_keep_existing_default_semantics(self):
        samples = [
            _sample(metadata={}),
            _sample(metadata={"rollout_turns": 3}),
            _sample(metadata={"rollout_turns": 5}),
        ]

        metrics = compute_num_turn_metrics(samples)

        assert metrics["num_turn/min"] == 1
        assert metrics["num_turn/max"] == 5
        assert metrics["num_turn/mean"] == pytest.approx(3.0)
        assert metrics["num_turn/p50"] == pytest.approx(3.0)

    def test_float_rollout_turns_are_aggregated_as_is(self):
        samples = [
            _sample(metadata={"rollout_turns": 1}),
            _sample(metadata={"rollout_turns": 2.5}),
        ]

        metrics = compute_num_turn_metrics(samples)

        assert metrics["num_turn/min"] == 1
        assert metrics["num_turn/max"] == 2.5
        assert metrics["num_turn/mean"] == pytest.approx(1.75)
        assert metrics["num_turn/p50"] == pytest.approx(1.75)


class TestComputeStopReasonMetrics:
    def test_empty_samples_return_empty(self):
        assert compute_stop_reason_metrics([]) == {}

    def test_single_sample_falls_back_to_status(self):
        metrics = compute_stop_reason_metrics([_sample(status=Sample.Status.TRUNCATED)])

        assert metrics == {"stop_reason/truncated/count": 1, "stop_reason/truncated/frac": 1.0}

    def test_mixed_reasons_fractions_sum_to_one(self):
        samples = [
            _sample(metadata={"rollout_stop_reason": "env_done"}),
            _sample(metadata={"rollout_stop_reason": "env_done"}),
            _sample(status=Sample.Status.COMPLETED),
            _sample(status=Sample.Status.TRUNCATED),
            _sample(status=Sample.Status.ABORTED),
        ]

        metrics = compute_stop_reason_metrics(samples)

        assert metrics["stop_reason/env_done/count"] == 2
        assert metrics["stop_reason/env_done/frac"] == pytest.approx(0.4)
        assert metrics["stop_reason/completed/count"] == 1
        assert metrics["stop_reason/completed/frac"] == pytest.approx(0.2)
        assert metrics["stop_reason/truncated/frac"] == pytest.approx(0.2)
        assert metrics["stop_reason/aborted/frac"] == pytest.approx(0.2)

        fractions = [value for key, value in metrics.items() if key.endswith("/frac")]
        assert sum(fractions) == pytest.approx(1.0)

        assert list(metrics) == [
            f"stop_reason/{reason}/{suffix}"
            for reason in ("aborted", "completed", "env_done", "truncated")
            for suffix in ("count", "frac")
        ]

    def test_explicit_reason_takes_precedence_over_status(self):
        sample = _sample(status=Sample.Status.COMPLETED, metadata={"rollout_stop_reason": "env_done"})

        metrics = compute_stop_reason_metrics([sample])

        assert metrics == {"stop_reason/env_done/count": 1, "stop_reason/env_done/frac": 1.0}

    @pytest.mark.parametrize("reason", ["", None])
    def test_empty_explicit_reason_falls_back_to_status(self, reason):
        sample = _sample(status=Sample.Status.TRUNCATED, metadata={"rollout_stop_reason": reason})

        metrics = compute_stop_reason_metrics([sample])

        assert metrics == {"stop_reason/truncated/count": 1, "stop_reason/truncated/frac": 1.0}

    def test_explicit_reason_is_normalized(self):
        sample = _sample(status=Sample.Status.COMPLETED, metadata={"rollout_stop_reason": "  MAX_TURNS  "})

        metrics = compute_stop_reason_metrics([sample])

        assert metrics == {"stop_reason/max_turns/count": 1, "stop_reason/max_turns/frac": 1.0}

    def test_non_string_explicit_reason_falls_back_to_status(self):
        sample = _sample(status=Sample.Status.FAILED, metadata={"rollout_stop_reason": 42})

        metrics = compute_stop_reason_metrics([sample])

        assert metrics == {"stop_reason/failed/count": 1, "stop_reason/failed/frac": 1.0}

    def test_every_status_value_gets_its_own_bucket(self):
        samples = [Sample(status=status) for status in Sample.Status]

        metrics = compute_stop_reason_metrics(samples)

        assert set(metrics) == {
            f"stop_reason/{status.value}/{suffix}" for status in Sample.Status for suffix in ("count", "frac")
        }
        assert all(
            value == pytest.approx(1 / len(Sample.Status)) for key, value in metrics.items() if key.endswith("/frac")
        )

    def test_missing_status_counts_as_unknown(self):
        sample = _sample(status=None, metadata={})

        metrics = compute_stop_reason_metrics([sample])

        assert metrics == {"stop_reason/unknown/count": 1, "stop_reason/unknown/frac": 1.0}


def test_compute_metrics_from_samples_reports_stop_reason_and_turn_quantiles():
    pytest.importorskip("megatron.core")

    import relax.distributed.ray.rollout as rollout_module

    args = SimpleNamespace(
        log_reward_category=None,
        reward_key=None,
        log_passrate=False,
        advantage_estimator="grpo",
    )
    samples = [
        Sample(
            status=Sample.Status.COMPLETED,
            reward=1.0,
            response_length=4,
            metadata={"rollout_turns": 3, "rollout_stop_reason": "env_done"},
        ),
        Sample(
            status=Sample.Status.TRUNCATED,
            reward=0.0,
            response_length=8,
            metadata={"rollout_turns": 9},
        ),
    ]

    metrics = rollout_module.compute_metrics_from_samples(args, samples)

    assert metrics["num_turn/min"] == 3
    assert metrics["num_turn/max"] == 9
    assert metrics["num_turn/mean"] == pytest.approx(6.0)
    assert metrics["num_turn/p50"] == pytest.approx(6.0)
    assert metrics["num_turn/p90"] == pytest.approx(8.4)
    assert metrics["num_turn/p95"] == pytest.approx(8.7)
    assert metrics["num_turn/p99"] == pytest.approx(8.94)
    assert metrics["stop_reason/env_done/count"] == 1
    assert metrics["stop_reason/env_done/frac"] == pytest.approx(0.5)
    assert metrics["stop_reason/truncated/count"] == 1
    assert metrics["stop_reason/truncated/frac"] == pytest.approx(0.5)
