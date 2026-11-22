"""Prometheus metrics of the proxy (own registry, so several apps can live in one process)."""

from __future__ import annotations

from collections.abc import Sequence

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

__all__ = ["Metrics"]

_LAT = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)


class Metrics:
    def __init__(self, workers: Sequence[str], registry: CollectorRegistry | None = None) -> None:
        r = self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "pcrouter_requests", "Requests by worker and outcome", ["worker", "status"], registry=r
        )
        self.routed = Counter(
            "pcrouter_routed", "Routing decisions per worker", ["worker"], registry=r
        )
        self.retries = Counter("pcrouter_retries", "Retries after a connection failure", registry=r)
        self.rejected = Counter(
            "pcrouter_rejected", "Requests refused by admission control", registry=r
        )
        self.prompt_tokens = Counter(
            "pcrouter_prompt_tokens", "Prompt tokens routed", ["worker"], registry=r
        )
        self.predicted_cached = Counter(
            "pcrouter_predicted_cached_tokens",
            "Prompt tokens the index expected cached",
            ["worker"],
            registry=r,
        )
        self.actual_cached = Counter(
            "pcrouter_cached_tokens",
            "Prompt tokens the worker reported cached",
            ["worker"],
            registry=r,
        )
        self.decision = Histogram(
            "pcrouter_decision_seconds",
            "Time to choose a worker",
            buckets=(1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 1e-2),
            registry=r,
        )
        self.ttft = Histogram(
            "pcrouter_ttft_seconds", "Time to first streamed event", buckets=_LAT, registry=r
        )
        self.duration = Histogram(
            "pcrouter_request_seconds", "Request duration", buckets=_LAT, registry=r
        )
        self.inflight = Gauge("pcrouter_inflight", "In-flight requests", ["worker"], registry=r)
        self.pending = Gauge(
            "pcrouter_pending_prefill_tokens", "Queued prefill tokens", ["worker"], registry=r
        )
        self.index_chunks = Gauge(
            "pcrouter_index_chunks", "Chunks the index holds", ["worker"], registry=r
        )
        self.healthy = Gauge(
            "pcrouter_worker_healthy", "1 if the worker is in rotation", ["worker"], registry=r
        )
        for w in workers:
            self.healthy.labels(w).set(1)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
