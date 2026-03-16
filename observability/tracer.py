# observability/tracer.py
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TraceEvent:
    stage: str
    duration_ms: float
    payload: Dict[str, Any] = field(default_factory=dict)


class PipelineTracer:
    def __init__(self):
        self.events: List[TraceEvent] = []

    def measure(self, stage: str):
        return _TraceTimer(self, stage)

    def add_event(self, stage: str, duration_ms: float, payload: Dict[str, Any] | None = None):
        self.events.append(
            TraceEvent(
                stage=stage,
                duration_ms=round(duration_ms, 2),
                payload=payload or {},
            )
        )

    def to_dict(self):
        return {
            "events": [
                {
                    "stage": e.stage,
                    "duration_ms": e.duration_ms,
                    "payload": e.payload,
                }
                for e in self.events
            ]
        }


class _TraceTimer:
    def __init__(self, tracer: PipelineTracer, stage: str):
        self.tracer = tracer
        self.stage = stage
        self.start = None
        self.payload = {}

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def set_payload(self, **kwargs):
        self.payload.update(kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        duration_ms = (end - self.start) * 1000
        self.tracer.add_event(self.stage, duration_ms, self.payload)