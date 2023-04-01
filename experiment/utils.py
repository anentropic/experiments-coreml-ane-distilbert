import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimingResult:
    execution_time_ns: int | None

@contextmanager
def timer():
    result = TimingResult(execution_time_ns=None)
    start_time = time.perf_counter_ns()
    yield result
    end_time = time.perf_counter_ns()
    result.execution_time_ns = end_time - start_time
