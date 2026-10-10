"""Deadline-aware time sharing between colocated prefill and decode.

The controller evaluates a short decode slice followed by prefill against
prefill followed by decode. It only delays admission: the ordinary scheduler
continues to own KV allocation, request limits, cancellation and completion.
"""

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SLORequest:
    age: float
    remaining: float


class PDSLOController:
    def __init__(self, slo_seconds: float):
        self.slo = slo_seconds
        self.prefill_samples = deque(maxlen=64)
        self.decode_seconds = {}
        self.output_lengths = deque(maxlen=512)
        self.defer_started = None
        self.counts = Counter()
        self.observed_seconds = Counter()
        self.decode_credit = 0.0
        self.target_decode_share = 0.0

    def observe_forward(
        self, *, prefill: bool, tokens: int, batch_size: int, seconds: float
    ):
        if not math.isfinite(seconds) or seconds <= 0 or batch_size <= 0:
            return
        self.observed_seconds["prefill" if prefill else "decode"] += seconds
        if prefill:
            self.prefill_samples.append((tokens, seconds))
            share = self.target_decode_share
            self.decode_credit = min(
                min(0.5, self.slo * 0.1),
                self.decode_credit + seconds * share / max(0.2, 1 - share),
            )
        else:
            self.decode_credit = max(0.0, self.decode_credit - seconds)
            previous = self.decode_seconds.get(batch_size, seconds)
            self.decode_seconds[batch_size] = 0.8 * previous + 0.2 * seconds

    def observe_completion(self, output_tokens: int, limit: int):
        if output_tokens > 0:
            self.output_lengths.append((limit, output_tokens))

    def remaining_tokens(self, generated: int, limit: int) -> float:
        upper = max(1, limit - generated)
        # Condition on having survived to this decode position. A plain mean
        # minus generated would incorrectly predict one token for long outputs.
        samples = [
            min(length, limit) - generated
            for cap, length in self.output_lengths
            if cap == limit and length > generated
        ]
        if len(samples) < 8:
            # OpenAI requests without max_tokens get a context-dependent cap.
            # Grouping only by that cap can leave every bucket permanently cold
            # even after hundreds of short, naturally completed responses.
            # Pool those observations, treating shorter capped outputs as
            # right-censored rather than as EOS (which would bias the estimate).
            return self._pooled_remaining_tokens(generated, limit)
        return min(upper, sum(samples) / len(samples))

    def _pooled_remaining_tokens(self, generated: int, limit: int) -> float:
        upper = max(1, limit - generated)
        observations = [
            (min(length, limit), length < cap)
            for cap, length in self.output_lengths
            if length > generated
        ]
        if len(observations) < 8:
            return float(upper)

        # Integrate the Kaplan-Meier survival curve conditional on having
        # reached `generated`. Truncate at the current request's real cap; if
        # all remaining observations are censored, retain a conservative tail
        # up to that cap instead of inventing an unobserved completion time.
        events = Counter(length for length, ended in observations if ended)
        counts = Counter(length for length, _ in observations)
        at_risk = len(observations)
        survival = 1.0
        previous = generated
        remaining = 0.0
        for length in sorted(counts):
            remaining += survival * (length - previous)
            survival *= 1 - events[length] / at_risk
            at_risk -= counts[length]
            previous = length
        remaining += survival * (limit - previous)
        return min(upper, max(1.0, remaining))

    def predict_decode(self, batch_size: int) -> float:
        batch_size = max(1, batch_size)
        if not self.decode_seconds:
            return 0.010 + 0.0003 * batch_size
        keys = sorted(self.decode_seconds)
        lower = max((k for k in keys if k <= batch_size), default=keys[0])
        upper = min((k for k in keys if k >= batch_size), default=keys[-1])
        if lower != upper:
            fraction = (batch_size - lower) / (upper - lower)
            return (
                self.decode_seconds[lower] * (1 - fraction)
                + self.decode_seconds[upper] * fraction
            )
        return self.decode_seconds[lower] * max(0.5, 1 + 0.03 * (batch_size - lower))

    def predict_prefill(self, tokens: int) -> float:
        if not self.prefill_samples:
            return 0.010 + tokens / 15000
        n = len(self.prefill_samples)
        sx = sum(x for x, _ in self.prefill_samples)
        sy = sum(y for _, y in self.prefill_samples)
        sxx = sum(x * x for x, _ in self.prefill_samples)
        sxy = sum(x * y for x, y in self.prefill_samples)
        denominator = n * sxx - sx * sx
        if denominator > 0:
            slope = max(0.0, (n * sxy - sx * sy) / denominator)
            intercept = max(0.002, (sy - slope * sx) / n)
            return intercept + slope * tokens
        return max(0.002, sy / n * (tokens + 128) / (sx / n + 128))

    def _loss(self, completion: float) -> float:
        # Mean completion time remains useful even after a deadline is missed.
        # The additional lateness cost protects the E2E tail without dropping
        # requests that are already late.
        late = max(0.0, completion - self.slo)
        near = max(0.0, completion - 0.8 * self.slo)
        return completion + 4 * late + 2 * near * near / self.slo

    def _decode_plan(self, requests, start=0.0, steps=math.inf, costs=None):
        """Integrate decode cost as completed requests leave the batch.

        Holding the initial batch size for every remaining token overstates
        the cost of admitting prefill, especially at CUDA graph size boundaries.
        """
        ordered = sorted(requests, key=lambda r: r.remaining)
        elapsed = start
        position = 0.0
        loss = 0.0
        for index, req in enumerate(ordered):
            next_position = min(steps, req.remaining)
            batch_size = len(ordered) - index
            cost = (
                costs[batch_size]
                if costs is not None
                else self.predict_decode(batch_size)
            )
            elapsed += (next_position - position) * cost
            position = next_position
            if req.remaining > steps:
                return (
                    loss,
                    elapsed,
                    [SLORequest(r.age, r.remaining - steps) for r in ordered[index:]],
                )
            loss += self._loss(req.age + elapsed)
        return loss, elapsed, []

    def should_decode(
        self,
        running: Sequence[SLORequest],
        waiting: Sequence[SLORequest],
        prefill_tokens: int,
        now: float,
    ) -> bool:
        if not running or not waiting:
            self.defer_started = None
            if not running:
                self.decode_credit = self.target_decode_share = 0.0
            return False
        n, m = len(running), len(waiting)
        costs = {size: self.predict_decode(size) for size in range(1, n + m + 1)}
        p = self.predict_prefill(prefill_tokens)
        d = costs[n]
        # Reserve GPU time for current decodes even when no output can finish
        # within one slice. A one-prefill lookahead misses repeated interruption.
        need = max(r.remaining * d / max(d, self.slo - r.age) for r in running)
        self.target_decode_share = min(0.8, max(0.0, need))
        # Re-evaluate every round. The bound prevents prediction errors from
        # starving admission, including under a continuous decode workload.
        max_slice = min(0.5, self.slo * 0.1)
        if self.defer_started is not None and now - self.defer_started >= max_slice:
            self.counts["bounded_prefill"] += 1
            self.decode_credit = 0.0
            self.defer_started = None
            return False

        if self.decode_credit >= 0.5 * d:
            if self.defer_started is None:
                self.defer_started = now
            self.counts["decode_credit"] += 1
            return True
        prefill_loss, _, _ = self._decode_plan(
            [*running, *waiting], start=p, costs=costs
        )
        remaining_slice = max_slice - (
            0 if self.defer_started is None else now - self.defer_started
        )
        # Time credit already reserves a full decode slice. Only add a tiny
        # completion opportunity here; another long lookahead would compound
        # admission delays and keep decode batches unnecessarily small.
        max_steps = max(0, min(2, int(remaining_slice / d)))
        # Finishing a request changes the future batch size. Evaluate those
        # boundaries as well as a few short slices; never block the CPU/GPU.
        horizons = {1, 4, 8, 16, 32, max_steps}
        horizons.update(math.ceil(r.remaining) for r in running)
        best_loss = prefill_loss
        best_steps = 0
        for steps in sorted(h for h in horizons if 0 < h <= max_steps):
            score, delay, survivors = self._decode_plan(
                running, steps=steps, costs=costs
            )
            rest, _, _ = self._decode_plan(
                [*survivors, *waiting], start=delay + p, costs=costs
            )
            score += rest
            if score < best_loss:
                best_loss, best_steps = score, steps

        # A small hysteresis avoids switching based on timing noise.
        defer = best_steps > 0 and best_loss < prefill_loss - 0.001 * (n + m)
        if defer:
            if self.defer_started is None:
                self.defer_started = now
            self.counts["decode"] += 1
        else:
            self.defer_started = None
            self.counts["prefill"] += 1
        return defer

    def stats(self):
        limit_counts = Counter(cap for cap, _ in self.output_lengths)
        return {
            "decisions": dict(self.counts),
            "observed_gpu_seconds": dict(self.observed_seconds),
            "target_decode_share": self.target_decode_share,
            "decode_credit_seconds": self.decode_credit,
            "prefill_seconds_4096": self.predict_prefill(4096),
            "prefill_samples": len(self.prefill_samples),
            "completion_samples": len(self.output_lengths),
            "completion_limit_buckets": len(limit_counts),
            "max_completion_limit_samples": max(limit_counts.values(), default=0),
            "decode_seconds": dict(self.decode_seconds),
        }
