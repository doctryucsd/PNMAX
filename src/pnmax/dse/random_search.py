from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import tempfile
from collections.abc import Sequence as AbcSequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from tqdm import tqdm

from pnmax.analytical_model import AnalyticalModel
from pnmax.database import Arch, LoopBounds, Workload

DEFAULT_DIMENSION_ORDER: tuple[str, ...] = ("N", "K", "P", "Q", "C", "R", "S")
_TRACE_NAME_RE = re.compile(r"^trace_(\d+)\.yaml$")


@dataclass(frozen=True)
class RandomSearchConfig:
    out_dir: Path | str
    num_traces: int
    max_attempts: int = 10_000
    num_workers: int = 1
    dimensions: Sequence[str] | None = None
    level_ids: Sequence[int] | None = None
    seed: int | None = None
    template_id: str | int | None = None
    machine_id: str | int | None = None
    batch_attempts: int = 128
    show_progress: bool = True
    use_early_checks: bool = True
    warmup_cache: bool = True
    search_pu_sharing: bool = False
    search_system_sharing: bool = False
    search_cache: bool = False
    output_dim_constraint: OutputDimensionFactorizationConstraint | None = None
    resume: bool = True
    resume_metadata: Mapping[str, Any] | None = None
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None


@dataclass(frozen=True)
class RandomSearchResult:
    output_dir: Path
    seed: int
    attempts: int
    accepted: int
    duplicates: int
    rejected: int
    generated_files: tuple[Path, ...]


@dataclass(frozen=True)
class OutputDimensionFactorizationConstraint:
    tensor_name: str = "Out"
    allowed_level_ids: tuple[int, ...] = (0, 1)
    force_other_levels_to_one: bool = True


@dataclass(frozen=True)
class _ResolvedOutputDimensionFactorizationConstraint:
    tensor_name: str
    constrained_dimensions: tuple[str, ...]
    allowed_levels: tuple[int, ...]
    allowed_level_positions: tuple[int, ...]
    forced_one_levels: tuple[int, ...]
    forced_one_level_positions: tuple[int, ...]


@dataclass(frozen=True)
class _ResolvedWorkloadContext:
    workload_index: int
    base_workload: Workload
    dimensions: tuple[str, ...]
    levels: tuple[int, ...]
    target_traces: int
    pragma_search: _ResolvedPragmaSearchContext
    output_constraint: _ResolvedOutputDimensionFactorizationConstraint | None


@dataclass(frozen=True)
class _AcceptedCandidate:
    workload_attempt_index: int
    workload_index: int
    dedup_key: str
    workload_payload: Mapping[str, Any]


@dataclass(frozen=True)
class _SampledAttemptDescriptor:
    workload_attempt_index: int
    workload_index: int
    dedup_key: str
    factors_by_dim: Mapping[str, tuple[int, ...]]
    sharing_pu: bool | None = None
    sharing_system: int | None = None
    cache_enabled: bool | None = None


@dataclass(frozen=True)
class _ResolvedPragmaSearchContext:
    highest_level: int
    base_sharing_pu: bool | None
    base_sharing_system: int | None
    base_cache_enabled: bool | None
    search_pu_sharing: bool
    search_system_sharing: bool
    search_cache: bool


@dataclass(frozen=True)
class _SampledPragmaState:
    sharing_pu: bool | None
    sharing_system: int | None
    cache_enabled: bool | None


@dataclass(frozen=True)
class _ScheduledChunk:
    submission_index: int
    workload_index: int
    start_workload_attempt_index: int
    attempted: int


@dataclass(frozen=True)
class _ChunkResult:
    submission_index: int
    workload_index: int
    start_workload_attempt_index: int
    attempted: int
    duplicates: int
    candidates: tuple[_AcceptedCandidate, ...]


@dataclass(frozen=True)
class _ResumeSnapshot:
    accepted_jsonl_path: Path
    state_path: Path
    summary_path: Path
    signature: Mapping[str, Any]
    existing_trace_count: int
    accepted_by_workload: tuple[int, ...]
    dedup_seen: frozenset[str]
    next_trace_index: int
    next_workload_attempt_indices: tuple[int, ...]


_WORKER_BASE_WORKLOADS: tuple[Workload, ...] = ()
_WORKER_ARCH: Arch | None = None
_WORKER_DIMENSIONS_BY_WORKLOAD: tuple[tuple[str, ...], ...] = ()
_WORKER_LEVELS_BY_WORKLOAD: tuple[tuple[int, ...], ...] = ()
_WORKER_PRAGMA_CONTEXTS_BY_WORKLOAD: tuple[_ResolvedPragmaSearchContext, ...] = ()
_WORKER_OUTPUT_CONSTRAINTS_BY_WORKLOAD: tuple[
    _ResolvedOutputDimensionFactorizationConstraint | None, ...
] = ()
_WORKER_USE_EARLY_CHECKS: bool = True


class _RoundRobinChunkScheduler:
    def __init__(
        self,
        *,
        targets_by_workload: Sequence[int],
        max_attempts: int,
        batch_size: int,
        initial_workload_attempts: Sequence[int] | None = None,
    ) -> None:
        self._active_workloads = [
            index for index, target in enumerate(targets_by_workload) if int(target) > 0
        ]
        if not self._active_workloads:
            raise ValueError("No workloads are eligible to generate traces.")

        if initial_workload_attempts is None:
            self._workload_attempts = [0] * len(targets_by_workload)
        else:
            seeded_attempts = [int(value) for value in initial_workload_attempts]
            if len(seeded_attempts) != len(targets_by_workload):
                raise ValueError(
                    "initial_workload_attempts must match targets_by_workload length."
                )
            self._workload_attempts = seeded_attempts
        self._max_attempts = max(0, int(max_attempts))
        self._batch_size = max(1, int(batch_size))
        self._cursor = 0
        self._attempts_scheduled = 0
        self._next_submission_index = 0

    def next_chunk(self) -> _ScheduledChunk | None:
        if not self._active_workloads or self._attempts_scheduled >= self._max_attempts:
            return None

        if self._cursor >= len(self._active_workloads):
            self._cursor = 0

        workload_index = self._active_workloads[self._cursor]
        remaining_attempts = self._max_attempts - self._attempts_scheduled
        attempted = min(self._batch_size, remaining_attempts)
        start_workload_attempt_index = self._workload_attempts[workload_index]
        self._workload_attempts[workload_index] += attempted
        submission_index = self._next_submission_index

        self._attempts_scheduled += attempted
        self._next_submission_index += 1
        if self._active_workloads:
            self._cursor = (self._cursor + 1) % len(self._active_workloads)

        return _ScheduledChunk(
            submission_index=submission_index,
            workload_index=workload_index,
            start_workload_attempt_index=start_workload_attempt_index,
            attempted=attempted,
        )

    def mark_workload_complete(self, workload_index: int) -> None:
        try:
            remove_index = self._active_workloads.index(int(workload_index))
        except ValueError:
            return

        self._active_workloads.pop(remove_index)
        if not self._active_workloads:
            self._cursor = 0
            return
        if remove_index < self._cursor:
            self._cursor -= 1
        if self._cursor >= len(self._active_workloads):
            self._cursor = 0

    @property
    def workload_attempts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._workload_attempts)


def derive_template_seed(
    arch_name: str,
    template_id: str | int,
    machine_id: str | int,
    *,
    base_seed: int = 0,
) -> int:
    """Return a deterministic seed for one (arch, template, machine) tuple."""
    return _stable_seed(
        "template",
        arch_name,
        template_id,
        machine_id,
        int(base_seed),
    )


def verify_constraints(
    arch: Arch,
    workload: Workload,
    materialized: Workload,
    *,
    model: AnalyticalModel | None = None,
) -> bool:
    """
    Validate one materialized workload instance with the analytical model.

    The `workload` argument is intentionally kept for compatibility with
    workflows that pass a `(arch, workload, materialized)` triple.
    """
    del workload
    try:
        analytical_model = (
            model if model is not None else AnalyticalModel(materialized, arch)
        )
        return analytical_model.verify_constraints()
    except Exception:
        return False


def random_search(
    base_workload: Workload,
    arch: Arch,
    *,
    out_dir: Path | str,
    num_traces: int,
    max_attempts: int,
    num_workers: int = 1,
    dimensions: Sequence[str] | None = None,
    level_ids: Sequence[int] | None = None,
    seed: int | None = None,
    template_id: str | int | None = None,
    machine_id: str | int | None = None,
    batch_attempts: int = 128,
    show_progress: bool = True,
    use_early_checks: bool = True,
    warmup_cache: bool = True,
    search_pu_sharing: bool = False,
    search_system_sharing: bool = False,
    search_cache: bool = False,
    output_dim_constraint: OutputDimensionFactorizationConstraint | None = None,
) -> RandomSearchResult:
    config = RandomSearchConfig(
        out_dir=out_dir,
        num_traces=num_traces,
        max_attempts=max_attempts,
        num_workers=num_workers,
        dimensions=dimensions,
        level_ids=level_ids,
        seed=seed,
        template_id=template_id,
        machine_id=machine_id,
        batch_attempts=batch_attempts,
        show_progress=show_progress,
        use_early_checks=use_early_checks,
        warmup_cache=warmup_cache,
        search_pu_sharing=search_pu_sharing,
        search_system_sharing=search_system_sharing,
        search_cache=search_cache,
        output_dim_constraint=output_dim_constraint,
    )
    return run_random_search(base_workload, arch, config)


def run_random_search(
    base_workload: Workload,
    arch: Arch,
    config: RandomSearchConfig,
) -> RandomSearchResult:
    return run_random_search_multi((base_workload,), arch, config)


def run_random_search_multi(
    workloads: Sequence[Workload],
    arch: Arch,
    config: RandomSearchConfig,
) -> RandomSearchResult:
    _validate_config(config)
    workload_items = tuple(workloads)
    if not workload_items:
        raise ValueError("At least one workload is required.")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contexts = _prepare_workload_contexts(workload_items, arch, config)
    run_seed = _resolve_run_seed(workload_items, arch, config)
    resume_snapshot = _prepare_resume_snapshot(
        contexts=contexts,
        arch=arch,
        out_dir=out_dir,
        seed=run_seed,
        config=config,
    )

    if config.warmup_cache:
        for context in contexts:
            _warm_factorization_cache(
                (
                    int(context.base_workload.problem.loop_vars[dim])
                    for dim in context.dimensions
                ),
                len(context.levels),
            )

    if resume_snapshot.existing_trace_count >= config.num_traces:
        _emit_progress_callback(
            config.progress_callback,
            attempts=0,
            legally_found=resume_snapshot.existing_trace_count,
            target_traces=config.num_traces,
        )
        result = RandomSearchResult(
            output_dir=out_dir,
            seed=run_seed,
            attempts=0,
            accepted=0,
            duplicates=0,
            rejected=0,
            generated_files=tuple(),
        )
        _write_search_state(
            path=resume_snapshot.state_path,
            signature=resume_snapshot.signature,
            seed=run_seed,
            attempts=result.attempts,
            accepted=result.accepted,
            duplicates=result.duplicates,
            rejected=result.rejected,
            total_saved=resume_snapshot.existing_trace_count,
            accepted_by_workload=resume_snapshot.accepted_by_workload,
            next_trace_index=resume_snapshot.next_trace_index,
            next_workload_attempt_indices=resume_snapshot.next_workload_attempt_indices,
        )
        _write_search_summary(
            path=resume_snapshot.summary_path,
            signature=resume_snapshot.signature,
            result=result,
            resume_enabled=config.resume,
            existing_trace_count=resume_snapshot.existing_trace_count,
            accepted_by_workload=resume_snapshot.accepted_by_workload,
            next_trace_index=resume_snapshot.next_trace_index,
            next_workload_attempt_indices=resume_snapshot.next_workload_attempt_indices,
            status="skipped",
        )
        return result

    if config.num_workers <= 1:
        return _run_single_process(
            contexts=contexts,
            arch=arch,
            out_dir=out_dir,
            seed=run_seed,
            max_attempts=config.max_attempts,
            num_traces=config.num_traces,
            show_progress=config.show_progress,
            use_early_checks=config.use_early_checks,
            batch_attempts=config.batch_attempts,
            resume_snapshot=resume_snapshot,
            resume_enabled=config.resume,
            progress_desc=_progress_description(config.resume_metadata),
            progress_callback=config.progress_callback,
        )

    return _run_multi_process(
        contexts=contexts,
        arch=arch,
        out_dir=out_dir,
        seed=run_seed,
        max_attempts=config.max_attempts,
        num_traces=config.num_traces,
        show_progress=config.show_progress,
        use_early_checks=config.use_early_checks,
        num_workers=config.num_workers,
        batch_attempts=config.batch_attempts,
        resume_snapshot=resume_snapshot,
        resume_enabled=config.resume,
        progress_desc=_progress_description(config.resume_metadata),
        progress_callback=config.progress_callback,
    )


def _run_single_process(
    *,
    contexts: tuple[_ResolvedWorkloadContext, ...],
    arch: Arch,
    out_dir: Path,
    seed: int,
    max_attempts: int,
    num_traces: int,
    show_progress: bool,
    use_early_checks: bool,
    batch_attempts: int,
    resume_snapshot: _ResumeSnapshot,
    resume_enabled: bool,
    progress_desc: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> RandomSearchResult:
    context_by_index = _context_map(contexts)
    targets_by_workload = tuple(context.target_traces for context in context_by_index)
    accepted_by_workload = list(resume_snapshot.accepted_by_workload)
    scheduler = _RoundRobinChunkScheduler(
        targets_by_workload=targets_by_workload,
        max_attempts=max_attempts,
        batch_size=batch_attempts,
        initial_workload_attempts=resume_snapshot.next_workload_attempt_indices,
    )
    completed_workload_attempts = list(resume_snapshot.next_workload_attempt_indices)

    attempts = 0
    accepted = 0
    duplicates = 0
    generated_files: list[Path] = []
    dedup_seen: set[str] = set(resume_snapshot.dedup_seen)
    next_file_idx = resume_snapshot.next_trace_index
    pbar = _make_progress_bar(show_progress, max_attempts, desc=progress_desc)
    _update_progress(
        pbar,
        0,
        attempts,
        resume_snapshot.existing_trace_count + accepted,
        num_traces,
        progress_callback=progress_callback,
    )

    try:
        while (resume_snapshot.existing_trace_count + accepted) < num_traces:
            chunk = scheduler.next_chunk()
            if chunk is None:
                break

            context = context_by_index[chunk.workload_index]
            if accepted_by_workload[chunk.workload_index] >= context.target_traces:
                scheduler.mark_workload_complete(chunk.workload_index)
                continue

            for workload_attempt_offset in range(chunk.attempted):
                attempts += 1
                _update_progress(
                    pbar,
                    1,
                    attempts,
                    resume_snapshot.existing_trace_count + accepted,
                    num_traces,
                    progress_callback=progress_callback,
                )
                workload_attempt_index = (
                    chunk.start_workload_attempt_index + workload_attempt_offset
                )
                descriptor = _sample_attempt_descriptor(
                    workload_index=chunk.workload_index,
                    base_workload=context.base_workload,
                    arch=arch,
                    dimensions=context.dimensions,
                    levels=context.levels,
                    pragma_search=context.pragma_search,
                    output_constraint=context.output_constraint,
                    seed=seed,
                    workload_attempt_index=workload_attempt_index,
                    use_early_checks=use_early_checks,
                )
                if descriptor is None:
                    continue
                if descriptor.dedup_key in dedup_seen:
                    duplicates += 1
                    continue
                dedup_seen.add(descriptor.dedup_key)

                candidate = _validate_attempt_descriptor(
                    descriptor=descriptor,
                    base_workload=context.base_workload,
                    arch=arch,
                    dimensions=context.dimensions,
                    levels=context.levels,
                )
                if candidate is None:
                    continue
                if accepted_by_workload[chunk.workload_index] >= context.target_traces:
                    scheduler.mark_workload_complete(chunk.workload_index)
                    continue

                output_path = out_dir / f"trace_{next_file_idx:06d}.yaml"
                _write_workload_payload(output_path, candidate.workload_payload)
                _persist_accepted_candidate(
                    accepted_jsonl_path=resume_snapshot.accepted_jsonl_path,
                    output_path=output_path,
                    trace_index=next_file_idx,
                    candidate=candidate,
                    workload_name=context.base_workload.name,
                )
                generated_files.append(output_path)
                next_file_idx += 1
                accepted_by_workload[chunk.workload_index] += 1
                accepted += 1
                _update_progress(
                    pbar,
                    0,
                    attempts,
                    resume_snapshot.existing_trace_count + accepted,
                    num_traces,
                    progress_callback=progress_callback,
                )

                if accepted_by_workload[chunk.workload_index] >= context.target_traces:
                    scheduler.mark_workload_complete(chunk.workload_index)
                if (resume_snapshot.existing_trace_count + accepted) >= num_traces:
                    break

            completed_workload_attempts[chunk.workload_index] = max(
                completed_workload_attempts[chunk.workload_index],
                chunk.start_workload_attempt_index + chunk.attempted,
            )
            _write_search_state(
                path=resume_snapshot.state_path,
                signature=resume_snapshot.signature,
                seed=seed,
                attempts=attempts,
                accepted=accepted,
                duplicates=duplicates,
                rejected=attempts - accepted - duplicates,
                total_saved=resume_snapshot.existing_trace_count + accepted,
                accepted_by_workload=tuple(accepted_by_workload),
                next_trace_index=next_file_idx,
                next_workload_attempt_indices=tuple(completed_workload_attempts),
            )
    finally:
        if pbar is not None:
            pbar.close()

    rejected = attempts - accepted - duplicates
    result = RandomSearchResult(
        output_dir=out_dir,
        seed=seed,
        attempts=attempts,
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
        generated_files=tuple(generated_files),
    )
    _write_search_summary(
        path=resume_snapshot.summary_path,
        signature=resume_snapshot.signature,
        result=result,
        resume_enabled=resume_enabled,
        existing_trace_count=resume_snapshot.existing_trace_count,
        accepted_by_workload=tuple(accepted_by_workload),
        next_trace_index=next_file_idx,
        next_workload_attempt_indices=tuple(completed_workload_attempts),
        status="ok",
    )
    return result


def _run_multi_process(
    *,
    contexts: tuple[_ResolvedWorkloadContext, ...],
    arch: Arch,
    out_dir: Path,
    seed: int,
    max_attempts: int,
    num_traces: int,
    show_progress: bool,
    use_early_checks: bool,
    num_workers: int,
    batch_attempts: int,
    resume_snapshot: _ResumeSnapshot,
    resume_enabled: bool,
    progress_desc: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> RandomSearchResult:
    context_by_index = _context_map(contexts)
    targets_by_workload = tuple(context.target_traces for context in context_by_index)
    accepted_by_workload = list(resume_snapshot.accepted_by_workload)
    scheduler = _RoundRobinChunkScheduler(
        targets_by_workload=targets_by_workload,
        max_attempts=max_attempts,
        batch_size=batch_attempts,
        initial_workload_attempts=resume_snapshot.next_workload_attempt_indices,
    )
    completed_workload_attempts = list(resume_snapshot.next_workload_attempt_indices)

    attempts = 0
    accepted = 0
    duplicates = 0
    generated_files: list[Path] = []
    dedup_seen: set[str] = set(resume_snapshot.dedup_seen)
    next_file_idx = resume_snapshot.next_trace_index
    pbar = _make_progress_bar(show_progress, max_attempts, desc=progress_desc)
    _update_progress(
        pbar,
        0,
        attempts,
        resume_snapshot.existing_trace_count + accepted,
        num_traces,
        progress_callback=progress_callback,
    )

    workers = max(1, int(num_workers))
    workload_payloads = tuple(
        context_by_index[idx].base_workload.to_dict()
        for idx in range(len(context_by_index))
    )
    dimensions_by_workload = tuple(
        context_by_index[idx].dimensions for idx in range(len(context_by_index))
    )
    levels_by_workload = tuple(
        context_by_index[idx].levels for idx in range(len(context_by_index))
    )
    pragma_contexts_by_workload = tuple(
        context_by_index[idx].pragma_search for idx in range(len(context_by_index))
    )
    output_constraints_by_workload = tuple(
        context_by_index[idx].output_constraint for idx in range(len(context_by_index))
    )

    ctx = mp.get_context("spawn")
    pending: dict[Any, _ScheduledChunk] = {}
    completed: dict[int, _ChunkResult] = {}
    next_emit = 0
    stop = False

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                workload_payloads,
                arch,
                dimensions_by_workload,
                levels_by_workload,
                pragma_contexts_by_workload,
                output_constraints_by_workload,
                use_early_checks,
            ),
        ) as executor:
            while len(pending) < workers:
                chunk = scheduler.next_chunk()
                if chunk is None:
                    break
                future = executor.submit(
                    _worker_sample_chunk,
                    chunk.workload_index,
                    chunk.start_workload_attempt_index,
                    chunk.attempted,
                    seed,
                )
                pending[future] = chunk

            while pending and not stop:
                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    scheduled_chunk = pending.pop(future)
                    worker_result = future.result()
                    completed[scheduled_chunk.submission_index] = _ChunkResult(
                        submission_index=scheduled_chunk.submission_index,
                        workload_index=worker_result.workload_index,
                        start_workload_attempt_index=(
                            worker_result.start_workload_attempt_index
                        ),
                        attempted=worker_result.attempted,
                        duplicates=worker_result.duplicates,
                        candidates=worker_result.candidates,
                    )

                while next_emit in completed and not stop:
                    chunk = completed.pop(next_emit)
                    attempts += chunk.attempted
                    duplicates += chunk.duplicates
                    _update_progress(
                        pbar,
                        chunk.attempted,
                        attempts,
                        resume_snapshot.existing_trace_count + accepted,
                        num_traces,
                        progress_callback=progress_callback,
                    )

                    for candidate in chunk.candidates:
                        workload_index = candidate.workload_index
                        if (
                            accepted_by_workload[workload_index]
                            >= targets_by_workload[workload_index]
                        ):
                            continue
                        if candidate.dedup_key in dedup_seen:
                            duplicates += 1
                            continue
                        dedup_seen.add(candidate.dedup_key)
                        output_path = out_dir / f"trace_{next_file_idx:06d}.yaml"
                        _write_workload_payload(output_path, candidate.workload_payload)
                        _persist_accepted_candidate(
                            accepted_jsonl_path=resume_snapshot.accepted_jsonl_path,
                            output_path=output_path,
                            trace_index=next_file_idx,
                            candidate=candidate,
                            workload_name=context_by_index[
                                workload_index
                            ].base_workload.name,
                        )
                        generated_files.append(output_path)
                        next_file_idx += 1
                        accepted_by_workload[workload_index] += 1
                        accepted += 1
                        _update_progress(
                            pbar,
                            0,
                            attempts,
                            resume_snapshot.existing_trace_count + accepted,
                            num_traces,
                            progress_callback=progress_callback,
                        )
                        if (
                            accepted_by_workload[workload_index]
                            >= targets_by_workload[workload_index]
                        ):
                            scheduler.mark_workload_complete(workload_index)
                        if (
                            resume_snapshot.existing_trace_count + accepted
                        ) >= num_traces:
                            stop = True
                            break

                    if (
                        accepted_by_workload[chunk.workload_index]
                        >= targets_by_workload[chunk.workload_index]
                    ):
                        scheduler.mark_workload_complete(chunk.workload_index)

                    completed_workload_attempts[chunk.workload_index] = max(
                        completed_workload_attempts[chunk.workload_index],
                        chunk.start_workload_attempt_index + chunk.attempted,
                    )
                    next_emit += 1
                    if attempts >= max_attempts:
                        stop = True

                    _write_search_state(
                        path=resume_snapshot.state_path,
                        signature=resume_snapshot.signature,
                        seed=seed,
                        attempts=attempts,
                        accepted=accepted,
                        duplicates=duplicates,
                        rejected=attempts - accepted - duplicates,
                        total_saved=resume_snapshot.existing_trace_count + accepted,
                        accepted_by_workload=tuple(accepted_by_workload),
                        next_trace_index=next_file_idx,
                        next_workload_attempt_indices=tuple(
                            completed_workload_attempts
                        ),
                    )

                    while not stop and len(pending) < workers:
                        scheduled_chunk = scheduler.next_chunk()
                        if scheduled_chunk is None:
                            break
                        future = executor.submit(
                            _worker_sample_chunk,
                            scheduled_chunk.workload_index,
                            scheduled_chunk.start_workload_attempt_index,
                            scheduled_chunk.attempted,
                            seed,
                        )
                        pending[future] = scheduled_chunk

            if stop:
                for future in pending:
                    future.cancel()
    finally:
        if pbar is not None:
            pbar.close()

    rejected = attempts - accepted - duplicates
    result = RandomSearchResult(
        output_dir=out_dir,
        seed=seed,
        attempts=attempts,
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
        generated_files=tuple(generated_files),
    )
    _write_search_summary(
        path=resume_snapshot.summary_path,
        signature=resume_snapshot.signature,
        result=result,
        resume_enabled=resume_enabled,
        existing_trace_count=resume_snapshot.existing_trace_count,
        accepted_by_workload=tuple(accepted_by_workload),
        next_trace_index=next_file_idx,
        next_workload_attempt_indices=tuple(completed_workload_attempts),
        status="ok",
    )
    return result


def _worker_init(
    workload_payloads: tuple[Mapping[str, Any], ...],
    arch: Arch,
    dimensions_by_workload: tuple[tuple[str, ...], ...],
    levels_by_workload: tuple[tuple[int, ...], ...],
    pragma_contexts_by_workload: tuple[_ResolvedPragmaSearchContext, ...],
    output_constraints_by_workload: tuple[
        _ResolvedOutputDimensionFactorizationConstraint | None, ...
    ],
    use_early_checks: bool,
) -> None:
    global _WORKER_BASE_WORKLOADS
    global _WORKER_ARCH
    global _WORKER_DIMENSIONS_BY_WORKLOAD
    global _WORKER_LEVELS_BY_WORKLOAD
    global _WORKER_PRAGMA_CONTEXTS_BY_WORKLOAD
    global _WORKER_OUTPUT_CONSTRAINTS_BY_WORKLOAD
    global _WORKER_USE_EARLY_CHECKS

    _WORKER_BASE_WORKLOADS = tuple(
        Workload.from_dict(payload, source=f"worker-init-{idx}")
        for idx, payload in enumerate(workload_payloads)
    )
    _WORKER_ARCH = arch
    _WORKER_DIMENSIONS_BY_WORKLOAD = dimensions_by_workload
    _WORKER_LEVELS_BY_WORKLOAD = levels_by_workload
    _WORKER_PRAGMA_CONTEXTS_BY_WORKLOAD = pragma_contexts_by_workload
    _WORKER_OUTPUT_CONSTRAINTS_BY_WORKLOAD = output_constraints_by_workload
    _WORKER_USE_EARLY_CHECKS = bool(use_early_checks)


def _worker_sample_chunk(
    workload_index: int,
    start_workload_attempt_index: int,
    attempted: int,
    seed: int,
) -> _ChunkResult:
    if not _WORKER_BASE_WORKLOADS or _WORKER_ARCH is None:
        raise RuntimeError("Worker context is not initialized.")

    accepted: list[_AcceptedCandidate] = []
    dedup_seen: set[str] = set()
    duplicates = 0

    base_workload = _WORKER_BASE_WORKLOADS[workload_index]
    dimensions = _WORKER_DIMENSIONS_BY_WORKLOAD[workload_index]
    levels = _WORKER_LEVELS_BY_WORKLOAD[workload_index]
    pragma_search = _WORKER_PRAGMA_CONTEXTS_BY_WORKLOAD[workload_index]
    output_constraint = _WORKER_OUTPUT_CONSTRAINTS_BY_WORKLOAD[workload_index]

    for attempt_offset in range(max(0, int(attempted))):
        workload_attempt_index = start_workload_attempt_index + attempt_offset
        descriptor = _sample_attempt_descriptor(
            workload_index=workload_index,
            base_workload=base_workload,
            arch=_WORKER_ARCH,
            dimensions=dimensions,
            levels=levels,
            pragma_search=pragma_search,
            output_constraint=output_constraint,
            seed=seed,
            workload_attempt_index=workload_attempt_index,
            use_early_checks=_WORKER_USE_EARLY_CHECKS,
        )
        if descriptor is None:
            continue
        if descriptor.dedup_key in dedup_seen:
            duplicates += 1
            continue
        dedup_seen.add(descriptor.dedup_key)

        candidate = _validate_attempt_descriptor(
            descriptor=descriptor,
            base_workload=base_workload,
            arch=_WORKER_ARCH,
            dimensions=dimensions,
            levels=levels,
        )
        if candidate is not None:
            accepted.append(candidate)

    return _ChunkResult(
        submission_index=-1,
        workload_index=workload_index,
        start_workload_attempt_index=start_workload_attempt_index,
        attempted=max(0, int(attempted)),
        duplicates=duplicates,
        candidates=tuple(accepted),
    )


def _sample_attempt_descriptor(
    *,
    workload_index: int,
    base_workload: Workload,
    arch: Arch,
    dimensions: tuple[str, ...],
    levels: tuple[int, ...],
    pragma_search: _ResolvedPragmaSearchContext,
    seed: int,
    workload_attempt_index: int,
    use_early_checks: bool,
    output_constraint: _ResolvedOutputDimensionFactorizationConstraint | None = None,
) -> _SampledAttemptDescriptor | None:
    attempt_seed = _stable_seed("attempt", seed, workload_index, workload_attempt_index)
    rng = random.Random(attempt_seed)

    factors_by_dim = _sample_factors_for_dimensions(
        base_workload=base_workload,
        arch=arch,
        dimensions=dimensions,
        levels=levels,
        output_constraint=output_constraint,
        rng=rng,
    )
    if factors_by_dim is None:
        return None

    sampled_pragma = _sample_pragma_state(pragma_search=pragma_search, rng=rng)
    dedup_key = _build_dedup_key(
        workload_name=base_workload.name,
        compute=base_workload.compute,
        dimensions=dimensions,
        levels=levels,
        factors_by_dim=factors_by_dim,
        sharing_pu=sampled_pragma.sharing_pu,
        sharing_system=sampled_pragma.sharing_system,
        cache_enabled=sampled_pragma.cache_enabled,
    )

    if use_early_checks and not _passes_early_schedule_checks(
        arch=arch,
        levels=levels,
        dimensions=dimensions,
        factors_by_dim=factors_by_dim,
    ):
        raise RuntimeError(
            "Constraint-aware random search generated a schedule that failed "
            "the early architecture checks."
        )

    return _SampledAttemptDescriptor(
        workload_attempt_index=workload_attempt_index,
        workload_index=workload_index,
        dedup_key=dedup_key,
        factors_by_dim=factors_by_dim,
        sharing_pu=sampled_pragma.sharing_pu,
        sharing_system=sampled_pragma.sharing_system,
        cache_enabled=sampled_pragma.cache_enabled,
    )


def _validate_attempt_descriptor(
    *,
    descriptor: _SampledAttemptDescriptor,
    base_workload: Workload,
    arch: Arch,
    dimensions: tuple[str, ...],
    levels: tuple[int, ...],
) -> _AcceptedCandidate | None:
    materialized = _materialize_workload(
        base_workload=base_workload,
        dimensions=dimensions,
        levels=levels,
        factors_by_dim=descriptor.factors_by_dim,
        sharing_pu=descriptor.sharing_pu,
        sharing_system=descriptor.sharing_system,
        cache_enabled=descriptor.cache_enabled,
    )
    model = AnalyticalModel(materialized, arch)
    if not verify_constraints(arch, base_workload, materialized, model=model):
        return None

    return _AcceptedCandidate(
        workload_attempt_index=descriptor.workload_attempt_index,
        workload_index=descriptor.workload_index,
        dedup_key=descriptor.dedup_key,
        workload_payload=materialized.to_dict(),
    )


def _sample_factors_for_dimensions(
    *,
    base_workload: Workload,
    arch: Arch,
    dimensions: tuple[str, ...],
    levels: tuple[int, ...],
    rng: random.Random,
    output_constraint: _ResolvedOutputDimensionFactorizationConstraint | None = None,
) -> dict[str, tuple[int, ...]] | None:
    remaining_bounds: dict[str, int] = {}
    for dim in dimensions:
        bound = int(base_workload.problem.loop_vars[dim])
        if bound <= 0:
            raise ValueError(f"Problem dimension '{dim}' must be > 0, got {bound}.")
        remaining_bounds[dim] = bound

    constrained_output_dims = (
        frozenset(output_constraint.constrained_dimensions)
        if output_constraint is not None
        else frozenset()
    )
    forced_one_positions = (
        frozenset(output_constraint.forced_one_level_positions)
        if output_constraint is not None
        else frozenset()
    )
    allowed_output_positions = (
        frozenset(output_constraint.allowed_level_positions)
        if output_constraint is not None
        else frozenset(range(len(levels)))
    )

    def _position_allowed(dim: str, position: int) -> bool:
        if dim not in constrained_output_dims:
            return True
        return (
            position in allowed_output_positions
            and position not in forced_one_positions
        )

    constrained_positions = _constrained_level_positions(levels, arch)
    constrained_position_ids = {position for position, _ in constrained_positions}
    sampled_by_dim = {dim: [1] * len(levels) for dim in dimensions}
    for level_position, cap in constrained_positions:
        active_dims = tuple(
            dim for dim in dimensions if _position_allowed(dim, level_position)
        )
        sampled_level = _sample_constrained_level_factors(
            remaining_bounds=remaining_bounds,
            dimensions=active_dims,
            cap=cap,
            rng=rng,
        )
        for dim in dimensions:
            factor = int(sampled_level.get(dim, 1))
            sampled_by_dim[dim][level_position] = factor
            remaining_bounds[dim] //= factor

    for dim in dimensions:
        unconstrained_positions = [
            position
            for position in range(len(levels))
            if position not in constrained_position_ids
            and _position_allowed(dim, position)
        ]
        if not unconstrained_positions:
            if remaining_bounds[dim] != 1:
                return None
            continue
        sampled_tail = _sample_factorization(
            remaining_bounds[dim],
            len(unconstrained_positions),
            rng,
        )
        for level_position, factor in zip(unconstrained_positions, sampled_tail):
            sampled_by_dim[dim][level_position] = int(factor)

    return {
        dim: tuple(
            int(sampled_by_dim[dim][level_position])
            for level_position in range(len(levels))
        )
        for dim in dimensions
    }


def _sample_factors_for_dimensions_legacy(
    *,
    base_workload: Workload,
    dimensions: tuple[str, ...],
    num_levels: int,
    rng: random.Random,
) -> dict[str, tuple[int, ...]]:
    sampled: dict[str, tuple[int, ...]] = {}
    for dim in dimensions:
        bound = int(base_workload.problem.loop_vars[dim])
        if bound <= 0:
            raise ValueError(f"Problem dimension '{dim}' must be > 0, got {bound}.")
        sampled[dim] = _sample_factorization(bound, num_levels, rng)
    return sampled


def _constrained_level_positions(
    levels: tuple[int, ...], arch: Arch
) -> tuple[tuple[int, int], ...]:
    constrained: list[tuple[int, int]] = []
    for level_position, level in enumerate(levels):
        cap = _level_cap(level, arch)
        if cap is None:
            continue
        constrained.append((level_position, cap))
    constrained.sort(key=lambda item: (item[1], levels[item[0]]))
    return tuple(constrained)


def _level_cap(level: int, arch: Arch) -> int | None:
    if int(level) == 0:
        return int(arch.specs.simd_lanes)
    if int(level) >= 2:
        capacity = arch.hierarchy.organization.get(int(level))
        if capacity is None:
            return None
        return int(capacity)
    return None


def _sample_constrained_level_factors(
    *,
    remaining_bounds: Mapping[str, int],
    dimensions: tuple[str, ...],
    cap: int,
    rng: random.Random,
) -> dict[str, int]:
    if cap <= 0:
        raise ValueError(f"Constrained level capacity must be > 0, got {cap}.")
    if not dimensions:
        return {}

    divisors_by_dim = {
        dim: tuple(
            divisor
            for divisor in _divisors(int(remaining_bounds[dim]))
            if divisor <= cap
        )
        for dim in dimensions
    }
    cache: dict[tuple[int, int], int] = {}

    def _count(dim_index: int, remaining_cap: int) -> int:
        key = (dim_index, remaining_cap)
        cached = cache.get(key)
        if cached is not None:
            return cached
        if dim_index >= len(dimensions):
            return 1

        total = 0
        current_dim = dimensions[dim_index]
        for divisor in divisors_by_dim[current_dim]:
            if divisor > remaining_cap:
                break
            total += _count(dim_index + 1, remaining_cap // divisor)
        cache[key] = total
        return total

    sampled: dict[str, int] = {}
    remaining_cap = int(cap)
    for dim_index, dim in enumerate(dimensions):
        weighted_divisors: list[tuple[int, int]] = []
        total_weight = 0
        for divisor in divisors_by_dim[dim]:
            if divisor > remaining_cap:
                break
            weight = _count(dim_index + 1, remaining_cap // divisor)
            if weight <= 0:
                continue
            weighted_divisors.append((divisor, weight))
            total_weight += weight

        if total_weight <= 0:
            raise RuntimeError(
                f"Unable to sample constrained level factors for cap={cap}."
            )

        pick = rng.randrange(total_weight)
        accumulator = 0
        chosen = weighted_divisors[-1][0]
        for divisor, weight in weighted_divisors:
            accumulator += weight
            if pick < accumulator:
                chosen = divisor
                break
        sampled[dim] = chosen
        remaining_cap //= chosen

    return sampled


def _materialize_workload(
    *,
    base_workload: Workload,
    dimensions: tuple[str, ...],
    levels: tuple[int, ...],
    factors_by_dim: Mapping[str, tuple[int, ...]],
    sharing_pu: bool | None = None,
    sharing_system: int | None = None,
    cache_enabled: bool | None = None,
) -> Workload:
    materialized = copy.deepcopy(base_workload)
    bounds_payload: dict[int, list[dict[str, int]]] = {}

    for level_pos, level in enumerate(levels):
        bounds_payload[level] = [
            {dim: int(factors_by_dim[dim][level_pos])} for dim in dimensions
        ]

    materialized.bounds = LoopBounds(bounds_payload)
    if materialized.layout is not None:
        if cache_enabled is not None:
            materialized.layout.cache = bool(cache_enabled)
            if not cache_enabled:
                materialized.layout.cache_groups = ()
        if materialized.layout.sharing is not None:
            if sharing_pu is not None:
                materialized.layout.sharing.pu = bool(sharing_pu)
            if sharing_system is not None:
                materialized.layout.sharing.system = int(sharing_system)
        materialized.layout.refresh()
    materialized.refresh()
    return materialized


def _passes_early_schedule_checks(
    *,
    arch: Arch,
    levels: tuple[int, ...],
    dimensions: tuple[str, ...],
    factors_by_dim: Mapping[str, tuple[int, ...]],
) -> bool:
    level_products = [1] * len(levels)
    for dim in dimensions:
        factors = factors_by_dim[dim]
        for idx, factor in enumerate(factors):
            level_products[idx] *= factor

    if level_products[0] > arch.specs.simd_lanes:
        return False

    organization = arch.hierarchy.organization
    for idx, level in enumerate(levels):
        if level < 2:
            continue
        capacity = organization.get(level)
        if capacity is None:
            return False
        if level_products[idx] > capacity:
            return False
    return True


def _build_dedup_key(
    *,
    workload_name: str,
    compute: str,
    dimensions: tuple[str, ...],
    levels: tuple[int, ...],
    factors_by_dim: Mapping[str, tuple[int, ...]],
    sharing_pu: bool | None = None,
    sharing_system: int | None = None,
    cache_enabled: bool | None = None,
) -> str:
    items: list[str] = [workload_name, compute]
    for level_idx, level in enumerate(levels):
        for dim in dimensions:
            items.append(f"{dim}{level}={factors_by_dim[dim][level_idx]}")
    items.append(f"sharing_pu={sharing_pu}")
    items.append(f"sharing_system={sharing_system}")
    items.append(f"cache={cache_enabled}")
    payload = "|".join(items)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@lru_cache(maxsize=None)
def _divisors(value: int) -> tuple[int, ...]:
    if value <= 0:
        raise ValueError(f"Factorization bound must be > 0, got {value}.")
    lower: list[int] = []
    upper: list[int] = []
    root = int(math.isqrt(value))
    for candidate in range(1, root + 1):
        if value % candidate != 0:
            continue
        lower.append(candidate)
        quotient = value // candidate
        if quotient != candidate:
            upper.append(quotient)
    upper.reverse()
    return tuple(lower + upper)


@lru_cache(maxsize=None)
def _count_factorizations(bound: int, levels: int) -> int:
    if bound <= 0:
        return 0
    if levels <= 0:
        return 0
    if levels == 1:
        return 1
    total = 0
    for divisor in _divisors(bound):
        total += _count_factorizations(bound // divisor, levels - 1)
    return total


def _sample_factorization(
    bound: int, levels: int, rng: random.Random
) -> tuple[int, ...]:
    if levels <= 0:
        raise ValueError(f"levels must be > 0, got {levels}.")
    if bound <= 0:
        raise ValueError(f"bound must be > 0, got {bound}.")
    if levels == 1:
        return (bound,)

    divisor_weights: list[tuple[int, int]] = []
    total = 0
    for divisor in _divisors(bound):
        weight = _count_factorizations(bound // divisor, levels - 1)
        if weight <= 0:
            continue
        divisor_weights.append((divisor, weight))
        total += weight

    if total <= 0:
        raise RuntimeError(
            f"Cannot sample factorization for bound={bound}, levels={levels}."
        )

    pick = rng.randrange(total)
    acc = 0
    for divisor, weight in divisor_weights:
        acc += weight
        if pick < acc:
            tail = _sample_factorization(bound // divisor, levels - 1, rng)
            return (divisor, *tail)

    raise RuntimeError("Weighted factorization sampling failed unexpectedly.")


def _warm_factorization_cache(bounds: Sequence[int] | Any, num_levels: int) -> None:
    unique_bounds = {int(bound) for bound in bounds}
    for bound in unique_bounds:
        if bound <= 0:
            continue
        _divisors(bound)
        for levels in range(1, num_levels + 1):
            _count_factorizations(bound, levels)


def _resolve_dimensions(
    base_workload: Workload, dimensions: Sequence[str] | None
) -> tuple[str, ...]:
    available = {dim.upper() for dim in base_workload.problem.loop_vars}
    if not available:
        raise ValueError("Workload has no problem dimensions.")

    canonical_available = [d for d in DEFAULT_DIMENSION_ORDER if d in available]
    extra_available = sorted(d for d in available if d not in DEFAULT_DIMENSION_ORDER)
    default_order = canonical_available + extra_available

    if dimensions is None:
        return tuple(default_order)

    requested: list[str] = []
    seen: set[str] = set()
    for dim in dimensions:
        canonical = str(dim).strip().upper()
        if not canonical:
            continue
        if canonical not in available:
            raise ValueError(
                f"Unknown dimension '{dim}'. Available dimensions: {sorted(available)}"
            )
        if canonical in seen:
            continue
        seen.add(canonical)
        requested.append(canonical)

    remaining = [dim for dim in default_order if dim not in seen]
    resolved = tuple(requested + remaining)
    if not resolved:
        raise ValueError("Resolved dimension order is empty.")
    return resolved


def _resolve_levels(
    base_workload: Workload,
    arch: Arch,
    level_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    if level_ids is not None:
        levels = tuple(sorted({int(level) for level in level_ids}))
    elif arch.levels > 0:
        levels = tuple(range(arch.levels))
    else:
        bounds_levels = base_workload.bounds.levels()
        if not bounds_levels:
            raise ValueError(
                "Cannot infer tiling levels from architecture or workload."
            )
        levels = tuple(sorted(bounds_levels))

    if not levels:
        raise ValueError("Tiling levels cannot be empty.")
    if levels[0] != 0:
        raise ValueError(f"Tiling levels must start at 0, got {levels}.")
    if levels != tuple(range(levels[-1] + 1)):
        raise ValueError(
            f"Tiling levels must be contiguous [0..L]. Got {levels}. "
            "This keeps Workload bounds structurally valid."
        )
    if len(levels) < 2:
        raise ValueError(f"At least two levels are required (L0 and L1), got {levels}.")
    return levels


def _resolve_run_seed(
    workloads: Sequence[Workload], arch: Arch, config: RandomSearchConfig
) -> int:
    workload_items = tuple(workloads)
    if not workload_items:
        raise ValueError("At least one workload is required to derive a seed.")

    if config.template_id is not None or config.machine_id is not None:
        return derive_template_seed(
            arch.name,
            template_id=config.template_id if config.template_id is not None else "0",
            machine_id=config.machine_id if config.machine_id is not None else "0",
            base_seed=int(config.seed or 0),
        )
    if config.seed is not None:
        return int(config.seed)
    if len(workload_items) == 1:
        workload = workload_items[0]
        return _stable_seed("random-search", arch.name, workload.name, workload.compute)

    seed_parts: list[object] = ["random-search", arch.name]
    for workload in workload_items:
        seed_parts.extend((workload.name, workload.compute))
    return _stable_seed(*seed_parts)


def _prepare_workload_contexts(
    workloads: Sequence[Workload],
    arch: Arch,
    config: RandomSearchConfig,
) -> tuple[_ResolvedWorkloadContext, ...]:
    workload_items = tuple(workloads)
    targets = _allocate_trace_targets(config.num_traces, len(workload_items))
    contexts: list[_ResolvedWorkloadContext] = []

    for workload_index, workload in enumerate(workload_items):
        try:
            dimensions = _resolve_dimensions(workload, config.dimensions)
            levels = _resolve_levels(workload, arch, config.level_ids)
            pragma_search = _resolve_pragma_search_context(
                workload=workload,
                levels=levels,
                config=config,
            )
            output_constraint = _resolve_output_dim_constraint(
                workload=workload,
                dimensions=dimensions,
                levels=levels,
                constraint=config.output_dim_constraint,
            )
        except ValueError as exc:
            raise ValueError(
                f"Failed to configure workload[{workload_index}] '{workload.name}': {exc}"
            ) from exc

        contexts.append(
            _ResolvedWorkloadContext(
                workload_index=workload_index,
                base_workload=workload,
                dimensions=dimensions,
                levels=levels,
                target_traces=targets[workload_index],
                pragma_search=pragma_search,
                output_constraint=output_constraint,
            )
        )

    return tuple(contexts)


def _allocate_trace_targets(num_traces: int, num_workloads: int) -> tuple[int, ...]:
    if num_workloads <= 0:
        raise ValueError(f"num_workloads must be > 0, got {num_workloads}.")
    base, remainder = divmod(int(num_traces), int(num_workloads))
    return tuple(
        base + (1 if index < remainder else 0) for index in range(num_workloads)
    )


def _context_map(
    contexts: Sequence[_ResolvedWorkloadContext],
) -> tuple[_ResolvedWorkloadContext, ...]:
    if not contexts:
        raise ValueError("At least one workload context is required.")

    by_index = sorted(contexts, key=lambda context: context.workload_index)
    if tuple(ctx.workload_index for ctx in by_index) != tuple(range(len(by_index))):
        raise ValueError("Workload indices must be contiguous and start at 0.")
    return tuple(by_index)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _sample_pragma_state(
    *,
    pragma_search: _ResolvedPragmaSearchContext,
    rng: random.Random,
) -> _SampledPragmaState:
    if pragma_search.search_pu_sharing:
        sharing_pu: bool | None = bool(rng.randrange(2))
    else:
        sharing_pu = pragma_search.base_sharing_pu

    if pragma_search.search_system_sharing:
        sharing_system: int | None = (
            int(pragma_search.highest_level) if rng.randrange(2) else 0
        )
    else:
        sharing_system = pragma_search.base_sharing_system

    if pragma_search.search_cache:
        cache_enabled: bool | None = bool(rng.randrange(2))
    else:
        cache_enabled = pragma_search.base_cache_enabled

    return _SampledPragmaState(
        sharing_pu=sharing_pu,
        sharing_system=sharing_system,
        cache_enabled=cache_enabled,
    )


def _resolve_pragma_search_context(
    *,
    workload: Workload,
    levels: tuple[int, ...],
    config: RandomSearchConfig,
) -> _ResolvedPragmaSearchContext:
    layout = workload.layout
    sharing = layout.sharing if layout is not None else None
    search_pu_sharing = bool(config.search_pu_sharing)
    search_system_sharing = bool(config.search_system_sharing)
    search_cache = bool(config.search_cache)

    if (search_pu_sharing or search_system_sharing or search_cache) and layout is None:
        raise ValueError("pragma search requires Layout-Pragma to be defined.")
    if (search_pu_sharing or search_system_sharing) and sharing is None:
        raise ValueError(
            "sharing pragma search requires Layout-Pragma.sharing to be defined."
        )

    highest_level = int(max(levels))
    return _ResolvedPragmaSearchContext(
        highest_level=highest_level,
        base_sharing_pu=bool(sharing.pu) if sharing is not None else None,
        base_sharing_system=int(sharing.system) if sharing is not None else None,
        base_cache_enabled=bool(layout.cache) if layout is not None else None,
        search_pu_sharing=search_pu_sharing,
        search_system_sharing=search_system_sharing,
        search_cache=search_cache,
    )


def _resolve_output_dim_constraint(
    *,
    workload: Workload,
    dimensions: tuple[str, ...],
    levels: tuple[int, ...],
    constraint: OutputDimensionFactorizationConstraint | None,
) -> _ResolvedOutputDimensionFactorizationConstraint | None:
    if constraint is None:
        return None

    tensor_name = str(constraint.tensor_name).strip()
    if not tensor_name:
        raise ValueError("output_dim_constraint.tensor_name cannot be empty.")

    allowed_levels = tuple(
        sorted({int(level) for level in constraint.allowed_level_ids})
    )
    if not allowed_levels:
        raise ValueError(
            "output_dim_constraint.allowed_level_ids must contain at least one level."
        )

    unknown_levels = [level for level in allowed_levels if level not in levels]
    if unknown_levels:
        raise ValueError(
            f"output_dim_constraint.allowed_level_ids must be drawn from {levels}, "
            f"got {unknown_levels}."
        )

    allowed_level_positions = tuple(
        index for index, level in enumerate(levels) if level in allowed_levels
    )
    if not allowed_level_positions:
        raise ValueError(
            "output_dim_constraint.allowed_level_ids did not resolve to any "
            "materialized level positions."
        )

    tensor_dimensions = tuple(
        workload.get_tensor_level_bounds(tensor_name, level=levels[0]).keys()
    )
    if not tensor_dimensions:
        raise ValueError(f"Tensor '{tensor_name}' does not expose any loop dimensions.")

    missing_dimensions = [dim for dim in tensor_dimensions if dim not in dimensions]
    if missing_dimensions:
        raise ValueError(
            f"output_dim_constraint tensor '{tensor_name}' resolved dimensions "
            f"{missing_dimensions} that are absent from the active search dimensions "
            f"{dimensions}."
        )

    forced_one_levels = (
        tuple(level for level in levels if level not in allowed_levels)
        if constraint.force_other_levels_to_one
        else tuple()
    )
    forced_one_level_positions = tuple(
        index for index, level in enumerate(levels) if level in forced_one_levels
    )

    return _ResolvedOutputDimensionFactorizationConstraint(
        tensor_name=tensor_name,
        constrained_dimensions=tensor_dimensions,
        allowed_levels=allowed_levels,
        allowed_level_positions=allowed_level_positions,
        forced_one_levels=forced_one_levels,
        forced_one_level_positions=forced_one_level_positions,
    )


def _validate_config(config: RandomSearchConfig) -> None:
    if int(config.num_traces) <= 0:
        raise ValueError(f"num_traces must be > 0, got {config.num_traces}.")
    if int(config.max_attempts) <= 0:
        raise ValueError(f"max_attempts must be > 0, got {config.max_attempts}.")
    if int(config.num_workers) <= 0:
        raise ValueError(f"num_workers must be > 0, got {config.num_workers}.")
    if int(config.batch_attempts) <= 0:
        raise ValueError(f"batch_attempts must be > 0, got {config.batch_attempts}.")


def _prepare_resume_snapshot(
    *,
    contexts: Sequence[_ResolvedWorkloadContext],
    arch: Arch,
    out_dir: Path,
    seed: int,
    config: RandomSearchConfig,
) -> _ResumeSnapshot:
    signature = _build_resume_signature(
        contexts=contexts,
        arch=arch,
        seed=seed,
        resume_metadata=config.resume_metadata,
    )
    accepted_jsonl_path = out_dir / "accepted.jsonl"
    state_path = out_dir / "state.json"
    summary_path = out_dir / "summary.json"

    if not config.resume and _has_existing_search_artifacts(out_dir):
        raise ValueError(
            f"Output directory '{out_dir}' already contains search artifacts; "
            "remove them or rerun without --no-resume."
        )

    state_payload = _read_json_object(state_path)
    if state_payload is not None and _canonicalize_resume_signature(
        state_payload.get("signature")
    ) != _canonicalize_resume_signature(signature):
        raise ValueError(
            "Existing search output is incompatible with the current workload, "
            "architecture, or search-space configuration."
        )

    context_by_identity: dict[tuple[str, str], _ResolvedWorkloadContext] = {}
    for context in contexts:
        identity = (context.base_workload.name, context.base_workload.compute)
        if identity in context_by_identity:
            raise ValueError(
                "Resume requires workloads to have unique (name, compute) pairs."
            )
        context_by_identity[identity] = context

    trace_paths = _trace_paths(out_dir)
    accepted_by_workload = [0] * len(contexts)
    dedup_seen: set[str] = set()
    for trace_path in trace_paths:
        payload = _load_yaml_mapping(trace_path)
        workload = Workload.from_dict(payload, source=str(trace_path))
        identity = (workload.name, workload.compute)
        context = context_by_identity.get(identity)
        if context is None:
            raise ValueError(
                f"Existing trace '{trace_path}' does not match the current workloads."
            )
        accepted_by_workload[context.workload_index] += 1
        dedup_seen.add(_dedup_key_for_workload(workload, context))

    next_workload_attempt_indices = [0] * len(contexts)
    if state_payload is not None:
        raw_attempt_indices = state_payload.get("next_workload_attempt_indices")
        if isinstance(raw_attempt_indices, AbcSequence) and not isinstance(
            raw_attempt_indices, (str, bytes)
        ):
            parsed_indices = [int(value) for value in raw_attempt_indices]
            if len(parsed_indices) == len(contexts):
                next_workload_attempt_indices = parsed_indices

    for row in _read_jsonl_objects(accepted_jsonl_path):
        try:
            workload_index = int(row["workload_index"])
            workload_attempt_index = int(row["workload_attempt_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= workload_index < len(next_workload_attempt_indices):
            next_workload_attempt_indices[workload_index] = max(
                next_workload_attempt_indices[workload_index],
                workload_attempt_index + 1,
            )

    for workload_index, accepted_count in enumerate(accepted_by_workload):
        next_workload_attempt_indices[workload_index] = max(
            next_workload_attempt_indices[workload_index],
            int(accepted_count),
        )

    return _ResumeSnapshot(
        accepted_jsonl_path=accepted_jsonl_path,
        state_path=state_path,
        summary_path=summary_path,
        signature=signature,
        existing_trace_count=len(trace_paths),
        accepted_by_workload=tuple(int(count) for count in accepted_by_workload),
        dedup_seen=frozenset(dedup_seen),
        next_trace_index=_next_trace_index(out_dir),
        next_workload_attempt_indices=tuple(next_workload_attempt_indices),
    )


def _build_resume_signature(
    *,
    contexts: Sequence[_ResolvedWorkloadContext],
    arch: Arch,
    seed: int,
    resume_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    arch_payload = getattr(arch, "_raw", None)
    if isinstance(arch_payload, Mapping):
        arch_hash = _payload_hash(arch_payload)
    else:
        arch_hash = _payload_hash(
            {
                "name": arch.name,
                "defined_levels": list(arch.defined_levels),
                "simd_lanes": int(arch.specs.simd_lanes),
                "level_capacities": {
                    str(level): int(capacity)
                    for level, capacity in arch.level_capacities.items()
                },
            }
        )

    return {
        "version": 1,
        "seed": int(seed),
        "arch": {
            "name": arch.name,
            "payload_hash": arch_hash,
        },
        "workloads": [
            {
                "workload_index": int(context.workload_index),
                "name": context.base_workload.name,
                "compute": context.base_workload.compute,
                "payload_hash": _payload_hash(context.base_workload.to_dict()),
                "dimensions": list(context.dimensions),
                "levels": list(context.levels),
                "search_pu_sharing": bool(context.pragma_search.search_pu_sharing),
                "search_system_sharing": bool(
                    context.pragma_search.search_system_sharing
                ),
                "search_cache": bool(context.pragma_search.search_cache),
                "output_dim_constraint": _serialize_output_dim_constraint(
                    context.output_constraint
                ),
            }
            for context in contexts
        ],
        "resume_metadata": (
            _normalize_json_value(dict(resume_metadata))
            if resume_metadata is not None
            else None
        ),
    }


def _canonicalize_resume_signature(signature: Any) -> Any:
    normalized = _normalize_json_value(signature)
    if not isinstance(normalized, Mapping):
        return normalized
    payload = dict(normalized)
    payload.pop("resume_metadata", None)
    return payload


def _serialize_output_dim_constraint(
    constraint: (
        OutputDimensionFactorizationConstraint
        | _ResolvedOutputDimensionFactorizationConstraint
        | None
    ),
) -> dict[str, Any] | None:
    if constraint is None:
        return None

    payload: dict[str, Any] = {
        "tensor_name": str(constraint.tensor_name),
    }
    if isinstance(constraint, OutputDimensionFactorizationConstraint):
        payload["allowed_level_ids"] = [
            int(level) for level in constraint.allowed_level_ids
        ]
        payload["force_other_levels_to_one"] = bool(
            constraint.force_other_levels_to_one
        )
        return payload

    payload["allowed_level_ids"] = [int(level) for level in constraint.allowed_levels]
    payload["allowed_level_positions"] = [
        int(position) for position in constraint.allowed_level_positions
    ]
    payload["constrained_dimensions"] = [
        str(dim) for dim in constraint.constrained_dimensions
    ]
    payload["forced_one_level_ids"] = [
        int(level) for level in constraint.forced_one_levels
    ]
    payload["forced_one_level_positions"] = [
        int(position) for position in constraint.forced_one_level_positions
    ]
    payload["force_other_levels_to_one"] = bool(constraint.forced_one_levels)
    return payload


def _payload_hash(payload: Mapping[str, Any]) -> str:
    normalized = _normalize_json_value(dict(payload))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, AbcSequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_normalize_json_value(item) for item in value]
    return value


def _has_existing_search_artifacts(out_dir: Path) -> bool:
    if any(out_dir.glob("trace_*.yaml")):
        return True
    return any(
        (out_dir / filename).exists()
        for filename in ("accepted.jsonl", "state.json", "summary.json")
    )


def _trace_paths(out_dir: Path) -> tuple[Path, ...]:
    indexed_paths: list[tuple[int, Path]] = []
    for file_path in out_dir.glob("trace_*.yaml"):
        match = _TRACE_NAME_RE.match(file_path.name)
        if match is None:
            continue
        indexed_paths.append((int(match.group(1)), file_path))
    indexed_paths.sort(key=lambda item: item[0])
    return tuple(path for _, path in indexed_paths)


def _dedup_key_for_workload(
    workload: Workload, context: _ResolvedWorkloadContext
) -> str:
    factors_by_dim = {
        dim: tuple(workload.bounds.extent(dim, level=level) for level in context.levels)
        for dim in context.dimensions
    }
    sharing = workload.layout.sharing if workload.layout is not None else None
    return _build_dedup_key(
        workload_name=workload.name,
        compute=workload.compute,
        dimensions=context.dimensions,
        levels=context.levels,
        factors_by_dim=factors_by_dim,
        sharing_pu=sharing.pu if sharing is not None else None,
        sharing_system=sharing.system if sharing is not None else None,
        cache_enabled=workload.layout.cache if workload.layout is not None else None,
    )


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read YAML file '{path}'.") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Unable to parse YAML file '{path}'.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping-form YAML payload in '{path}'.")
    return payload


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload in '{path}'.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in '{path}'.")
    return payload


def _read_jsonl_objects(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return tuple()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL line {index} in '{path}'.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL payload at line {index} in '{path}'.")
        rows.append(payload)
    return tuple(rows)


def _persist_accepted_candidate(
    *,
    accepted_jsonl_path: Path,
    output_path: Path,
    trace_index: int,
    candidate: _AcceptedCandidate,
    workload_name: str,
) -> None:
    _append_jsonl(
        accepted_jsonl_path,
        {
            "trace_index": int(trace_index),
            "trace_file": output_path.name,
            "workload_index": int(candidate.workload_index),
            "workload_name": str(workload_name),
            "workload_attempt_index": int(candidate.workload_attempt_index),
            "dedup_key": candidate.dedup_key,
        },
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{line}\n")
        fh.flush()
        os.fsync(fh.fileno())


def _write_search_state(
    *,
    path: Path,
    signature: Mapping[str, Any],
    seed: int,
    attempts: int,
    accepted: int,
    duplicates: int,
    rejected: int,
    total_saved: int,
    accepted_by_workload: Sequence[int],
    next_trace_index: int,
    next_workload_attempt_indices: Sequence[int],
) -> None:
    _write_json_atomic(
        path,
        {
            "version": 1,
            "signature": dict(signature),
            "seed": int(seed),
            "attempts": int(attempts),
            "accepted": int(accepted),
            "duplicates": int(duplicates),
            "rejected": int(rejected),
            "total_saved": int(total_saved),
            "accepted_by_workload": [int(value) for value in accepted_by_workload],
            "next_trace_index": int(next_trace_index),
            "next_workload_attempt_indices": [
                int(value) for value in next_workload_attempt_indices
            ],
        },
    )


def _write_search_summary(
    *,
    path: Path,
    signature: Mapping[str, Any],
    result: RandomSearchResult,
    resume_enabled: bool,
    existing_trace_count: int,
    accepted_by_workload: Sequence[int],
    next_trace_index: int,
    next_workload_attempt_indices: Sequence[int],
    status: str,
) -> None:
    _write_json_atomic(
        path,
        {
            "version": 1,
            "status": str(status),
            "output_dir": str(result.output_dir),
            "seed": int(result.seed),
            "resume_enabled": bool(resume_enabled),
            "existing_trace_count": int(existing_trace_count),
            "attempts": int(result.attempts),
            "accepted": int(result.accepted),
            "duplicates": int(result.duplicates),
            "rejected": int(result.rejected),
            "saved_total": int(existing_trace_count + result.accepted),
            "accepted_by_workload": [int(value) for value in accepted_by_workload],
            "generated_files": [str(path) for path in result.generated_files],
            "next_trace_index": int(next_trace_index),
            "next_workload_attempt_indices": [
                int(value) for value in next_workload_attempt_indices
            ],
            "signature": dict(signature),
        },
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_normalize_json_value(dict(payload)), indent=2, sort_keys=True)
    _atomic_write_text(path, f"{text}\n")


def _next_trace_index(out_dir: Path) -> int:
    max_idx = -1
    for file_path in out_dir.glob("trace_*.yaml"):
        match = _TRACE_NAME_RE.match(file_path.name)
        if not match:
            continue
        max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _write_workload_payload(path: Path, workload_payload: Mapping[str, Any]) -> None:
    yaml_text = yaml.safe_dump(
        dict(workload_payload),
        sort_keys=False,
        default_flow_style=False,
    )
    _atomic_write_text(path, yaml_text)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
            temp_path = Path(fh.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _progress_description(resume_metadata: Mapping[str, Any] | None) -> str:
    if resume_metadata is None:
        return "Random search"
    progress_label = str(resume_metadata.get("progress_label", "")).strip()
    if progress_label:
        return progress_label
    space_id = str(resume_metadata.get("space_id", "")).strip()
    if space_id:
        return f"space {space_id}"
    return "Random search"


def _make_progress_bar(show_progress: bool, total: int, *, desc: str) -> tqdm | None:
    if not show_progress:
        return None
    return tqdm(total=total, desc=desc, unit="attempt")


def _update_progress(
    pbar: tqdm | None,
    delta_attempts: int,
    attempts: int,
    legally_found: int,
    target_traces: int,
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    _emit_progress_callback(
        progress_callback,
        attempts=attempts,
        legally_found=legally_found,
        target_traces=target_traces,
    )
    if pbar is None:
        return
    pbar.update(delta_attempts)
    pbar.set_postfix_str(
        f"attempted={attempts}/{int(pbar.total or 0)} legally_found={legally_found}/{target_traces}"
    )


def _emit_progress_callback(
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    attempts: int,
    legally_found: int,
    target_traces: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "attempts": int(attempts),
            "legally_found": int(legally_found),
            "target_traces": int(target_traces),
        }
    )
