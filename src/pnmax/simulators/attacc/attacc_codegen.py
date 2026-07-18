from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from pnmax.database import TensorAttrSpec, Workload

__all__ = ["TraceParameters", "AttaccCodegen", "AttaccCodegenError"]


class AttaccCodegenError(ValueError):
    """Raised when a workload cannot be translated into AttAcc parameters."""


@dataclass(frozen=True)
class TraceParameters:
    """Container for the arguments expected by the AttAcc trace generator."""

    dhead: int
    nhead: int
    seqlen: int
    maxlen: int
    dbyte: int


class AttaccCodegen:
    """Translate :class:`Workload` objects into AttAcc simulator arguments."""

    _SEQ_KEYS: Sequence[str] = "K"
    _HEAD_DIM_KEYS: Sequence[str] = "C"
    _NUM_HEAD_KEYS: Sequence[str] = "N"
    _DEFAULT_MAXLEN: int = 4096

    def __init__(self, workload: Workload) -> None:
        self.workload = workload

    def derive_trace_params(
        self,
        *,
        override_sequence: int | None = None,
        override_head_dim: int | None = None,
        override_num_heads: int | None = None,
        override_dtype_bytes: int | None = None,
        override_maxlen: int | None = None,
    ) -> TraceParameters:
        """Return AttAcc trace generator parameters for *workload*."""

        seq_len = (
            self._validate_positive(override_sequence, "sequence length")
            if override_sequence
            else self._dimension_from_problem(self._SEQ_KEYS, "sequence length")
        )
        head_dim = (
            self._validate_positive(override_head_dim, "head dimension")
            if override_head_dim
            else self._dimension_from_problem(self._HEAD_DIM_KEYS, "head dimension")
        )
        num_heads = (
            self._validate_positive(override_num_heads, "number of heads")
            if override_num_heads
            else self._dimension_from_problem(
                self._NUM_HEAD_KEYS, "number of heads", allow_missing=True
            )
        )
        if num_heads is None:
            num_heads = 1

        dtype_bytes = (
            self._validate_positive(override_dtype_bytes, "dtype bytes")
            if override_dtype_bytes
            else self._infer_dtype_bytes()
        )

        max_len = override_maxlen or max(seq_len, self._DEFAULT_MAXLEN)
        max_len = self._validate_positive(max_len, "max length")
        if max_len < seq_len:
            raise AttaccCodegenError(
                f"Max length ({max_len}) must be ≥ sequence length ({seq_len})."
            )

        return TraceParameters(
            dhead=head_dim,
            nhead=num_heads,
            seqlen=seq_len,
            maxlen=max_len,
            dbyte=dtype_bytes,
        )

    def _dimension_from_problem(
        self, keys: Sequence[str], label: str, *, allow_missing: bool = False
    ) -> int | None:
        dims = self.workload.problem.dimensions
        for key in keys:
            value = dims.get(key)
            if value is not None:
                return self._validate_positive(int(value), label)
        if allow_missing:
            return None
        formatted = ", ".join(keys)
        raise AttaccCodegenError(
            f"Cannot infer {label}; workload problem definition is missing "
            f"one of the following dimensions: {formatted}."
        )

    def _infer_dtype_bytes(self) -> int:
        specs: Iterable[TensorAttrSpec] = self.workload.tensor_attrs.values()
        bit_widths = {
            spec.bits for spec in specs if spec.bits is not None and spec.bits > 0
        }
        if not bit_widths:
            return 2
        if len(bit_widths) > 1:
            raise AttaccCodegenError(
                "Tensor attributes specify multiple precisions; please override the "
                "dtype manually with --dtype-bytes."
            )
        bits = bit_widths.pop()
        if bits % 8:
            raise AttaccCodegenError(
                f"Tensor precision {bits} is not byte-aligned; cannot target AttAcc."
            )
        return bits // 8

    @staticmethod
    def _validate_positive(value: int | None, label: str) -> int:
        if value is None:
            raise AttaccCodegenError(f"{label} must be provided.")
        if value <= 0:
            raise AttaccCodegenError(f"{label} must be positive, got {value}.")
        return int(value)
