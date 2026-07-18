"""CLI entry point for running AttAcc simulations from workload files."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from pnmax.paths import repo_root

PROJECT_ROOT = repo_root()
EXTERNAL_PATH = PROJECT_ROOT / "external"

if str(EXTERNAL_PATH) not in sys.path:
    sys.path.append(str(EXTERNAL_PATH))

ATTACC_ROOT = EXTERNAL_PATH / "attacc"
DEFAULT_RAMULATOR_DIR = ATTACC_ROOT / "ramulator2"
TRACE_GENERATOR_RELATIVE = Path("trace_gen") / "gen_trace_attacc_bank.py"
DEFAULT_TRACE_GENERATOR = DEFAULT_RAMULATOR_DIR / TRACE_GENERATOR_RELATIVE
DEFAULT_RAMULATOR_BINARIES = (Path("ramulator2"), Path("build") / "ramulator2")

from pnmax.database.workload import Workload


def _load_attacc_codegen():
    module_path = (
        PROJECT_ROOT / "src" / "pnmax" / "simulators" / "attacc" / "attacc_codegen.py"
    )
    spec = importlib.util.spec_from_file_location("attacc_codegen", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_attacc_codegen = _load_attacc_codegen()
TraceParameters = _attacc_codegen.TraceParameters
AttaccCodegen = _attacc_codegen.AttaccCodegen
AttaccCodegenError = _attacc_codegen.AttaccCodegenError

TCK_NS = 0.769  # HBM3 clock period used by AttAcc Ramulator


class AttaccSimulationError(RuntimeError):
    """Raised when invoking the AttAcc tooling fails."""


@dataclass(frozen=True)
class SimulationResult:
    trace: TraceParameters
    cycles: int
    mac: int
    softmax: int
    mvgb: int
    mvsb: int
    wrgb: int

    def latency_ns(self, tck_ns: float = TCK_NS) -> float:
        return float(self.cycles) * tck_ns

    def as_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "cycles": self.cycles,
            "latency_ns": self.latency_ns(),
            "mac": self.mac,
            "softmax": self.softmax,
            "move_to_gemv_buffer": self.mvgb,
            "move_to_softmax_buffer": self.mvsb,
            "write_to_gemv_buffer": self.wrgb,
            "trace_params": self.trace,
        }


def load_workload(path: Path) -> Workload:
    workload = Workload.from_file(path)
    workload.validate_attacc_layout()
    return workload


def attacc_sim(
    workload_file: Union[str, Path],
    *,
    ramulator_dir: Optional[Union[str, Path]] = None,
    generator_script: Optional[Union[str, Path]] = None,
    ramulator_bin: Optional[Union[str, Path]] = None,
    artifact_dir: Optional[Union[str, Path]] = None,
    nhead: Optional[int] = None,
    dhead: Optional[int] = None,
    seqlen: Optional[int] = None,
    maxlen: Optional[int] = None,
    dtype_bytes: Optional[int] = None,
    power_constraint: bool = True,
    verbose: bool = False,
) -> SimulationResult:
    workload_path = Path(workload_file).resolve()
    ram_dir = Path(ramulator_dir or DEFAULT_RAMULATOR_DIR).resolve()
    if not ram_dir.exists():
        raise FileNotFoundError(f"Ramulator directory not found: '{ram_dir}'.")
    generator = (
        Path(generator_script)
        if generator_script
        else (ram_dir / TRACE_GENERATOR_RELATIVE)
    ).resolve()
    ram_bin = (
        Path(ramulator_bin).resolve()
        if ramulator_bin
        else _resolve_ramulator_binary(ram_dir, DEFAULT_RAMULATOR_BINARIES)
    )

    if not generator.exists():
        raise FileNotFoundError(
            f"Trace generator script not found: '{generator}'. Did you clone the "
            "AttAcc simulator?"
        )

    workload = load_workload(workload_path)
    codegen = AttaccCodegen(workload)
    trace_params = codegen.derive_trace_params(
        override_sequence=seqlen,
        override_head_dim=dhead,
        override_num_heads=nhead,
        override_dtype_bytes=dtype_bytes,
        override_maxlen=maxlen,
    )

    trace_path, yaml_path, temp_dir = _prepare_artifact_paths(
        workload.name, artifact_dir
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _run_trace_generator(
            generator,
            trace_params,
            trace_path,
            verbose=verbose,
        )
        _write_ramulator_yaml(yaml_path, trace_path, power_constraint)
        ram_output = _run_ramulator(
            ram_bin,
            yaml_path,
            cwd=ram_dir,
            verbose=verbose,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    metrics = _parse_ramulator_output(ram_output)
    return SimulationResult(
        trace=trace_params,
        cycles=metrics.get("memory_system_cycles", 0),
        mac=metrics.get("total_num_pim_mac_all_bank_requests", 0),
        softmax=metrics.get("total_num_pim_softmax_requests", 0),
        mvgb=metrics.get("total_num_pim_move_to_gemv_buffer_requests", 0),
        mvsb=metrics.get("total_num_pim_move_to_softmax_buffer_requests", 0),
        wrgb=metrics.get("total_num_pim_write_to_gemv_buffer_requests", 0),
    )


def _run_trace_generator(
    generator: Path,
    params: TraceParameters,
    trace_path: Path,
    *,
    verbose: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(generator),
        "--dhead",
        str(params.dhead),
        "--nhead",
        str(params.nhead),
        "--seqlen",
        str(params.seqlen),
        "--maxlen",
        str(params.maxlen),
        "--dbyte",
        str(params.dbyte),
        "--output",
        str(trace_path),
    ]
    if verbose:
        print("Running trace generator:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _write_ramulator_yaml(
    yaml_path: Path,
    trace_path: Path,
    power_constraint: bool,
) -> None:
    timing = "HBM3_5.2Gbps" if power_constraint else "HBM3_5.2Gbps_NPC"
    payload = textwrap.dedent(f"""
        Frontend:
          impl: PIMLoadStoreTrace
          path: {trace_path}
          clock_ratio: 1

          Translation:
            impl: NoTranslation
            max_addr: 2147483648

        MemorySystem:
          impl: PIMDRAM
          clock_ratio: 1
          DRAM:
            impl: HBM3-PIM
            org:
              preset: HBM3_8Gb_2R
              channel: 16
            timing:
              preset: {timing}

          Controller:
            impl: HBM3-PIM
            Scheduler:
              impl: PIM
            RefreshManager:
              impl: AllBankHBM3
            plugins:

          AddrMapper:
            impl: HBM3-PIM
        """).lstrip()
    yaml_path.write_text(payload)


def _run_ramulator(
    ramulator_bin: Path,
    yaml_config: Path,
    *,
    cwd: Path,
    verbose: bool = False,
) -> str:
    cmd = [str(ramulator_bin), "-f", str(yaml_config)]
    if verbose:
        print("Running Ramulator:", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AttaccSimulationError(
            f"Ramulator execution failed with code {exc.returncode}:\n{exc.stdout}"
        ) from exc
    if verbose and result.stdout:
        print(result.stdout)
    return result.stdout


def _parse_ramulator_output(output: str) -> Dict[str, int]:
    metrics: Dict[str, int] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value.isdigit():
            continue
        metrics[key] = int(value)
    return metrics


def _prepare_artifact_paths(
    workload_name: str,
    artifact_dir: Optional[Union[str, Path]],
) -> tuple[Path, Path, Optional[Path]]:
    if artifact_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="attacc_sim_"))
        return temp_dir / "attacc_bank.trace", temp_dir / "attacc_bank.yaml", temp_dir

    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = _sanitize_name(workload_name)
    return (
        target_dir / f"{prefix}_attacc_bank.trace",
        target_dir / f"{prefix}_attacc_bank.yaml",
        None,
    )


def _sanitize_name(name: str) -> str:
    filtered = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    filtered = filtered.strip("_").lower()
    return filtered or "attacc"


def _resolve_ramulator_binary(ramulator_dir: Path, candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else ramulator_dir / candidate
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Ramulator binary not found. Please build AttAcc's Ramulator2 "
        f"(looked under '{ramulator_dir}')."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AttAcc Ramulator simulations from workload YAML files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("workload", type=Path, help="Path to a workload YAML file.")
    parser.add_argument(
        "--ramulator-dir",
        type=Path,
        default=DEFAULT_RAMULATOR_DIR,
        help="Location of AttAcc's ramulator2 tree.",
    )
    parser.add_argument(
        "--generator-script",
        type=Path,
        default=None,
        help="Optional override for gen_trace_attacc_bank.py.",
    )
    parser.add_argument(
        "--ramulator-bin",
        type=Path,
        default=None,
        help="Path to the ramulator2 executable to use.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory where intermediate trace/yaml files will be kept.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=None,
        help="Override the number of attention heads per HBM.",
    )
    parser.add_argument(
        "--dhead",
        type=int,
        default=None,
        help="Override the head dimension.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=None,
        help="Override the sequence length dimension.",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=None,
        help="Override the maximum sequence length for the trace generator.",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=None,
        help="Force a specific tensor precision in bytes.",
    )
    parser.add_argument(
        "--no-power-constraint",
        dest="power_constraint",
        action="store_false",
        help="Disable AttAcc's DRAM power constraint (NPC preset).",
    )
    parser.set_defaults(power_constraint=True)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print executed commands and Ramulator output.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = attacc_sim(
            args.workload,
            ramulator_dir=args.ramulator_dir,
            generator_script=args.generator_script,
            ramulator_bin=args.ramulator_bin,
            artifact_dir=args.artifact_dir,
            nhead=args.nhead,
            dhead=args.dhead,
            seqlen=args.seqlen,
            maxlen=args.maxlen,
            dtype_bytes=args.dtype_bytes,
            power_constraint=args.power_constraint,
            verbose=args.verbose,
        )
    except (
        AttaccSimulationError,
        AttaccCodegenError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        parser.error(str(exc))
        return

    print(
        f"Simulation complete. Cycles={result.cycles}, "
        f"MAC={result.mac}, Softmax={result.softmax}, "
        f"MVGB={result.mvgb}, MVSB={result.mvsb}, WRGB={result.wrgb}, "
        f"latency_ns={result.latency_ns():.2f}"
    )


if __name__ == "__main__":
    main()
