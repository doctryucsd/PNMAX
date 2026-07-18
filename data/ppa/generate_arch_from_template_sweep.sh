#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/archs/lowered/geometry_sweep}"
GENERATOR_SCRIPT="${GENERATOR_SCRIPT:-data/ppa/generate_arch_from_template.py}"
ARCHS=(hbm_pim upmem)
VARIANTS=()

if [[ -n "${ARCHS_OVERRIDE:-}" ]]; then
  read -r -a ARCHS <<< "${ARCHS_OVERRIDE}"
fi

if [[ -n "${VARIANTS_OVERRIDE:-}" ]]; then
  read -r -a VARIANTS <<< "${VARIANTS_OVERRIDE}"
fi

_resolve_path() {
  local candidate="$1"
  if [[ "${candidate}" = /* || "${candidate}" =~ ^[A-Za-z]:[\\/].* ]]; then
    printf '%s\n' "${candidate}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${candidate}"
  fi
}

_dedupe_items() {
  local -n source_items="$1"
  local -n out_items="$2"
  local item
  declare -A seen_items=()

  out_items=()
  for item in "${source_items[@]}"; do
    if [[ -z "${item}" ]]; then
      continue
    fi
    if [[ -n "${seen_items[${item}]+x}" ]]; then
      continue
    fi
    seen_items["${item}"]=1
    out_items+=("${item}")
  done
}

_validate_archs() {
  local arch
  for arch in "${ARCHS[@]}"; do
    case "${arch}" in
      hbm_pim|upmem)
        ;;
      *)
        echo "Unsupported architecture in ARCHS_OVERRIDE: ${arch}" >&2
        exit 1
        ;;
    esac
  done
}

_validate_variants() {
  local variant
  for variant in "${VARIANTS[@]}"; do
    case "${variant}" in
      1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)
        ;;
      *)
        echo "Unsupported variant id in VARIANTS_OVERRIDE: ${variant}" >&2
        exit 1
        ;;
    esac
  done
}

_default_v_mat() {
  local arch="$1"
  case "${arch}" in
    hbm_pim) printf '32\n' ;;
    upmem) printf '64\n' ;;
  esac
}

_default_buffer_size() {
  local arch="$1"
  case "${arch}" in
    hbm_pim) printf '512B\n' ;;
    upmem) printf '32KB\n' ;;
  esac
}

_variant_num_pu() {
  local variant="$1"
  case "${variant}" in
    2|3) printf '4\n' ;;
    4|5) printf '16\n' ;;
    *) printf '8\n' ;;
  esac
}

_variant_h_mat() {
  local variant="$1"
  case "${variant}" in
    3|7) printf '32\n' ;;
    5|6) printf '8\n' ;;
    *) printf '16\n' ;;
  esac
}

_variant_v_mat() {
  local arch="$1"
  local variant="$2"

  case "${variant}" in
    2|6)
      case "${arch}" in
        hbm_pim) printf '64\n' ;;
        upmem) printf '128\n' ;;
      esac
      ;;
    4|7)
      case "${arch}" in
        hbm_pim) printf '16\n' ;;
        upmem) printf '32\n' ;;
      esac
      ;;
    *)
      _default_v_mat "${arch}"
      ;;
  esac
}

_variant_burst_len() {
  local variant="$1"
  case "${variant}" in
    8) printf '2\n' ;;
    9) printf '8\n' ;;
    *) printf '4\n' ;;
  esac
}

_variant_buffer_size() {
  local arch="$1"
  local variant="$2"
  case "${variant}" in
    10)
      case "${arch}" in
        hbm_pim) printf '128B\n' ;;
        upmem) printf '8KB\n' ;;
      esac
      ;;
    11)
      case "${arch}" in
        hbm_pim) printf '256B\n' ;;
        upmem) printf '16KB\n' ;;
      esac
      ;;
    12)
      case "${arch}" in
        hbm_pim) printf '1KB\n' ;;
        upmem) printf '64KB\n' ;;
      esac
      ;;
    13)
      case "${arch}" in
        hbm_pim) printf '2KB\n' ;;
        upmem) printf '128KB\n' ;;
      esac
      ;;
    *)
      _default_buffer_size "${arch}"
      ;;
  esac
}

_variant_interconnect() {
  local variant="$1"
  case "${variant}" in
    14) printf 'intra-bg\n' ;;
    15) printf 'inter-bg\n' ;;
    *) printf 'host\n' ;;
  esac
}

_target_filename() {
  local arch="$1"
  local variant="$2"
  local num_pu="$3"
  local h_mat="$4"
  local v_mat="$5"
  local burst_len="$6"
  local buffer_size="$7"
  local interconnect="$8"
  local stem="${arch}__pu-${num_pu}__hmat-${h_mat}__vmat-${v_mat}"

  if [[ "${burst_len}" != "4" ]]; then
    stem+="_burst-${burst_len}"
  fi
  if [[ "${buffer_size}" != "$(_default_buffer_size "${arch}")" ]]; then
    stem+="_buffer-${buffer_size}"
  fi
  if [[ "${interconnect}" != "host" ]]; then
    stem+="_interconnect-${interconnect}"
  fi
  printf '%s.yaml\n' "${stem}"
}

_generator_args_for_variant() {
  local arch="$1"
  local variant="$2"
  local burst_len="$3"
  local buffer_size="$4"
  local h_mat="$5"
  local v_mat="$6"
  local num_pu="$7"
  local interconnect="$8"

  if [[ "${burst_len}" != "4" ]]; then
    printf '%s\0%s\0' "--burst-len" "${burst_len}"
  fi
  if [[ "${buffer_size}" != "$(_default_buffer_size "${arch}")" ]]; then
    printf '%s\0%s\0' "--buffer-size" "${buffer_size}"
  fi
  if [[ "${h_mat}" != "16" ]]; then
    printf '%s\0%s\0' "--h-mat" "${h_mat}"
  fi
  if [[ "${v_mat}" != "$(_default_v_mat "${arch}")" ]]; then
    printf '%s\0%s\0' "--v-mat" "${v_mat}"
  fi
  if [[ "${num_pu}" != "8" ]]; then
    printf '%s\0%s\0' "--num-pu" "${num_pu}"
  fi
  if [[ "${interconnect}" != "host" ]]; then
    printf '%s\0%s\0' "--interconnect" "${interconnect}"
  fi
}

_run_one_variant() {
  local arch="$1"
  local variant="$2"
  local index="$3"
  local total="$4"
  local num_pu
  local h_mat
  local v_mat
  local burst_len
  local buffer_size
  local interconnect
  local template_path
  local output_path
  local filename
  local -a cmd=()
  local -a extra_args=()
  local status=0

  num_pu="$(_variant_num_pu "${variant}")"
  h_mat="$(_variant_h_mat "${variant}")"
  v_mat="$(_variant_v_mat "${arch}" "${variant}")"
  burst_len="$(_variant_burst_len "${variant}")"
  buffer_size="$(_variant_buffer_size "${arch}" "${variant}")"
  interconnect="$(_variant_interconnect "${variant}")"

  template_path="$(_resolve_path "data/archs/lowered/baseline/${arch}.yaml")"
  filename="$(_target_filename "${arch}" "${variant}" "${num_pu}" "${h_mat}" "${v_mat}" "${burst_len}" "${buffer_size}" "${interconnect}")"
  output_path="$(_resolve_path "${OUTPUT_ROOT}")/${arch}/${filename}"

  mapfile -d '' -t extra_args < <(
    _generator_args_for_variant "${arch}" "${variant}" "${burst_len}" "${buffer_size}" "${h_mat}" "${v_mat}" "${num_pu}" "${interconnect}"
  )

  cmd=(
    uv run python "$(_resolve_path "${GENERATOR_SCRIPT}")"
    --template "${template_path}"
    --output "${output_path}"
    "${extra_args[@]}"
  )

  echo "[start ${index}/${total}] arch=${arch} variant=${variant} target=${output_path}"
  set +e
  "${cmd[@]}"
  status=$?
  set -e

  if (( status == 0 )); then
    echo "[done] arch=${arch} variant=${variant} target=${output_path}"
    return 0
  fi

  echo "[fail exit=${status}] arch=${arch} variant=${variant} target=${output_path}" >&2
  failed_targets+=("${output_path}")
  failed_statuses+=("${status}")
  return 1
}

generator_abs="$(_resolve_path "${GENERATOR_SCRIPT}")"
output_root_abs="$(_resolve_path "${OUTPUT_ROOT}")"

if [[ ! -f "${generator_abs}" ]]; then
  echo "Generator script not found: ${generator_abs}" >&2
  exit 1
fi

deduped_archs=()
_dedupe_items ARCHS deduped_archs
ARCHS=("${deduped_archs[@]}")
_validate_archs

if (( ${#VARIANTS[@]} == 0 )); then
  VARIANTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
fi
deduped_variants=()
_dedupe_items VARIANTS deduped_variants
VARIANTS=("${deduped_variants[@]}")
_validate_variants

mkdir -p "${output_root_abs}"

total_jobs=$(( ${#ARCHS[@]} * ${#VARIANTS[@]} ))
success_count=0
failure_count=0
index=0
declare -a failed_targets=()
declare -a failed_statuses=()

echo "Generating ${total_jobs} architecture variants."
echo "Selected archs: ${ARCHS[*]}"
echo "Selected variants: ${VARIANTS[*]}"
echo "Generator script: ${generator_abs}"
echo "Output root: ${output_root_abs}"
echo "Overwrite policy: replace existing files in place"

for arch in "${ARCHS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    index=$(( index + 1 ))
    if _run_one_variant "${arch}" "${variant}" "${index}" "${total_jobs}"; then
      success_count=$(( success_count + 1 ))
    else
      failure_count=$(( failure_count + 1 ))
    fi
  done
done

echo
echo "Template-arch sweep summary"
echo "Total combinations: ${total_jobs}"
echo "Successful: ${success_count}"
echo "Failed: ${failure_count}"

if (( failure_count > 0 )); then
  echo "Failed targets:"
  for i in "${!failed_targets[@]}"; do
    echo "  ${failed_targets[${i}]} (exit=${failed_statuses[${i}]})"
  done
  exit 1
fi