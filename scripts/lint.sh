#!/usr/bin/env bash
# scripts/lint.sh
# --------------------------------------------------------
# Run all code-quality checks using the UV build system.
# --------------------------------------------------------
source "$(dirname "$0")/lib/common.sh"
cd "$REPO_ROOT"

# --------------------------------------------------------
# Argument parsing
# --------------------------------------------------------
FIX=false
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)         FIX=true;    shift ;;
    --only)        ONLY="$2";   shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--fix] [--only ruff-format|ruff-check|pyright]"
      exit 0 ;;
    *) log_error "Unknown argument: $1"; exit 1 ;;
  esac
done

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
run_step() {
  local name="$1"; shift
  [[ -n "$ONLY" && "$ONLY" != "$name" ]] && return 0
  log_section "$name"
  "$@"
  log_success "$name passed"
}

# --------------------------------------------------------
# Auto-fixing
# --------------------------------------------------------
if $FIX; then
  log_info "Auto-fixing with ruff..."
  uv_run ruff format .
  uv_run ruff check --fix .
  log_success "Formatting and lint fixes applied"
fi

# --------------------------------------------------------
# Linting Checks
# --------------------------------------------------------
run_step "ruff-format"   uv_run ruff format --check .
run_step "ruff-check"    uv_run ruff check .
run_step "pyright"       uv_run pyright .

log_success "All lint checks passed"