#!/usr/bin/env bash
# scripts/test.sh
# --------------------------------------------------------
# Run tests using UV.
# --------------------------------------------------------
source "$(dirname "$0")/lib/common.sh"
cd "$REPO_ROOT"

# --------------------------------------------------------
# Argument parsing
# --------------------------------------------------------
COVERAGE=false
PYTEST_EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coverage)  COVERAGE=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--coverage] [-- <pytest-args>]"
      exit 0 ;;
    --) shift; PYTEST_EXTRA=("$@"); break ;;
    *)  log_error "Unknown argument: $1"; exit 1 ;;
  esac
done

# --------------------------------------------------------
# Coverage flags
# --------------------------------------------------------
COV_FLAGS=()
if $COVERAGE; then
  COV_FLAGS=(
    --cov=src
    --cov-report=term-missing
  )
  rm -f .coverage
  mkdir -p reports
  log_info "Coverage tracking enabled."
fi

# --------------------------------------------------------
# Run tests
# --------------------------------------------------------
log_section "Running tests"

uv_run pytest tests \
  --tb=short \
  -v \
  -s \
  "${COV_FLAGS[@]}" \
  "${PYTEST_EXTRA[@]}"

log_success "All tests passed"

# --------------------------------------------------------
# Coverage Reports
# --------------------------------------------------------
if $COVERAGE; then
  log_section "Generating Coverage Reports"
  uv run coverage xml -o reports/coverage.xml
  uv run coverage html -d reports/coverage-html

  if [[ -f reports/coverage-html/index.html ]]; then
    log_info "Coverage report: reports/coverage-html/index.html"
  fi
fi