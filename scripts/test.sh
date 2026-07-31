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
INTEGRATION=false
RUN_ALL=false
PYTEST_EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --coverage)      COVERAGE=true; shift ;;
    --integration)   INTEGRATION=true; shift ;;
    --all)           RUN_ALL=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--coverage] [--integration] [--all] [-- <pytest-args>]"
      echo ""
      echo "  --coverage      Enable coverage tracking"
      echo "  --integration   Run only integration tests (requires API keys)"
      echo "  --all           Run all tests (unit + integration) with combined coverage"
      echo "  -- <args>       Pass extra arguments to pytest"
      exit 0 ;;
    --) shift; PYTEST_EXTRA=("$@"); break ;;
    *)  log_error "Unknown argument: $1"; exit 1 ;;
  esac
done
# --------------------------------------------------------
# --all implies coverage
# --------------------------------------------------------
if $RUN_ALL; then
  COVERAGE=true
fi
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
if $RUN_ALL; then
  log_section "Running unit tests"
  uv_run pytest tests \
    --tb=short \
    -v \
    -s \
    --cov=src \
    --cov-report=term-missing \
    -m "not integration" \
    "${PYTEST_EXTRA[@]}"
  log_success "Unit tests passed"

  log_section "Running integration tests"
  uv_run pytest tests \
    --tb=short \
    -v \
    -s \
    --cov=src \
    --cov-report=term-missing \
    --cov-append \
    -m integration \
    "${PYTEST_EXTRA[@]}"
  log_success "Integration tests passed"
else
  MARKER_FLAGS=()
  if $INTEGRATION; then
    MARKER_FLAGS=(-m integration)
    log_info "Running integration tests only."
  else
    MARKER_FLAGS=(-m "not integration")
  fi

  log_section "Running tests"
  uv_run pytest tests \
    --tb=short \
    -v \
    -s \
    "${COV_FLAGS[@]}" \
    "${MARKER_FLAGS[@]}" \
    "${PYTEST_EXTRA[@]}"
  log_success "All tests passed"
fi
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


echo $OPENAI_API_KEY
