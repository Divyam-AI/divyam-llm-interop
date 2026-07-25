#!/usr/bin/env bash
# scripts/setup-dev.sh
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$REPO_ROOT"

# --------------------------------------------------------
# Parse flags
# --------------------------------------------------------
UPGRADE_MODE="false"

for arg in "$@"; do
  case "$arg" in
    --upgrade) UPGRADE_MODE="true" ;;
  esac
done

# --------------------------------------------------------
# UV Installation
# --------------------------------------------------------
log_section "UV Build System"
if ! command -v uv &>/dev/null; then
  log_info "UV not found - installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  log_success "UV installed"
else
  log_info "UV already installed: $(uv --version)"
fi

# --------------------------------------------------------
# Check prerequisites
# --------------------------------------------------------
log_section "Checking prerequisites"
require_cmd git

# uv handles Python — check it can find a compatible version
if PYTHON_PATH=$(uv python find ">=3.10,<4.0" 2>/dev/null); then
  log_info "Python: $("$PYTHON_PATH" --version) at ${PYTHON_PATH}"
else
  log_info "No Python >=3.10 found, installing..."
  uv python install ">=3.10,<4.0"
  PYTHON_PATH=$(uv python find ">=3.10,<4.0")
  log_success "Installed Python: $("$PYTHON_PATH" --version) at ${PYTHON_PATH}"
fi

# --------------------------------------------------------
# Install Dependencies
# --------------------------------------------------------
log_section "Installing dependencies"
uv sync --all-groups
log_success "Dependencies synced to .venv"

# --------------------------------------------------------
# Optional: Lock file upgrade phase
# --------------------------------------------------------
if [[ "${UPGRADE_MODE}" == "true" ]]; then
  log_section "Upgrading dependencies and regenerating lock file"
  uv lock --upgrade
  log_success "Lock file upgraded and resolved successfully"

  log_info "Re-syncing environment to apply the newly upgraded lockfile..."
  uv sync --all-groups
  log_success "Environment synced to latest upgrades"
fi

# --------------------------------------------------------
# Make all scripts executable
# --------------------------------------------------------
log_section "Script permissions"
chmod +x scripts/*.sh
log_success "All scripts are executable"

# --------------------------------------------------------
# Summary
# --------------------------------------------------------
echo ""
log_success "Dev environment ready!"
echo ""
echo -e "  ${BOLD}Common commands:${RESET}"
echo "  ./scripts/setup-dev.sh --upgrade  # refresh lockfile & upgrade packages"
echo "  ./scripts/lint.sh                 # run all linters"
echo "  ./scripts/lint.sh --fix           # auto-fix formatting"
echo "  ./scripts/test.sh                 # run tests"
echo "  ./scripts/test.sh --coverage      # run tests with coverage"
echo ""
echo -e "  ${BOLD}IDE Interpreter Path:${RESET}"
echo "  ${REPO_ROOT}/.venv/bin/python"
echo ""