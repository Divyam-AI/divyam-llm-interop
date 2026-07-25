#!/usr/bin/env bash
# scripts/lib/common.sh
# --------------------------------------------------------
# Shared utilities sourced by every script in scripts/.
# --------------------------------------------------------
set -euo pipefail

# --------------------------------------------------------
# Colours
# --------------------------------------------------------
if [[ -t 1 ]]; then
  RED='\033[0;31m'; YELLOW='\033[0;33m'; GREEN='\033[0;32m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
  RED=''; YELLOW=''; GREEN=''; CYAN=''; BOLD=''; RESET=''
fi

# --------------------------------------------------------
# Logging
# --------------------------------------------------------
log_info()    { echo -e "${CYAN}[INFO]  $*${RESET}"; }
log_success() { echo -e "${GREEN}[OK]    $*${RESET}"; }
log_warn()    { echo -e "${YELLOW}[WARN]  $*${RESET}"; }
log_error()   { echo -e "${RED}[ERROR] $*${RESET}" >&2; }
log_section() { echo -e "\n${BOLD}---  $*  ---${RESET}\n"; }

# --------------------------------------------------------
# Repo root
# --------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --------------------------------------------------------
# Require a command to exist
# --------------------------------------------------------
require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" &>/dev/null; then
    log_error "'${cmd}' is not installed or not on PATH."
    exit 1
  fi
}

# --------------------------------------------------------
# Run inside the UV virtualenv
# Usage: uv_run ruff check .
# --------------------------------------------------------
uv_run() {
  require_cmd uv
  uv run "$@"
}

# --------------------------------------------------------
# Section timing
# --------------------------------------------------------
SECONDS=0
print_elapsed() {
  echo -e "\n${BOLD}Completed in ${SECONDS}s${RESET}"
}
trap print_elapsed EXIT