#!/bin/bash
set -Eeuo pipefail

# Rover install & run script
# ==========================
#
# Usage:
#   curl -fsSL https://your-domain.com/rover.sh | bash
#   curl -fsSL https://your-domain.com/rover.sh | bash -s -- --since 2m
#   curl -fsSL https://your-domain.com/rover.sh | bash -s -- --project myrepo --output report.html
#
# What this does:
#   1. Check Docker is installed and running
#   2. Build or pull the Rover Docker image
#   3. Mount transcript dirs (Claude, Cursor, Codex) read-only
#   4. Run analysis inside the container
#   5. Save HTML report to your current directory
#
# Your code never leaves your machine. Only aggregate metadata + scores
# are processed inside the container.

ROVER_IMAGE="${ROVER_IMAGE:-rover:latest}"
ROVER_VERSION="0.1.0"

# --- Colors ---
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  _IS_TTY=1
else
  _IS_TTY=0
fi

_color() {
  if [ "${_IS_TTY:-0}" != "1" ]; then printf '%s' "$2"; return; fi
  local code
  case "$1" in
    yellow) code='33' ;;
    green)  code='32' ;;
    cyan)   code='36' ;;
    red)    code='31' ;;
    *)      printf '%s' "$2"; return ;;
  esac
  printf '\033[%sm%s\033[0m' "$code" "$2"
}

_bold() {
  if [ "${_IS_TTY:-0}" != "1" ]; then printf '%s' "$1"; return; fi
  printf '\033[1m%s\033[0m' "$1"
}

# --- Error handler ---
_rover_on_error() {
  local ec=$?
  local failed_line="${BASH_LINENO[0]:-?}"
  local failed_cmd="${BASH_COMMAND:-?}"
  {
    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "Rover hit an unexpected error."
    echo ""
    echo "  exit code: $ec"
    echo "  line:      $failed_line"
    echo "  command:   $failed_cmd"
    echo ""
    echo "Please report this issue so we can fix it."
    echo "────────────────────────────────────────────────────────"
  } &>2
}
trap _rover_on_error ERR

# --- Docker check ---
require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo ""
    echo "Error: Docker is not installed." &>2
    echo "" &>2
    echo "Rover runs inside a Docker container so your code stays local." &>2
    echo "Install Docker: https://docs.docker.com/get-docker/" &>2
    echo ""
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo ""
    echo "Error: Docker daemon is not running." &>2
    echo "Start Docker and try again." &>2
    echo ""
    exit 1
  fi
}

# --- Build image ---
build_image() {
  if docker image inspect "$ROVER_IMAGE" >/dev/null 2>&1; then
    echo "  Using existing image: $(_bold "$ROVER_IMAGE")"
    return 0
  fi

  # Try to pull first (if published), else build from local Dockerfile
  echo "  Pulling/building image: $(_bold "$ROVER_IMAGE")..."
  if docker pull "$ROVER_IMAGE" >/dev/null 2>&1; then
    echo "  Pulled from registry."
    return 0
  fi

  # Build from the Dockerfile in the same directory as this script
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "$SCRIPT_DIR/Dockerfile" ]; then
    docker build -t "$ROVER_IMAGE" "$SCRIPT_DIR" >/dev/null 2>&1
    echo "  Built locally from Dockerfile."
  else
    echo "Error: Could not find Docker image $ROVER_IMAGE or a local Dockerfile." &>2
    exit 1
  fi
}

# --- OS-aware paths ---
detect_cursor_paths() {
  case "$(uname -s)" in
    Darwin)
      CURSOR_DIR="${CURSOR_DIR:-$HOME/Library/Application Support/Cursor/User/workspaceStorage}"
      CURSOR_GLOBAL="${CURSOR_GLOBAL:-$HOME/Library/Application Support/Cursor/User/globalStorage/state.vscdb}"
      ;;
    MINGW*|MSYS*|CYGWIN*)
      _appdata="${APPDATA:-$HOME/AppData/Roaming}"
      _appdata="${_appdata//\\//}"
      CURSOR_DIR="${CURSOR_DIR:-$_appdata/Cursor/User/workspaceStorage}"
      CURSOR_GLOBAL="${CURSOR_GLOBAL:-$_appdata/Cursor/User/globalStorage/state.vscdb}"
      ;;
    *)
      CURSOR_DIR="${CURSOR_DIR:-$HOME/.config/Cursor/User/workspaceStorage}"
      CURSOR_GLOBAL="${CURSOR_GLOBAL:-$HOME/.config/Cursor/User/globalStorage/state.vscdb}"
      ;;
  esac
}

# --- Compose docker run args ---
compose_mounts() {
  local mounts=()
  local cwd
  cwd="$(pwd)"

  # Claude Code
  if [ -d "$HOME/.claude/projects" ]; then
    mounts+=("-v" "$HOME/.claude/projects:/claude/projects:ro")
  fi

  # Codex CLI
  if [ -d "$HOME/.codex/sessions" ]; then
    mounts+=("-v" "$HOME/.codex/sessions:/codex/sessions:ro")
  fi

  # Cursor
  detect_cursor_paths
  if [ -d "$CURSOR_DIR" ]; then
    mounts+=("-v" "$CURSOR_DIR:/cursor/workspaceStorage:ro")
  fi
  if [ -f "$CURSOR_GLOBAL" ]; then
    mounts+=("-v" "$CURSOR_GLOBAL:/cursor/globalStorage/state.vscdb:ro")
  fi

  # Current repo for git metrics (read-only)
  mounts+=("-v" "$cwd:/workspace:ro")
  # Writable output directory
  mounts+=("-v" "$cwd:/reports")

  printf '%s\n' "${mounts[@]}"
}

# --- Open report ---
open_report() {
  local path="$1"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$path" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then
    open "$path" >/dev/null 2>&1 &
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m webbrowser "$path" >/dev/null 2>&1 &
  fi
}

# --- Help ---
show_help() {
  cat <<'EOF'
Usage: rover.sh [OPTIONS]

Options:
  --since <duration>    Analyze sessions since duration (default: 2m)
                        Formats: 6h, 7d, 2w, 1m, or YYYY-MM-DD
  --project <name>      Select project by repo name
  --output <path>       Output HTML report path (default: rover-report.html)
  --no-repo             Skip git metrics (transcripts only)
  --no-llm              Skip LLM analysis (rule-based only)
  --all                 Analyze every project found
  --open                Open report in browser after generation
  -h, --help            Show this help message

Environment:
  ROVER_IMAGE           Docker image to use (default: rover:latest)
  ROVER_LLM_API_KEY     OpenAI-compatible API key (optional)
  ROVER_LLM_BASE_URL    OpenAI-compatible base URL (optional)
  ROVER_LLM_MODEL       Model name (default: gpt-4o-mini)
  OLLAMA_HOST           Ollama host (default: http://localhost:11434)
  OLLAMA_MODEL          Ollama model (default: llama3.2)
  NO_COLOR              Disable colored output

Examples:
  curl -fsSL https://your-domain.com/rover.sh | bash
  curl -fsSL https://your-domain.com/rover.sh | bash -s -- --since 7d
  curl -fsSL https://your-domain.com/rover.sh | bash -s -- --project myrepo --output report.html
EOF
}

# --- Main ---
# Handle help flag before anything else
for arg in "$@"; do
  if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
    show_help
    exit 0
  fi
done

echo ""
echo "$(_bold "🏗️  Rover") — Local builder analytics"
echo ""

require_docker
build_image

# Pre-flight check: did we find any transcript dirs?
FOUND_TRANSCRIPTS=0
if [ -d "$HOME/.claude/projects" ]; then FOUND_TRANSCRIPTS=$((FOUND_TRANSCRIPTS + 1)); fi
if [ -d "$HOME/.codex/sessions" ]; then FOUND_TRANSCRIPTS=$((FOUND_TRANSCRIPTS + 1)); fi
detect_cursor_paths
if [ -d "$CURSOR_DIR" ]; then FOUND_TRANSCRIPTS=$((FOUND_TRANSCRIPTS + 1)); fi

if [ "$FOUND_TRANSCRIPTS" -eq 0 ]; then
  echo ""
  echo "$(_color yellow "⚠ No transcript directories found.")"
  echo ""
  echo "  Rover looks for AI coding transcripts in:"
  echo "    • ~/.claude/projects/           (Claude Code)"
  echo "    • ~/.codex/sessions/            (Codex CLI)"
  echo "    • ~/.config/Cursor/...          (Cursor IDE)"
  echo ""
  echo "  Make sure you have coding agent transcripts before running Rover."
  echo ""
  exit 1
fi

echo ""
echo "  Found $FOUND_TRANSCRIPTS transcript source(s). Analyzing..."
echo ""

# Collect mount args
mapfile -t MOUNT_ARGS < <(compose_mounts || true)

# Run analysis in Docker
OUTPUT_FILE="rover-report.html"
EXTRA_ARGS=("$@")

# Default --since 2m if not provided
if ! printf '%s\n' "${EXTRA_ARGS[@]}" | grep -q '^--since'; then
  EXTRA_ARGS+=("--since" "2m")
fi

# Default --output if not provided
if ! printf '%s\n' "${EXTRA_ARGS[@]}" | grep -q '^--output'; then
  EXTRA_ARGS+=("--output" "/reports/$OUTPUT_FILE")
fi

# TTY passthrough for rich TUI progress bars
TTY_ARGS=""
[ -t 0 ] && [ -t 1 ] && TTY_ARGS="-it"

# Pass LLM env vars through to container
docker run --rm $TTY_ARGS \
  -e ROVER_IN_DOCKER=1 \
  -e NO_COLOR="${NO_COLOR:-}" \
  -e ROVER_LLM_API_KEY="${ROVER_LLM_API_KEY:-}" \
  -e ROVER_LLM_BASE_URL="${ROVER_LLM_BASE_URL:-}" \
  -e ROVER_LLM_MODEL="${ROVER_LLM_MODEL:-}" \
  -e OLLAMA_HOST="${OLLAMA_HOST:-}" \
  -e OLLAMA_MODEL="${OLLAMA_MODEL:-}" \
  "${MOUNT_ARGS[@]}" \
  "$ROVER_IMAGE" \
  analyze \
  "${EXTRA_ARGS[@]}"

# The report is written to the mounted cwd
REPORT_PATH="$(pwd)/$OUTPUT_FILE"

if [ -f "$REPORT_PATH" ]; then
  echo ""
  echo "$(_color green "✔ Report ready:") $(_bold "$REPORT_PATH")"
  echo ""
  echo "  $(_color cyan "Open in browser:")"
  echo "    file://$REPORT_PATH"
  echo ""
  open_report "$REPORT_PATH"
else
  echo ""
  echo "$(_color yellow "⚠ Report was not generated. Check the output above for errors.")"
  exit 1
fi
