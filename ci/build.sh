#!/usr/bin/env bash
#
# Build the practice exam code.
#
# Usage:
#   ci/build.sh
set -euo pipefail

# Root directory.
ROOT_DIR="$(dirname "${BASH_SOURCE[0]}")/.."

# Practice exam code directory.
CODE_DIR="$ROOT_DIR/exams/exam1/code"

# Build the practice exam code.
cd "$CODE_DIR"
make build
