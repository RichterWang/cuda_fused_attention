#!/usr/bin/env bash
# Six representative shapes, five kernels, console only. This is the smoke test
# used in the demonstration; it is not the source of the reported numbers.
set -euo pipefail

build_dir="${1:-build}"
"$build_dir/bench_flash_attention_lite"
