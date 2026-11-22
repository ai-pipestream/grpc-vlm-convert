#!/bin/sh
# Fails unless the llama.cpp VLM server put the whole model on the Intel GPU.
#
#   sh serving/llama-vlm-intel/check-gpu.sh [container]   (default vlm-server)
#
# Reads the load log of the container's latest boot (compose.yaml runs it
# with -lv 3) and requires:
#   - "offloaded N/N layers to GPU" with both numbers equal;
#   - a SYCL0 model buffer (the weights live on the device);
#   - the vision projector on SYCL0 ("CLIP using SYCL0 backend"), never
#     "CLIP using CPU backend".
# Then it sends one short text prompt and requires a reply, so a server that
# loaded on the GPU but cannot run there fails too. Exit 0 only when all hold.
set -eu
container=${1:-vlm-server}
log=$(mktemp)
trap 'rm -f "$log"' EXIT
# Only the latest boot counts: read the log from the container's start time.
started=$(docker inspect -f '{{.State.StartedAt}}' "$container")
docker logs --since "$started" "$container" > "$log" 2>&1

fail() { echo "check-gpu: FAIL: $*" >&2; exit 1; }
offload=$(grep -o "offloaded [0-9]*/[0-9]* layers to GPU" "$log" | tail -1)
[ -n "$offload" ] || fail "no offload line (is -lv 3 set?)"
done_layers=$(echo "$offload" | sed 's/offloaded \([0-9]*\)\/.*/\1/')
all_layers=$(echo "$offload" | sed 's/offloaded [0-9]*\/\([0-9]*\) .*/\1/')
[ "$done_layers" = "$all_layers" ] || fail "only $offload"
grep -q "SYCL0 model buffer size" "$log" || fail "no SYCL0 model buffer: weights are not on the GPU"
if grep -q "CLIP using CPU backend" "$log"; then fail "the vision projector runs on CPU"; fi
grep -q "CLIP using SYCL0 backend" "$log" || fail "no line puts the vision projector on SYCL0"

port=$(docker port "$container" 8080/tcp | head -1 | sed 's/.*://')
reply=$(curl -sf --max-time 120 "http://127.0.0.1:$port/completion" \
  -H 'Content-Type: application/json' -d '{"prompt":"Say ok.","n_predict":4}') ||
  fail "the server did not answer a completion"
echo "$reply" | grep -q '"content"' || fail "completion reply has no content"
echo "check-gpu: ok ($offload, weights and vision projector on SYCL0)"
