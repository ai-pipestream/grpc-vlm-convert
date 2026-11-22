# Qwen2.5-VL on the Intel GPU (llama.cpp SYCL)

The VLM endpoint grpc-vlm-convert and grpc-enrich call on krick-1 (port
8085). Until now it was a hand-run `docker run` of a floating
`llama.cpp:server-intel` tag with `-ngl 99` and no device pin, so nothing
stopped it from serving on CPU when SYCL found no GPU, and the default log
level hid the offload lines that would show it.

```bash
docker rm -f vlm-server                      # the hand-run container, once
docker compose -f serving/llama-vlm-intel/compose.yaml up -d
sh serving/llama-vlm-intel/check-gpu.sh      # exit 0 only when fully on SYCL0
```

`compose.yaml` pins the image digest (the build krick-1 runs today),
`--device SYCL0` and `-lv 3`. `check-gpu.sh` fails unless the latest boot
offloaded every layer, put the weights in a SYCL0 buffer and ran the vision
projector on SYCL0, and the server answers a short completion.

Not yet done: an API key (llama-server reads `LLAMA_API_KEY`; the clients
need to send it first), and a VRAM budget for the B70, which this server
shares with North, the OpenVINO model server, gRParse and grpc-asr (one
`UR_RESULT_ERROR_OUT_OF_RESOURCES` crash in the old container's log points at
contention). The port binds 127.0.0.1 unless `VLM_PUBLISH_ADDRESS` says
otherwise; set it to the tailnet address to keep today's remote callers.

llama.cpp also has an OpenVINO backend (`GGML_OPENVINO`), but no published
server image and no validation for vision models yet, so this stays on SYCL.
