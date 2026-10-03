# North Micro Vision serving

An OpenAI-compatible endpoint for `CohereLabs/North-Micro-Vision-Instruct`
(2.4B, Apache 2.0), the open base of Cohere's document parser, for use as a
`grpc-vlm-convert` backend (`GRPC_VLM_ENDPOINT`, preset
`VLM_PRESET_NORTH_MICRO_VISION`). No GGUF or vLLM build exists for this
architecture yet, so the model runs on transformers, and the same server
builds three ways:

| image | accelerator | base |
|---|---|---|
| `Dockerfile.cuda` | NVIDIA | `pytorch/pytorch` CUDA runtime |
| `Dockerfile.xpu` | Intel Arc / Battlemage | `intel/intel-extension-for-pytorch` XPU |
| `Dockerfile.cpu` | CPU | `python:3.12-slim` + CPU torch wheels |

Every dependency is open source; nothing here calls a hosted API.

`compose.yaml` runs one of them on port 8086 with the weights cached in a
volume. `smoke.py` sends a page image and prints the markdown plus token
counts and wall time. The server serializes generation (one request at a
time), decodes greedily unless `temperature` is set, and honours
`max_tokens` and `stop`.

Whoever reaches the port gets the GPU, so the defaults are closed.
`compose.yaml` publishes 8086 on `127.0.0.1` only; to serve other hosts
set `NORTH_PUBLISH_ADDRESS` (for example `0.0.0.0`) together with
`NORTH_API_KEY`, and give clients the same key (`GRPC_VLM_API_KEY` for
grpc-vlm-convert). Images come as base64 data URIs; http(s) image URLs
are not fetched unless `NORTH_ALLOW_REMOTE_IMAGES=1`, since fetching
lets any caller make the server request any URL it can reach.

Environment: `NORTH_DEVICE` (`auto` | `cuda` | `xpu` | `cpu`),
`NORTH_MODEL_ID` (default the model above), `NORTH_MODEL_REVISION` (a
commit hash pins the weights; unset follows `main`),
`NORTH_MAX_NEW_TOKENS` (cap, 8192), `NORTH_API_KEY` (bearer key required
on `/v1/` when set; never logged), `NORTH_ALLOW_REMOTE_IMAGES` (`0`),
`NORTH_MAX_BODY_BYTES` (request body cap, 64 MiB; a body needs a
Content-Length), `NORTH_MAX_IMAGE_BYTES` (per image, 32 MiB),
`NORTH_MAX_IMAGE_PIXELS` (per image, 40 million), `HF_HOME` (cache, `/hf`
in the images).

The request guards have unit tests that stub the model stack, so only
`fastapi` and `pillow` are needed: `python -m unittest test_server` from
this directory.
