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

Environment: `NORTH_DEVICE` (`auto` | `cuda` | `xpu` | `cpu`),
`NORTH_MODEL_ID` (default the model above), `NORTH_MAX_NEW_TOKENS` (cap,
8192), `HF_HOME` (cache, `/hf` in the images).
