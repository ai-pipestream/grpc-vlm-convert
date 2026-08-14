# grpc-vlm-convert

gRPC VLM convert collector: Granite-Docling / SmolDocling-class page parse into the gRParse Document data plane

This repo is a spec plus a standalone C++ gRPC server. It is not
PipeStream core and not a Docling Python wrapper. The VLM itself
(Granite-Docling, SmolDocling, GOT-OCR, …) is a *separate* server this
binary calls over an OpenAI-compatible HTTP endpoint; no model weights
are loaded or downloaded here.

## Build and test

Requires CMake ≥ 3.20, a C++17 compiler, and `buf` for proto lint.
gRPC, cpp-httplib, and nlohmann/json are fetched by CMake.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure   # golden test skips (77) without a VLM endpoint
buf lint
```

The unit/e2e tests fake the VLM HTTP endpoint in-process — no network,
no GPU. The golden test runs one real page against a live endpoint when
provisioned:

```sh
GRPC_VLM_TEST_ENDPOINT=http://localhost:8080 \
GRPC_VLM_TEST_PNG=/path/to/page.png \
  ./build/vlm-golden-test
```

## Run

```sh
GRPC_VLM_ENDPOINT=http://vlm:8080 ./build/grpc-vlm-convert-server
```

Configuration is entirely `GRPC_VLM_*` environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `GRPC_VLM_LISTEN_ADDRESS` | `0.0.0.0:50058` | gRPC listen address |
| `GRPC_VLM_ENDPOINT` | *(empty)* | OpenAI-compatible VLM endpoint. Empty is legal at startup; `ConvertPages` then needs a per-request endpoint or fails `FAILED_PRECONDITION` |
| `GRPC_VLM_PRESETS` | *(all built-ins)* | Comma list of preset names the endpoint claims to serve, reported by `GetServiceInfo` |
| `GRPC_VLM_CONCURRENCY` | `2` | Pages in flight against the VLM per stream |
| `GRPC_VLM_MAX_PAGE_BYTES` | `33554432` | Per-page PNG cap (`RESOURCE_EXHAUSTED`) |
| `GRPC_VLM_MAX_PAGES` | `512` | Per-stream page cap (`RESOURCE_EXHAUSTED`) |
| `GRPC_VLM_VLM_TIMEOUT_SECONDS` | `300` | Deadline for one page's VLM call |
| `GRPC_VLM_METRICS_INTERVAL_SECONDS` | `60` | Stdout metrics line interval, 0 disables |

Clients stream `ConvertPagesRequest` (one `ConvertOptions`, then one
`PageImage` PNG per page) and receive `PageStarted` / `PageDocument` /
`PageRaw` events per page **in completion order** — a page is emitted the
moment its VLM call returns, out-of-order pages are legal and key on
`page_no` — followed by a `ConvertComplete` trailer. Health
(`grpc.health.v1.Health`) and server reflection are registered.

Docker: `docker build -t grpc-vlm-convert .` — the build stage runs the
test suite and gates the image; the runtime is diskless (`--read-only`).

## Start here (humans and LLMs)

1. [`AGENTS.md`](AGENTS.md) — read order, definition of done, git
2. [`docs/architecture.md`](docs/architecture.md) — where this sits, language, live stream vs Docling
3. [`docs/design.md`](docs/design.md) — wire API, Document mapping, tests
4. [`docs/guidelines.md`](docs/guidelines.md) — fleet rules (streaming, proto, diskless, git)

Implementation is greenfield. Copy operational patterns from
`/work/main/grpc-services/gRParse (Document, page stream) and Docling's VlmPipeline only as the *reference for prompts/response shapes*, not the runtime.`.

## Docs

- [Architecture](docs/architecture.md) — where this sits in the collector fleet
- [Design](docs/design.md) — wire API, Document mapping, tests
- [Guidelines](docs/guidelines.md) — how to build it so it matches the fleet

## Remotes

- **Forgejo** (`git.rokkon.com/ai-pipestream/grpc-vlm-convert`) is the source of truth. `main` lives here.
- **GitHub** is a public push-mirror of `main`. Do not merge to GitHub `main`.
- GitHub's default branch is `development` so LLM / `gh` work lands there instead of clobbering the mirror.

Push Forgejo first. GitHub `main` updates from the Forgejo push-mirror.
