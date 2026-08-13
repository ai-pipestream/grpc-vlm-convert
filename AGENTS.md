# AGENTS.md — grpc-vlm-convert

You are implementing **grpc-vlm-convert** from scratch in this repo. There is no
application code yet. Specs are the source of truth.

## Read this first, in order

1. This file
2. `docs/architecture.md` — fleet boundary, language, what we refuse to own
3. `docs/design.md` — wire API sketch, Document mapping, tests
4. `docs/guidelines.md` — fleet rules (streaming, proto, git, tests)

Do not start coding until those four are in your context. If architecture
and an existing sibling disagree on *process* (diskless, health, buf),
follow the sibling. If they disagree on *product* (live stream, Document
plane), follow architecture.md.

## This service

gRPC VLM convert collector: Granite-Docling / SmolDocling-class page parse into the gRParse Document data plane

- **Language:** C++ mapper + HTTP/gRPC client to a VLM server. Prefer PNG pages from the caller; PDF rasterize is fallback.
- **Copy from:** /work/main/grpc-services/gRParse (Document, page stream) and Docling's VlmPipeline only as the *reference for prompts/response shapes*, not the runtime.
- **Stack:** Alternate parse, not enrichment. Preset enum + preset_raw. No HF download at RPC time. FAILED_PRECONDITION if no endpoint.
- **Live stream:** PageStarted / PageDocument as *that page's* VLM returns (out-of-order OK with page_no). ConvertComplete trailer. Do not wait for the last page.

## Definition of done (v1)

ConvertPages stream, fake VLM returning canned DocTags+markdown, page-2 failure does not block page 3, health+reflection, Dockerfile.

Also: README with build/run; proto lint clean; tests that fail if someone
turns the stream back into a batch (assert an event before the input is
fully consumed, or per-item events before Complete).

## Workspace

Checkout path: `/work/main/grpc-services/grpc-vlm-convert`.
Git: `origin` = Forgejo (push `main` here). `github` = GitHub mirror.
Never merge GitHub `main`. See `docs/guidelines.md`.

gRParse wiring (`COLLECTOR_*` enum, endpoint env) is a **follow-up**.
Ship a working server in this repo first.
