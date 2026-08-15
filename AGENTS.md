# AGENTS.md: grpc-vlm-convert

The v1 server is implemented in this repo (C++ in `src/`, tests in
`tests/`, proto in `proto/`). The specs in `docs/` remain the source of
truth for behavior; when code and a spec disagree, fix the spec or the
code, not both silently.

## Read this first, in order

1. This file
2. `docs/architecture.md`: fleet boundary, language, what we refuse to own
3. `docs/design.md`: wire API, Document mapping, tests
4. `docs/guidelines.md`: fleet rules (streaming, proto, git, tests)

Do not start coding until those four are in your context. If architecture
and an existing sibling disagree on *process* (diskless, health, buf),
follow the sibling. If they disagree on *product* (live stream, Document
plane), follow architecture.md.

## This service

gRPC VLM convert collector: vision-language-model page parse into the
gRParse Document data plane.

- **Language:** C++ mapper plus an HTTP/gRPC client to a VLM server. Prefer PNG pages from the caller; PDF rasterize is the fallback.
- **Copy from:** `/work/main/grpc-services/gRParse` (Document, page stream). Prompt and response-shape behavior is specified in `docs/design.md` and pinned by the tests.
- **Stack:** Alternate parse, not enrichment. Preset enum + preset_raw. No Hugging Face download at RPC time. FAILED_PRECONDITION if no endpoint.
- **Live stream:** PageStarted / PageDocument as *that page's* VLM returns (out-of-order OK with page_no). ConvertComplete trailer. Do not wait for the last page.

## Definition of done (v1)

ConvertPages stream, fake VLM returning canned DocTags + markdown, page-2
failure does not block page 3, health + reflection, Dockerfile. All of
this ships; keep it green.

Also: README with build/run; proto lint clean; tests that fail if someone
turns the stream back into a batch (assert an event before the input is
fully consumed, or per-item events before Complete).

## Workspace

Checkout path: `/work/main/grpc-services/grpc-vlm-convert`.
Git: `origin` = Forgejo (push `main` here). `github` = GitHub mirror.
Never merge GitHub `main`. See `docs/guidelines.md`.

gRParse wiring (`COLLECTOR_*` enum, endpoint env) is a **follow-up**.
The working server lives in this repo.
