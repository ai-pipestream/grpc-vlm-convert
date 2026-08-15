# grpc-vlm-convert design

## 1. Goals

- Cover the preset vocabulary of the `VlmPreset` enum (see
  `proto/ai/pipestream/vlm/v1/vlm_convert.proto`): two DocTags-emitting
  presets, GOT-OCR-2, Granite Vision, DeepSeek-OCR, Nanonets-OCR2,
  GLM-OCR, and LightOnOCR, plus `VLM_PRESET_RAW` for open vocabularies
  (Unlimited-OCR, Dolphin, Chandra, dots.ocr, ...).
- Page-streamed `Document` events with real provenance boxes when the
  response format carries them (DocTags). Markdown and HTML responses
  get page-level provenance only; do not invent word boxes. Emit a page
  when that page is ready, not after the last page: live UI is the
  product path. Out-of-order pages are legal if `page_no` is set.
- Typed mapping. No `json_format.MessageToDict` bridge. DocTags to
  proto items in code.

## 2. Non-goals (v1)

- Replacing RapidOCR as the default. This collector is opt-in.
- In-process torch / mlx / vLLM.
- A special two-stage flow for any preset beyond "call the endpoint
  twice if the preset says so."
- Grounding parse for raw-vocabulary models beyond what the model
  returns: we store boxes the model emitted, we do not snap them to a
  second OCR pass in v1.

## 3. Wire API

`ai.pipestream.vlm.v1.VlmConvertService`

```text
rpc ConvertPages(stream ConvertPagesRequest) returns (stream ConvertPagesResponse);
rpc GetServiceInfo(GetServiceInfoRequest) returns (GetServiceInfoResponse);
```

(The design sketch named the event stream `ConvertPagesEvent`; the proto
ships it as `ConvertPagesResponse`.)

Two input styles, first message chooses: `pdf_chunk` messages the server
rasterizes (fallback), or repeated `page_image` with PNG + page_no +
width/height (preferred).

`ConvertOptions` carries a `preset` enum plus `preset_raw` for open
vocabularies, the expected `response_format` (`DOCTAGS` / `MARKDOWN` /
`HTML` / `OTSL` / `PLAINTEXT`), a `prompt` override, a `scale` hint, an
`endpoint` override, `concurrency` (pages in flight against the VLM),
and `abort_on_error`.

Events arrive in completion order, not page order:

1. `PageStarted`: page_no.
2. `PageDocument`: a `Document` fragment for that page (items,
   pictures, tables), or `PageRaw` (model text, or the error) if the
   call or the mapping failed.
3. `ConvertComplete`: pages started / ok / failed.

A UI paints `PageDocument` immediately. gRParse merges fragments with
the same additive rules as other collectors; it must not hold page 4
until page 3 arrives.

### HTTP shim

The same binary also serves an HTTP/JSON front end (`GRPC_VLM_HTTP_PORT`,
default 50059; 0 or empty disables it): `POST /v1/convert` (one JSON
object in, `{"events": [...]}` out), `POST /v1/convert/stream` (chunked
NDJSON, one event per line as it happens), and `GET /healthz`. The shim
is a framing adapter only: `VlmConvertServiceImpl::ConvertPagesCore`
takes the stream as three callables (read / write / cancelled), and the
gRPC override and the HTTP handlers both drive that one pipeline, so
concurrency caps, byte and page caps, `abort_on_error`, completion
order, and the error matrix (INVALID_ARGUMENT → 400,
RESOURCE_EXHAUSTED → 413, UNIMPLEMENTED → 501, else 500) cannot drift
between transports. Message bodies are canonical proto3 JSON
(`MessageToJsonString` / `JsonStringToMessage`); nlohmann/json touches
only the `{"options", "pages"}` envelope.

## 4. Response mapping

| Format | Mapping |
|---|---|
| DocTags | first-class: locations become `BoundingBox` `TOPLEFT`, tags become labels (`#/texts`, tables, pictures) |
| Markdown | `Markdown` declarative mapper (headings, lists, fenced code, pipe tables) + page bbox = full page |
| HTML | HTML collector on the snippet |
| OTSL | table-shaped items |
| Plaintext | one `TextItem` per page |

### DocTags mapping rules

Items emit in raw token order and body children refs follow it: no
texts-then-pictures-then-tables regrouping. Every provenance carries
`charspan`: `[0, len(text))` for text items, `[0, 0]` for floating
items. Where a chunk has no usable locations, the standing fallback
stamps the full-page box.

OTSL spans resolve onto the anchor cell (`row_span` / `col_span` and
end offsets). `<lcel>` / `<ucel>` / `<xcel>` fillers are not emitted;
`<srow>` starts a section row. Header flags (`column_header`,
`row_header`, `row_section`) are preserved on the emitted cells.

`<code>` text parses a leading `<_language_>` token into
`code_language` (exact, case-sensitive match against the
`CodeLanguageLabel` vocabulary, UNKNOWN fallback) plus
`code_language_raw`.

A `<caption>` inside a table, picture, or chart chunk becomes a CAPTION
text item linked via the item's `captions` ref, emitted before its item.

Classification tags inside `<picture>` and `<chart>` chunks produce a
classification prediction with confidence 1.0 and created_by/provenance
`load_from_doctags`, recorded in both `meta.classification` and the
`annotations` union. The recognized tag set is the v2 label list, the
legacy v1 labels, and the legacy aliases of the smol preset (e.g.
`line` and `dot_line` map to `line_chart`), all in
`kClassificationLabels` in `src/mapping/doctags.cpp`. A `<chart>`
additionally parses its embedded OTSL into
`meta.tabular_chart.chart_data` plus the `tabular_chart` annotation.

`<key_value_region>` becomes a `KeyValueItem` (in `key_value_items`,
label KEY_VALUE_REGION) whose `GraphData` holds the `<key_N>` and
`<value_N>` cells, each with its own box and with loc and link tokens
stripped from the text. TO_VALUE links come from the cells' `<link_N>`
tokens; links that point at missing cells are dropped. The region's
provenance comes from the loc run ahead of the first `<key_N>` cell;
without it, no provenance is stamped.

`<ordered_list>` and `<unordered_list>` become one list group in
`document.groups` (name `list`, label LIST; both list kinds fold onto
LIST) holding the chunk's `<list_item>` children. Items keep their own
boxes and charspans, are parented to the group, and ordered items carry
`enumerated` plus `1.`-style markers. The group ref sits in the body
children where the chunk appeared (source-order emission).

`<inline>` becomes an inline group (label INLINE, name `group`) whose
children are the chunk's items, all stamped with the chunk's first
(shared) box. Items keep their standard kinds: a `list_item` inside
`<inline>` stays a ListItem.

Picture and chart regions are cropped from the page raster (stb) and
attached as `ImageRef` PNG data URIs; a missing or undecodable raster
still yields the PictureItem, just without an image.

Logprobs: if the VLM endpoint returns them, attach `confidence` on the
`CollectorSource`. Skip silently when absent.

### Retries

The HTTP client retries a page's VLM call up to 5 times with
exponential backoff (100 ms base: 0.1 s, 0.2 s, 0.4 s, ...) on HTTP
429/500/502/503/504 and on connect-level transport failures (connection
refused while the VLM server starts). Other statuses, and 200s that do
not parse, fail without a retry, so a persistently failing page
surfaces as a failed `PageRaw` after 6 attempts total. The configured
timeout applies per attempt (worst case 6 × timeout); tests pin the
backoff base to zero via `set_retry_backoff_base_ms`.

## 5. Presets vs endpoints

This service does not vendor 12 model graphs. `GetServiceInfo` reports
which presets the configured endpoint claims to serve. An unknown
`preset_raw` is forwarded as the model name on the wire to the
endpoint. A preset with no endpoint configured is
`FAILED_PRECONDITION` at RPC start, not a download from Hugging Face.

Generation parameters come from the preset table in `src/presets.cpp`:
the two DocTags presets send explicit stop sequences, one
`stop: ["</doctag>", "<end_of_utterance>"]` with `max_tokens` 4096 and
the other `stop: ["</doctag>", "<|end_of_text|>"]` with `max_tokens`
8192; every other preset omits `stop` and uses 4096.

## 6. Coordinator contract

`COLLECTOR_VLM` in gRParse:

- Prefer sending rasters the CV path already rendered when both
  collectors run (do not rasterize twice).
- If only VLM is selected, either this service rasterizes or gRParse
  renders and this service only maps. Pick the latter once
  `GRPARSE_PAGE_IMAGES` exists; the preview PNG is already the right
  size class, but VLM may want a dedicated scale. Options carry
  `scale`; gRParse should honor it when it is the renderer.

## 7. Tests

- Fake VLM HTTP server returns a canned DocTags page; the mapper
  produces one heading + one paragraph with boxes.
- Markdown canned response → heading labels, no fake word boxes.
- Endpoint 503 on page 2 → that page `PageRaw` / failure event, page 3
  still runs (`abort_on_error` false). The 503 is retried first (the
  client test counts the attempts), with test backoff pinned to zero.
- No endpoint configured → `FAILED_PRECONDITION`.
- Golden: one real page against a live VLM endpoint, behind a flag, not
  CI default.
