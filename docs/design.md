# grpc-vlm-convert design

## 1. Goals

- Feature parity with Docling `ProcessingPipeline.VLM` and the
  `VlmModelType` presets (SmolDocling, Granite-Docling, GOT-OCR-2,
  Granite Vision, DeepSeek-OCR, Nanonets-OCR2, GLM-OCR, LightOnOCR,
  plus `*_raw` for Unlimited-OCR / Dolphin / Chandra / dots.ocr / …).
- Page-streamed `Document` events with real provenance boxes when the
  response format carries them (DocTags). Markdown/HTML responses get
  page-level provenance only — do not invent word boxes. Emit a page
  when that page is ready, not after the last page: Docling's VLM
  convert is batch; live UI is the product path here. Out-of-order
  pages are legal if `page_no` is set.
- Typed mapping. No `json_format.MessageToDict` bridge. DocTags →
  proto items in code.

## 2. Non-goals (v1)

- Replacing RapidOCR as the default. This collector is opt-in.
- In-process torch / mlx / vLLM.
- Two-stage Granite-Docling as a special snowflake beyond "call the
  endpoint twice if the preset says so."
- Grounding parse for Unlimited-OCR beyond what the model returns
  (we store boxes the model emitted; we do not snap them to a second
  OCR pass in v1).

## 3. Wire API (sketch)

`ai.pipestream.vlm.v1.VlmConvertService`

```text
rpc ConvertPages(stream ConvertPagesRequest) returns (stream ConvertPagesEvent);
rpc GetServiceInfo(GetServiceInfoRequest) returns (ServiceInfo);
```

Two input styles, first message chooses:

- `pdf_bytes` / chunks — server rasterizes (fallback)
- `page_image` repeated — PNG + page_no + width/height (preferred)

Options:

- `preset` enum + `preset_raw`
- `response_format` — `DOCTAGS` / `MARKDOWN` / `HTML` / `OTSL` /
  `PLAINTEXT`
- `prompt` override
- `scale`
- `endpoint` override
- `concurrency` (pages in flight against the VLM)

Events (completion order, not page order):

1. `PageStarted` — page_no
2. `PageDocument` — a `Document` fragment for that page (items,
   pictures, tables) **or** `PageRaw` (model text) if mapping failed
3. `ConvertComplete` — pages ok / failed

A UI paints `PageDocument` immediately. gRParse merges fragments with
the same additive rules as other collectors; it must not hold page 4
until page 3 arrives.

## 4. Response mapping

| Format | Mapping |
|---|---|
| DocTags | first-class: locations become `BoundingBox` `TOPLEFT`, tags become labels (`#/texts`, tables, pictures) |
| Markdown | `Markdown` declarative mapper (headings, lists, fenced code, pipe tables) + page bbox = full page |
| HTML | HTML collector on the snippet |
| OTSL | table-shaped items |
| Plaintext | one `TextItem` per page |

DocTags mapping follows docling-core `load_from_doctags`:

- Items emit in raw token order; body children refs follow it (no
  texts→pictures→tables regrouping).
- Every provenance carries `charspan` — `[0, len(text))` for text items,
  `[0, 0]` for floating items.
- OTSL spans resolve onto the anchor cell (`row_span`/`col_span` and end
  offsets); `<lcel>`/`<ucel>`/`<xcel>` fillers are not emitted; `<srow>`
  starts a section row. Header flags (`column_header`/`row_header`/
  `row_section`) are kept — docling drops them; that is deliberate.
- `<code>` text parses a leading `<_language_>` token into
  `code_language` (exact, case-sensitive match against docling's
  `CodeLanguageLabel` values; UNKNOWN fallback) plus `code_language_raw`.
- `<caption>` inside table/picture/chart chunks becomes a CAPTION text
  item linked via the item's `captions` ref, emitted before its item.
- Classification tags inside `<picture>`/`<chart>` chunks (docling's v2 +
  legacy label lists, SmolDocling aliases mapped) produce a
  classification prediction with confidence 1.0 and
  created_by/provenance `load_from_doctags`, in both `meta.classification`
  and the `annotations` union.
- `<chart>` additionally parses its embedded OTSL into
  `meta.tabular_chart.chart_data` (+ the `tabular_chart` annotation).
- `<key_value_region>` becomes a `KeyValueItem` (`key_value_items`,
  label KEY_VALUE_REGION) whose `GraphData` holds the `<key_N>`/`<value_N>`
  cells — each with its own box and loc/link tokens stripped from the
  text — plus TO_VALUE links from the cells' `<link_N>` tokens, dropping
  links that point at missing cells like docling's validation. The
  proto's `GraphData` covers docling's cells/links shape exactly (no
  delta). The region's provenance comes from the loc run ahead of the
  first `<key_N>` cell; without it no provenance is stamped, as docling
  passes `prov=None`.
- `<ordered_list>`/`<unordered_list>` become a list group in
  `document.groups` (name `list`, label LIST — docling folds ORDERED_LIST
  onto LIST) holding the chunk's `<list_item>` children; items keep their
  own boxes/charspans, are parented to the group, and ordered items carry
  `enumerated` + `1.`-style markers. The group ref sits in the body
  children where the chunk appeared (source-order emission).
- `<inline>` becomes an inline group (label INLINE, docling's default
  name `group`) whose children are the chunk's items, all stamped with
  the chunk's first (shared) box. Structural delta: docling adds inline
  children via its plain-text path (a `list_item` inside `<inline>` is a
  TextItem, not a ListItem); we keep the standard item kinds. Where
  docling would stamp no provenance at all (missing locs), our standing
  full-page-box fallback applies instead.
- Picture/chart regions are cropped from the page raster (stb) and
  attached as `ImageRef` PNG data URIs; a missing or undecodable raster
  still yields the PictureItem, just without an image.

Logprobs: if the VLM endpoint returns them (Docling's OpenAI VLM
logprobs knob), attach `confidence` on the `CollectorSource`. Skip
silently when absent.

Retries follow docling's `api_image_request` (urllib3 `Retry`): up to 5
retries with exponential backoff (100ms base: 0.1s, 0.2s, 0.4s, …) on
HTTP 429/500/502/503/504 and on connect-level transport failures
(connection refused while vLLM starts). Other statuses — and 200s that
do not parse — fail without a retry, so a persistently failing page
surfaces as a failed `PageRaw` after 6 attempts total. The configured
timeout applies per attempt (worst case 6 × timeout); tests pin the
backoff base to zero via `set_retry_backoff_base_ms`.

## 5. Presets vs endpoints

This service does not vendor 12 model graphs. `GetServiceInfo`
reports which presets the **configured endpoint** claims to serve.
Unknown `preset_raw` is forwarded as the model name on the wire to
the endpoint. A preset with no endpoint configured is
`FAILED_PRECONDITION` at RPC start, not a download from Hugging Face.

Generation parameters follow docling's stage model specs: smoldocling
sends `stop: ["</doctag>", "<end_of_utterance>"]` with `max_tokens`
4096, granite-docling `stop: ["</doctag>", "<|end_of_text|>"]` with
`max_tokens` 8192; other presets omit `stop` and use 4096.

## 6. Coordinator contract

`COLLECTOR_VLM` in gRParse:

- Prefer sending rasters the CV path already rendered when both
  collectors run (do not rasterize twice).
- If only VLM is selected, either this service rasterizes or gRParse
  renders and this service only maps — pick the latter once
  `GRPARSE_PAGE_IMAGES` exists; the preview PNG is already the right
  size class, but VLM may want a dedicated scale. Options carry
  `scale`; gRParse should honor it when it is the renderer.

## 7. Tests

- Fake VLM HTTP server returns a canned DocTags page; mapper produces
  one heading + one paragraph with boxes.
- Markdown canned response → heading labels, no fake word boxes.
- Endpoint 503 on page 2 → that page `PageRaw`/failure event, page 3
  still runs (`abort_on_error` false); the 503 is retried first (client
  test counts the attempts), with test backoff pinned to zero.
- No endpoint configured → `FAILED_PRECONDITION`.
- Golden: one real Granite-Docling page behind a flag, not CI default.
