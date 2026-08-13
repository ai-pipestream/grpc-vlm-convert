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

Logprobs: if the VLM endpoint returns them (Docling's OpenAI VLM
logprobs knob), attach `confidence` on the `CollectorSource`. Skip
silently when absent.

## 5. Presets vs endpoints

This service does not vendor 12 model graphs. `GetServiceInfo`
reports which presets the **configured endpoint** claims to serve.
Unknown `preset_raw` is forwarded as the model name on the wire to
the endpoint. A preset with no endpoint configured is
`FAILED_PRECONDITION` at RPC start, not a download from Hugging Face.

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
  still runs (`abort_on_error` false).
- No endpoint configured → `FAILED_PRECONDITION`.
- Golden: one real Granite-Docling page behind a flag, not CI default.
