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

`GetServiceInfoResponse` carries a `UiInfo` block (title, path,
description), the same shape in every ai-pipestream service; the shared
demo shell reads it to build its tab bar.

Two input styles, first message chooses: `pdf_chunk` messages the server
rasterizes (fallback), or repeated `page_image` with PNG + page_no +
width/height (preferred). Every box is scaled to the page size, so a page
sent without a width or height takes both from its PNG header; one whose
header gives none fails as a `PageRaw` error before the model is called,
rather than mapping into items with empty boxes.

`ConvertOptions` carries a `preset` enum plus `preset_raw` for open
vocabularies, the expected `response_format` (`DOCTAGS` / `MARKDOWN` /
`HTML` / `OTSL` / `PLAINTEXT`), a `prompt` override, a `scale` hint, an
`endpoint` override, `concurrency` (pages in flight against the VLM),
and `abort_on_error`.

The `endpoint` override is refused with `PERMISSION_DENIED` unless the
operator sets `GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE=true`: it would let any
caller point this server at any host it can reach (SSRF). Naming the
configured endpoint itself is not an override, in any spelling that
reaches the same URL: the two are compared parsed, host case-blind, an
absent port as 80, trailing slashes and the `/v1` or
`/v1/chat/completions` suffix resolved the way calls resolve them. A
different host, port, path prefix or query is an override. The operator's
`GRPC_VLM_API_KEY` goes only to the configured endpoint, never to an
override.

An endpoint is an `http://` URL in one of three shapes: a base
(`http://vlm:8080`, optionally with a path prefix) gets
`/v1/chat/completions` appended, an OpenAI-style base ending in `/v1`
gets `/chat/completions`, and a full `.../chat/completions` URL
(Docling's `ApiVlmOptions.url`) is used as is. A `user:password@` part
is refused. Only the endpoint's scheme, host and port ever leave the
process (logs, errors, `GetServiceInfo`, `GenerationSource`), because
deployments put tokens in the path and query.

Events arrive in completion order, not page order:

1. `PageStarted`: page_no.
2. `PageDocument`: a `Document` fragment for that page (items,
   pictures, tables), or `PageRaw` (model text, or the error) if the
   call or the mapping failed.
3. `ConvertComplete`: pages started / ok / failed.

A `PageRaw` says why its page failed twice: `error` in words, and
`error_code` as a value to branch on.

| `PageErrorCode` | The page failed because |
| --- | --- |
| `ENDPOINT_UNREACHABLE` | no connection to the endpoint could be made or kept |
| `ENDPOINT_HTTP_ERROR` | the endpoint answered with a status other than 200 |
| `TIMEOUT` | the call outlived the per-page budget |
| `MALFORMED_RESPONSE` | a 200 that is not a chat completion with content |
| `UNMAPPABLE_OUTPUT` | the model answered outside the format asked for (`text` holds it) |
| `INVALID_PAGE` | the page names no size and its PNG header gives none |
| `INTERNAL` | this server failed while converting it |

For `ENDPOINT_HTTP_ERROR` the event also carries `endpoint_error`: the
HTTP status, and the message and type the endpoint's error reply gave
(the OpenAI error object, or the `error` / `detail` / `message` string or
plain-text body other servers send), cut to 2048 and 128 bytes with
`message_truncated` set when cut. That is what tells an image the model
refused from a server that ran out of memory; the same line goes to this
server's log. A reply that is markup, or larger than 64 KiB, is not read:
the status is still reported and nothing is guessed.

A stream does not end OK when the endpoint served none of it. If no page
converted and every page failed with one of the first three codes, the
page events and the trailer are delivered as usual and the status is then
`UNAVAILABLE`, naming the page count and the first failure; the metrics
line counts it under `streams{unavailable=}`, not `converted=`. A stream
where the model answered at all, however badly, still ends OK with its
failures in the events and the trailer. A client that leaves before the
trailer is written ends `CANCELLED` and is not counted as converted.

A `PageDocument` with no items is never a silent result. Every
`PageDocument` and every `PageRaw` whose model call succeeded carries the
call's `generation` (model, endpoint origin, stop reason, token cost), so a
page with no item to hold it still says how its answer ended. For the
layout formats (dots.ocr, Nemotron Parse, DocLang, Chandra) an answer that
maps to no items is one of three things:

- the model returned no text: an empty fragment with
  `PAGE_WARNING_CODE_EMPTY_RESPONSE`;
- the model declared the page empty in its format's own terms (dots.ocr
  `[]`, `<doclang></doclang>`, a Chandra `Blank-Page` block): an empty
  fragment, no warning;
- anything else (prose, a refusal, JSON cut off beyond repair, elements
  the mapper builds nothing from): a mapping failure, so the page is a
  `PageRaw` with the model's text and the reason, counted in
  `pages_failed`.

Docling degrades the third case to an empty page. This server does not,
because a blank page that reads as converted hides the loss.

A UI paints `PageDocument` immediately. gRParse merges fragments with
the same additive rules as other collectors; it must not hold page 4
until page 3 arrives.

### Health

This process holds no model, so its health is the VLM endpoint's. With no
endpoint a stream could use, or one that does not answer, every page of
every stream fails; a health check that said SERVING then would route an
orchestrator's traffic to a service that cannot do its one job.

An `EndpointMonitor` probes the configured endpoint once before the
listening line is printed and then every `GRPC_VLM_HEALTH_PROBE_SECONDS`
(default 15): one `GET` of the route beside its chat completions one
(`/v1/models` under the same path prefix and query), with the API key when
one is set. Any answer below 500 other than 401 and 403 is serving (an
endpoint without a model list answers 404, which still proves it is
reachable). Nothing answering within 5 s, a 5xx (a model still loading
answers 503) and 401/403 (the endpoint refuses this server's credentials)
are not serving. The verdict sets `grpc.health.v1.Health` for the overall
status and for `ai.pipestream.vlm.v1.VlmConvertService`, drives `/healthz`
(200 `ok` or 503 with the reason), and is logged on every change.

With `GRPC_VLM_ENDPOINT` empty there is nothing to probe: health is
serving only when per-request overrides are allowed, since a request can
then bring its own endpoint, and not serving otherwise. A probe interval
of 0 turns the probe off; health then reflects configuration alone.

Health gates nothing inside the server: a stream that arrives while the
endpoint is down still runs and fails page by page, exactly as before.

### HTTP shim

The same binary also serves an HTTP/JSON front end (`GRPC_VLM_HTTP_PORT`,
default 50059; 0 or empty disables it): `POST /v1/convert` (one
`ConvertHttpRequest` in, one `ConvertHttpResponse` out),
`POST /v1/convert/stream` (chunked NDJSON, one event per line as it
happens), and `GET /healthz`. The shim
is a framing adapter only: `VlmConvertServiceImpl::ConvertPagesCore`
takes the stream as three callables (read / write / cancelled), and the
gRPC override and the HTTP handlers both drive that one pipeline, so
concurrency caps, byte and page caps, `abort_on_error`, completion
order, and the error matrix (INVALID_ARGUMENT → 400,
PERMISSION_DENIED → 403, RESOURCE_EXHAUSTED → 413, UNIMPLEMENTED →
501, UNAVAILABLE → 503, else 500) cannot drift
between transports. The bodies are messages too
(`vlm_convert_http.proto`: `ConvertHttpRequest`, `ConvertHttpResponse`,
`ConvertHttpError` with a `ConvertHttpErrorCode`), each parsed or printed
whole by one protobuf call (`JsonStringToMessage` /
`MessageToJsonString`). The shim holds no JSON library and builds no JSON
text: the events of a buffered answer are kept as messages and printed
once with their body, and a request is read in one pass, so a field the
request message does not have is refused instead of read past. That file
is not part of the gRPC contract and gRParse does not vendor it. An event
protobuf cannot print fails the conversion INTERNAL on both routes (500
on the buffered one, an error line on the stream), never as the caller
having left.

The shim binds `GRPC_VLM_HTTP_HOST`, loopback by default: it converts
pages, and pays for VLM calls, for whoever reaches it, so any other host
requires `GRPC_VLM_HTTP_TOKEN` (startup fails without one), checked in a
pre-routing handler before the body is read. Bodies are capped at the
transport (`GRPC_VLM_HTTP_MAX_BODY_BYTES`, 413), because a request is
held several times over before the pipeline's own caps apply. The
synchronous route passes the connection's liveness as the pipeline's
`cancelled` probe, so a caller that hangs up halts its stream.

## 4. Response mapping

| Format | Mapping |
|---|---|
| DocTags | first-class: locations become `BoundingBox` `TOPLEFT`, tags become labels (`#/texts`, tables, pictures) |
| Markdown | `Markdown` declarative mapper (headings, lists, fenced code, pipe tables) + page bbox = full page |
| HTML | block-level snippet mapper (headings, paragraphs, list items, code, tables with real `TableData`) + page bbox = full page |
| OTSL | table-shaped items |
| Plaintext | one `TextItem` per page |
| DocLang | first `<doclang>` fragment (docling-core's DocLangDocDeserializer vocabulary): `<location>` quartets on the `<default_resolution>` grid (512) scale to the page; headings (level 1 is the title), text runs with formatting, lists split by `<ldiv>`, OTSL tables and indexes, pictures with label classification and `<tabular>` charts, float groups with caption and footnote satellites, `<layer>` content layers. No fragment fails with docling's message |
| DeepSeek-OCR markdown | `<|ref|>label<|/ref|><|det|>[[l,t,r,b]]<|/det|>` blocks on the 0..999 grid; unlabeled lead text is dropped as docling drops it; HTML tables become `TableData` |
| Unlimited-OCR markdown | `<|det|>label [l,t,r,b]<|/det|>` annotations rewritten to the DeepSeek-OCR shape (docling's normalize_unlimited_ocr_annotations), then mapped as DeepSeek-OCR |
| Chandra HTML | `<div data-bbox data-label>` blocks on the 0..1000 grid; inline HTML keeps formatting runs; tables keep spans and headers |
| dots.ocr JSON | layout cells with `bbox` (Qwen2-VL resized image pixels) and `category`; markdown headings carry their level and `orig`; captions link to the adjacent picture, table or code |
| Nemotron Parse 2.0 | `<x_><y_>text<x_><y_><class_>` regions on the padded square canvas, undone to the page; LaTeX tabular tables; charts with their pipe table as `tabular_chart` |
| Nemotron Parse 3.5 | normalized `<class_>` blocks with artifact and duplicate removal, boxes clamped to the page; LaTeX tabular tables with multirow/multicolumn; chart panels; TOC groups |
| MinerU2 | not served: a stream naming it fails `UNIMPLEMENTED` before a page is read (its two-pass layout-then-content protocol needs more than one call per page), and `GetServiceInfo.response_formats` omits it |

`ConvertOptions.response_format_raw` takes docling's string value
(`"dots_json"`); an unknown name, or one that disagrees with a set
`response_format`, fails `INVALID_ARGUMENT` before a page is read.

Headings come out at the depth the model wrote them. docling normalizes
heading depth document-wide for the Chandra, DeepSeek-OCR, Unlimited-OCR,
dots.ocr and Nemotron formats (when no heading sits at level 1, every
level moves up one); a page fragment cannot see the whole document, so
the consumer that assembles pages (gRParse) applies that step.

Deviations from docling 2.142.1, each a choice for a page-at-a-time
service rather than an oversight:

- Missing or malformed coordinates give the full-page box, never an
  invented one; oversize tables are cut to the caps below rather than
  raising.
- Markdown emphasis inside dots.ocr and Nemotron 3.5 text uses a
  CommonMark-style delimiter-run renderer, an approximation of marko.
- Nemotron Parse 2.0 tables use the same tabular parser as 3.5, not
  docling's LaTeX backend.
- Code language is the model's hint (`code_language_raw`), not detected.
- A chart whose table has no rows carries no `tabular_chart`.
- DocLang: `<thread>` continuations stay separate items, `<page_break>`
  does not open a new page (one response is one page), `<custom>`
  metadata and archive `<src>` images are ignored, rich table cells are
  flattened to text, and `field_*` form elements are walked as plain
  content.

### Nemotron Parse double and triple pass

`ConvertOptions.nemotron_parse` is docling's `NemotronParseOptions`. With
the single pass (the default) a page is one call with what docling's API
engine sends for its `nemotron_parse_v3_5` preset: greedy decoding
(temperature 0, top-k 1), repetition penalty 1.1, special tokens kept and
8744 tokens, with docling's page prompt
(`</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>`)
when the request sets none. With `NEMOTRON_PARSE_PASSES_DOUBLE` or `_TRIPLE` and the
`nemotron_parse_v3_5` format, a page follows docling's `_predict_nemotron`:

1. The page call, with docling's fullpage settings (temperature 0.2,
   repetition penalty 1.05, top-k 50, top-p 1.0, 8744 tokens, special
   tokens kept).
2. Triple only: a detection call with the no-text prompt and 4096 tokens.
   A failed detector crops nothing rather than falling back to a double
   pass.
3. Table and Chart regions (from the detector for triple, from the page
   call otherwise; with `TRIPLE_REGIONS_UNION`, page regions the detector
   missed are added) are cut from the raster with NVIDIA's margins
   (12 px plus 4% of the region), at most 80 per page, through a JPEG
   quality 95 round trip, and each is parsed on its own (temperature 0.1,
   repetition penalty 1.0) in batches of `crop_batch_size`.
4. A crop's blocks are mapped back to the page, and one replaces a page
   block only when it has the same class, overlaps it by IoU 0.5, keeps a
   table's column count, passes the strict grid or chart check, and shares
   at least 65% of its text (difflib ratio). A valid crop block with no
   page counterpart is inserted in reading order. The page block keeps its
   position and box.

The fragment's usage sums every call. It carries no logprobs or
alternates, because the merged text is not text the model generated, so
a multi-pass request with `top_logprobs` fails `INVALID_ARGUMENT`, as does
a multi-pass request for any other format. A failed detection or crop
call, or a raster that cannot be cropped, keeps the page pass text for
those regions and adds `PAGE_WARNING_CODE_NEMOTRON_PASS_INCOMPLETE` with
the counts; a crop the structure check refuses is the check working and
is not reported. A page call that did not finish (cut at `max_tokens`,
or blank) leaves nothing to crop from: the later passes do not run, and
the page carries the same warning saying so. A page's crop calls run beside its page slot only on
in-flight slots that are free right now, so `GRPC_VLM_MAX_INFLIGHT` still
bounds the calls open on the endpoint.

### Size caps and page warnings

Every mapper builds tables as a full rows × columns grid, so a model stuck
repeating itself (one 4000-cell row, then thousands of short rows) would
otherwise produce sixteen million `TableCell`s for one page. Tables, and a
chart's data table, keep at most their leading 2000 rows and 250 columns
within 50 000 grid cells. A grid whose cell copies (each anchor copied
onto every position its span covers) would cost more than 32 MiB, through
overlapping spans or long spanned text, is left empty and the cells stay.
Whatever was cut is reported beside the fragment as a typed `PageWarning`
on `PageDocument.warnings` (`PAGE_WARNING_CODE_TABLE_TRUNCATED` /
`PAGE_WARNING_CODE_TABLE_GRID_OMITTED`, the item's ref, and a message with
the numbers), so a cut table never passes for a complete one.

The same goes for what a mapper reads past. Model output is not always
well formed, and each mapper keeps what parses; what it could not use is
reported, never dropped in silence:

- `PAGE_WARNING_CODE_OUTPUT_TRUNCATED`: the answer breaks off inside its
  own layout (a dots.ocr array with no closing bracket after an unfinished
  element, a Nemotron Parse region or block that opens and never closes).
  The complete elements are mapped; the message gives the size of the
  tail that was not.
- `PAGE_WARNING_CODE_ELEMENTS_SKIPPED`: entries nothing was built from (a
  dots.ocr array entry that is not an object or has neither a usable
  bbox nor text, a Nemotron Parse 3.5 block with text and fewer than two
  corner tokens, a DocTags `<link_>` token that names no cell), counted.
- `PAGE_WARNING_CODE_LOCATION_UNREADABLE`: an item kept with its text on
  the full-page box because its location did not parse (a dots.ocr
  element whose bbox is missing or is not four numbers, a DocTags
  `<loc_>` token that is not a number), counted. Docling skips such a
  dots.ocr element; this server keeps its words.

A Nemotron Parse double or triple pass crops at most 80 regions; a page
with more reports the rest under
`PAGE_WARNING_CODE_NEMOTRON_PASS_INCOMPLETE`. When a pass merges crops
into the page text, the text is rebuilt from the page call's well-formed
blocks, so the pass reports what that call wrote outside them before the
mapper sees the rebuilt text.

### HTML mapping rules

Block matching spans newlines, since model output wraps its markup, and
is a linear scan rather than a regex: libstdc++'s regex engine recurses
per character, and one 60 KB block overflowed the stack. A tag name
matches exactly (`<p>` is not `<pre>`, `<tr>` is not `<track>`). A
`<table>` carries real `TableData`: `<tr>` rows of `<th>`/`<td>` cells,
1x1, ragged rows padded to a rectangular grid (padding is grid filler
only, never a source cell), and the leading run of all-`<th>` rows
flagged as column headers. Cells written without a surrounding `<tr>`
read as one row; markup with no cells at all keeps its stripped text as
a single cell. Spans (`rowspan` / `colspan`) are not read — that is the
HTML collector's job upstream.

Text extraction from a block or cell follows docling's
`chandra_utils._strip_tags`: a lowercase `<br>`, `<br/>` or `<br />`
becomes a single space *before* the generic tag strip (stripping it
bare would fuse the words around it), the remaining tags come off, the
six supported entities unescape, and whitespace runs collapse to one
space with the edges trimmed.

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

A tag outside the recognized vocabulary keeps its text as a TEXT item
and its own tag name in `label_raw`, so a label newer than this build
survives instead of being erased by the fallback.

`<code>` text parses a leading `<_language_>` token into
`code_language` (exact, case-sensitive match against the
`CodeLanguageLabel` vocabulary, UNKNOWN fallback) plus
`code_language_raw`.

A `<caption>` inside a table, picture, or chart chunk becomes a CAPTION
text item linked via the item's `captions` ref, emitted before its item.

Free text inside a `<picture>` or `<chart>` chunk is the model's own
description of the region: it becomes `meta.description.text` plus a
`description` annotation, both with created_by/provenance
`load_from_doctags` and no confidence, since the model reports none.

Classification tags inside `<picture>` and `<chart>` chunks produce a
classification prediction with created_by/provenance
`load_from_doctags`, recorded in both `meta.classification` and the
`annotations` union. The prediction carries no confidence: the tag is a
bare label with no probability behind it, and an unset optional says
"not reported" where 1.0 would say "certain". The recognized tag set is the v2 label list, the
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
still yields the PictureItem, just without an image. The raster is
decoded once per page, on the first crop, and never when its header
claims more than 40 million pixels; a page spends at most 100 crops and
twice its own pixels on them, so a model repeating `<picture>` cannot
multiply the page into its fragment. The data URIs a page carries are
also capped in bytes (`GRPC_VLM_MAX_PAGE_CROP_BYTES`, default 3 MiB):
one near-full-page crop at 300 DPI can come close to the 4 MiB receive
limit gRPC clients default to, and a `PageDocument` past the client's
limit fails its whole stream. The first crop that would cross the cap
is dropped, and so is every later crop on that page. A picture a cap
refused keeps its PictureItem without an image, and the page carries a
`PAGE_WARNING_CODE_PICTURE_IMAGES_SKIPPED` warning per cap that refused
one.

Logprobs: if the VLM endpoint returns them, the mean token
log-probability over the response rides the `CollectorSource` as
`raw_score` with `raw_score_kind` `page_mean_token_logprob` and
`raw_score_samples` set to the number of tokens the mean was taken
over. Skip silently when absent. Some OpenAI-compatible servers reject
the `logprobs` parameter with a 400; `GRPC_VLM_LOGPROBS=false` leaves it
off (and refuses `top_logprobs`, which needs it, with
`FAILED_PRECONDITION`).

It is deliberately not `confidence`. The mean is computed over the whole
page, so stamping it as a per-item confidence reports a crisp heading
and a hallucinated table as equally trustworthy; the kind names both the
statistic and its scope so no consumer mistakes it for a probability. It
is also neither exponentiated nor clamped: rescaling destroys the signal
and clamping hides an endpoint reporting nonsense. A real per-item
confidence needs per-item token spans, which the response does not
carry.

### Alternate readings

`ConvertOptions.top_logprobs` (0 to 20, default 0 = off) adds
`"top_logprobs": N` to the request. What comes back on
`logprobs.content[*].top_logprobs` becomes `Hypothesis` entries
(`text`, `raw_score`, `raw_score_kind` `token_logprob`) on the
fragment's **body group** `meta.alternatives`, `created_by` naming the
model that answered.

The body group, not the items: the response says nothing about which
item a token ended up in, so per-item alternates would be an invention.
Order is generation order, every alternate of token 1 then of token 2,
which is what makes the grouping recoverable from N. An alternate the
endpoint sends without a numeric score keeps its text and claims no
score. `Hypothesis.range` is never set for the same reason as above.

### Generation provenance

Every emitted item carries two sources: the `CollectorSource`
(`vlm-convert`, the resolved model name, this server's version) and a
`GenerationSource` describing the call that produced the page — the
model the endpoint says answered (left unset when it names none: the
requested name is in the `CollectorSource`, and an endpoint that serves
its one model whatever it is asked for would otherwise have its answers
recorded under a model that never ran), the endpoint origin (scheme, host and port; userinfo,
path and query are dropped because deployments hide tokens there), the
`finish_reason`
verbatim, and `prompt_tokens` / `completion_tokens` when the endpoint
reports usage. All of it is optional on the wire and recorded only when
present.

The `finish_reason` is the load-bearing one: `max_tokens` defaults to
4096, and a page whose answer hit that ceiling maps into a fragment
that looks complete and is not. A stop reason inside each item's source
list is easy to miss, so an answer the model did not finish (`"length"`,
`"content_filter"`, `"inference_error"`) is also reported where a caller
looks first: the `PageDocument` carries a
`PAGE_WARNING_CODE_GENERATION_TRUNCATED` warning, and the
`ConvertComplete` trailer counts the page in `pages_truncated`, whether
it mapped (`pages_ok`) or no longer parsed (`pages_failed`, a `PageRaw`
whose `generation` names the stop reason). The interval metrics line
prints the same count as `pages{truncated=}`.

### Retries

The HTTP client retries a page's VLM call up to 5 times with
exponential backoff (100 ms base: 0.1 s, 0.2 s, 0.4 s, ...) on HTTP
429/500/502/503/504 and on connect-level transport failures (connection
refused while the VLM server starts). Other statuses, and 200s that do
not parse, fail without a retry, so a persistently failing page
surfaces as a failed `PageRaw` after 6 attempts total. The configured
timeout (`GRPC_VLM_VLM_TIMEOUT_SECONDS`) is one wall-clock budget for
the whole call: attempts and backoff sleeps all spend from it, and an
endpoint that drips a byte at a time cannot stretch it (httplib's
per-read timeout alone would let it). Cancellation reaches the call
too: when the client cancels or its deadline passes, the stream notices
within 50 ms, and a watchdog per call shuts the in-flight socket down
(httplib `Client::stop`), mid-wait on the model included; no retry
follows and the page emits nothing. Tests pin the backoff base to zero
via `set_retry_backoff_base_ms`.

### Back-pressure and stopping

A page is admitted to the model queue (its `PageStarted` goes out) only
when its bytes fit two budgets: the stream's
(`GRPC_VLM_MAX_STREAM_BUFFERED_BYTES`) and the server's
(`GRPC_VLM_MAX_BUFFERED_BYTES`), both counting pages read but not yet
answered, queued or in flight. Until it fits, the read loop holds the
page and reads nothing more, so gRPC flow control holds a client that
sends faster than the model answers; waiters are served in arrival
order. The event queue is deliberately not bounded: a client that
uploads every page before reading any event (gRParse does) would
otherwise deadlock against it. It holds at most two events per page for
at most `max_pages` pages, but **the bytes of queued output are not
bounded**: both budgets count input PNG bytes only and are given back
once a page is mapped, so finished `PageDocument`s (text, tables, inline
picture crops) wait in server memory for as long as the client does not
read them. A client must read events concurrently with uploading pages;
one that uploads a whole document first makes the server hold the whole
converted document. Bounding output (for example, a page keeps its
stream lease until the writer hands its event to `Write`, so an unread
stream stops admitting pages) is the follow-up once gRParse reads
concurrently; done before that, it would deadlock gRParse, which writes
every page before reading any response.

Every page also takes one of the server's `GRPC_VLM_MAX_INFLIGHT`
slots for its VLM call and keeps it while the answer is mapped, so many
streams cannot pile requests onto an endpoint that serves one at a time,
nor decode more rasters at once than there are slots: a picture page
decodes its whole raster to crop from it, up to 4 bytes per pixel (about
160 MB at the raster pixel cap).

A stream halts once nobody will receive its answers: the client
cancelled or its deadline passed, the consumer stopped taking events, a
page turned out bad (`INVALID_ARGUMENT`, `RESOURCE_EXHAUSTED`, ...), or
`abort_on_error` met a failed page. Queued pages are then skipped, calls
in flight are cut, and nothing more is read.

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
8192; every other preset omits `stop` and uses 4096. The
`nemotron_parse_v3_5` format brings docling's own page settings with it,
whatever the preset (section 3, Nemotron Parse double and triple pass).

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
- Every way a page fails carries its `PageErrorCode`, an HTTP error its
  `endpoint_error`; a stream whose pages all failed at the endpoint
  (503s, a refusal, nothing listening, a timeout) ends `UNAVAILABLE`
  after its trailer and counts as `unavailable`, while one where the
  model answered ends OK.
- The HTTP bodies read back as `ConvertHttpResponse` messages; an
  unknown request field is 400; an unprintable event is INTERNAL on both
  routes (through the gateway's printer hook).
- Formats 6-13 (`docling-formats-test`): one answer per format shaped
  like docling's own tests and fixtures, checked for the items docling
  builds; MinerU2 fails typed. The service test covers
  `response_format_raw` selection, agreement, conflict and exact-name
  matching, MinerU2 `UNIMPLEMENTED` with no model call, and the
  `response_formats` advertisement.
- Nemotron multi-pass (`nemotron-multipass-test`, `nemotron-passes-test`):
  crop plans, grid and chart checks, remapping, merging and difflib's
  ratio against values docling's own functions produced; the stage
  settings, failed stages, batching, option checks, and the service path
  against a fake endpoint, including the in-flight cap, the single pass's
  settings and prompt on the wire, and page sizes taken from the PNG
  header.
- Golden: one real page against a live VLM endpoint, behind a flag, not
  CI default.
