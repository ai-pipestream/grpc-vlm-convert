// Adversarial tests: hostile model output, malformed endpoint responses,
// and wire-level edge cases. Every verify_* names the input class it
// attacks; failures here are bugs, not style notes.

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "fixture.h"
#include "mapping/image_crop.h"
#include "mapping/mapper.h"
#include "presets.h"
#include "service/vlm_convert_service.h"
#include "vlm_client.h"

namespace docv1 = ai::pipestream::document::v1;
namespace vlmv1 = ai::pipestream::vlm::v1;

namespace {

std::string base64_decode(const std::string& encoded) {
    auto value_of = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    int group = 0, bits = 0;
    for (char c : encoded) {
        int value = value_of(c);
        if (value < 0) {
            continue;
        }
        group = (group << 6) | value;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((group >> bits) & 0xFF));
        }
    }
    return out;
}

// Tiny in-memory PNGs in the color models stb must decode for crops:
// 4x3 grayscale, 2x2 palette, 2x2 RGBA, 1x1 RGB.
const std::string kGray4x3 =
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAAAAACRn/EaAAAAF0lEQVR4nGNg4BKRY9AwsnFjCIhKyQMADI8C"
    "lWcdFq8AAAAASUVORK5CYII=";
const std::string kPalette2x2 =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAMAAABFaP0WAAAADFBMVEX/AAAA/wAAAP///wDWAo97AAAADklE"
    "QVR4nGNgYGRgYgYAABEAB56iKhIAAAAASUVORK5CYII=";
const std::string kRgba2x2 =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAGUlEQVR4nGP4z8DQwPAfCBkY/jv8//+fAQA+"
    "HAe64izsegAAAABJRU5ErkJggg==";
const std::string kRgb1x1 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElF"
    "TkSuQmCC";

vlm::mapping::PageContext page_context() {
    vlm::mapping::PageContext page;
    page.page_no = 1;
    page.width = 1000;
    page.height = 1000;
    page.source.set_collector("vlm-convert");
    page.source.set_model("test-model");
    return page;
}

// ---------------------------------------------------------------------------
// DocTags tokenizer/parser attacks.
// ---------------------------------------------------------------------------

// Literal '<' in element text (comparisons, "a < b") is text, not markup —
// docling's tag regex requires a letter, '/', or '_' after '<'.
void verify_doctags_literal_less_than() {
    const std::string text =
        "<doctag><text><loc_50><loc_100><loc_400><loc_150>3 < 4 and a < b</text>"
        "<text><loc_50><loc_200><loc_400><loc_250>ratios 1:2</text></doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "literal '<' page maps: " + error);
    require(doc.texts_size() == 2, "literal '<' does not eat the second element");
    require(doc.texts(0).text().base().text() == "3 < 4 and a < b",
            "literal '<' kept in the text: " + doc.texts(0).text().base().text());
}

// Text ahead of a truncated tag at end of input is still text.
void verify_doctags_truncated_close() {
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags("<doctag><text>abc</te", page_context(), &doc, &error),
            "truncated close still maps: " + error);
    require(doc.texts_size() == 1 && doc.texts(0).text().base().text() == "abc",
            "text before the truncated tag survives");
}

void verify_doctags_stray_close_and_nesting() {
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(
                "<doctag></text></picture><text><loc_1><loc_1><loc_2><loc_2>hi</text></doctag>",
                page_context(), &doc, &error),
            "stray close tags map: " + error);
    require(doc.texts_size() == 1 && doc.texts(0).text().base().text() == "hi",
            "stray closes are ignored");

    // Nested same-name tags: the inner open auto-closes the outer element.
    doc.Clear();
    require(vlm::mapping::map_doctags("<doctag><text>outer<text>inner</text>tail</doctag>",
                                      page_context(), &doc, &error),
            "nested same-name tags map: " + error);
    require(doc.texts_size() == 2, "nested same-name tags split into two items");
    require(doc.texts(0).text().base().text() == "outer" &&
                doc.texts(1).text().base().text() == "inner",
            "nested texts");
}

void verify_doctags_loc_boundaries() {
    // Grid boundary: 0 and 500 exactly map to the page edges.
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(
                "<doctag><text><loc_0><loc_0><loc_500><loc_500>x</text></doctag>",
                page_context(), &doc, &error),
            "grid boundary maps: " + error);
    const docv1::BoundingBox& box = doc.texts(0).text().base().prov(0).bbox();
    require(box.l() == 0 && box.t() == 0 && box.r() == 1000 && box.b() == 1000,
            "loc 0 and 500 hit the page edges exactly");

    // Beyond the grid: no clamp, no crash (docling does not clamp either).
    doc.Clear();
    require(vlm::mapping::map_doctags(
                "<doctag><text><loc_0><loc_0><loc_750><loc_250>x</text></doctag>",
                page_context(), &doc, &error),
            "over-grid loc maps: " + error);
    require(doc.texts(0).text().base().prov(0).bbox().r() == 1500,
            "over-grid loc is not silently clamped");

    // Fewer than four locs: full-page fallback. More than four: first four.
    doc.Clear();
    require(vlm::mapping::map_doctags(
                "<doctag><text><loc_10><loc_20><loc_30>x</text></doctag>", page_context(), &doc,
                &error),
            "three locs map: " + error);
    const docv1::BoundingBox& wide = doc.texts(0).text().base().prov(0).bbox();
    require(wide.l() == 0 && wide.r() == 1000, "three locs fall back to the full page");
    doc.Clear();
    require(vlm::mapping::map_doctags(
                "<doctag><text><loc_10><loc_20><loc_30><loc_40><loc_50>x</text></doctag>",
                page_context(), &doc, &error),
            "five locs map: " + error);
    const docv1::BoundingBox& first_four = doc.texts(0).text().base().prov(0).bbox();
    require(first_four.l() == 20 && first_four.t() == 40 && first_four.r() == 60 &&
                first_four.b() == 80,
            "the fifth loc is ignored");

    // Negative locs: the box goes negative (docling parity), no crash.
    doc.Clear();
    require(vlm::mapping::map_doctags(
                "<doctag><text><loc_-10><loc_-20><loc_100><loc_200>x</text></doctag>",
                page_context(), &doc, &error),
            "negative locs map: " + error);
    require(doc.texts(0).text().base().prov(0).bbox().l() == -20, "negative locs pass through");

    // Garbage and overflowing loc values are dropped, not fatal.
    doc.Clear();
    require(vlm::mapping::map_doctags(
                "<doctag><text><loc_abc><loc_99999999999999999999999999>x</text></doctag>",
                page_context(), &doc, &error),
            "garbage locs map: " + error);
}

// Huge loc values must not turn into UB when the crop scales the box into
// raster pixels (double -> int overflow there is undefined behavior).
void verify_doctags_huge_loc_crop() {
    vlm::mapping::PageContext page = page_context();
    page.width = 4;
    page.height = 3;
    page.png = base64_decode(kGray4x3);
    const std::string text =
        "<doctag>"
        "<picture><loc_0><loc_0><loc_9223372036854775807><loc_500></picture>"
        "<picture><loc_-9223372036854775808><loc_-100><loc_500><loc_500></picture>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page, &doc, &error),
            "huge locs map: " + error);
    require(doc.pictures_size() == 2, "both pictures emitted");
    require(doc.pictures(0).has_image(), "huge positive loc clamps to the raster edge");
    require(doc.pictures(0).image().size().width() == 4 &&
                doc.pictures(0).image().size().height() == 3,
            "huge loc crop is the full raster");
    require(doc.pictures(1).has_image(), "huge negative loc clamps to zero");
}

void verify_doctags_text_edge_cases() {
    docv1::Document doc;
    std::string error;
    // A <_..._> sequence in a plain text element is text, not structure.
    require(vlm::mapping::map_doctags("<doctag><text><_Python_>not code</text></doctag>",
                                      page_context(), &doc, &error),
            "language token in text maps: " + error);
    require(doc.texts(0).text().base().text() == "<_Python_>not code",
            "language token kept as text outside <code>");

    // Invalid UTF-8: charspan counts stay deterministic, nothing crashes.
    doc.Clear();
    const std::string bad_utf8 = std::string("<doctag><text>a\xff\xfe b</text></doctag>");
    require(vlm::mapping::map_doctags(bad_utf8, page_context(), &doc, &error),
            "invalid UTF-8 maps: " + error);
    require(doc.texts_size() == 1, "invalid UTF-8 item emitted");

    // Deeply unclosed runs: hundreds of opens, one parser pass, no stack.
    std::string deep = "<doctag>";
    for (int i = 0; i < 500; i++) {
        deep += "<text>";
    }
    deep += "end</doctag>";
    require(vlm::mapping::map_doctags(deep, page_context(), &doc, &error),
            "deep unclosed run maps: " + error);

    // Empty elements still emit (docling keeps empty items too).
    doc.Clear();
    require(vlm::mapping::map_doctags(
                "<doctag><text><loc_1><loc_2><loc_3><loc_4></text></doctag>", page_context(),
                &doc, &error),
            "empty element maps: " + error);
    require(doc.texts_size() == 1, "empty element emits one item");
}

// ---------------------------------------------------------------------------
// OTSL span-resolution attacks (through map_otsl, the bare-format entry).
// ---------------------------------------------------------------------------

docv1::Document map_otsl_or_fail(const std::string& text) {
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_otsl(text, page_context(), &doc, &error),
            "otsl maps: " + error);
    return doc;
}

void verify_otsl_adversarial() {
    // xcel at row 0 / col 0: up and left of nothing — it occupies the grid
    // position but anchors nothing.
    docv1::Document doc =
        map_otsl_or_fail("<otsl><xcel><fcel>A<nl><fcel>B<fcel>C<nl></otsl>");
    {
        const docv1::TableData& data = doc.tables(0).data();
        require(data.num_rows() == 2 && data.num_cols() == 2, "xcel grid shape");
        require(data.table_cells_size() == 3, "xcel itself is not a cell");
        require(data.grid(0).cells(0).text().empty(), "orphan xcel stays an empty grid cell");
    }

    // An lcel chain running to the row end extends the anchor's col_span
    // exactly to num_cols, never beyond.
    doc = map_otsl_or_fail("<fcel>A<lcel><lcel><nl><fcel>B<nl>");
    {
        const docv1::TableData& data = doc.tables(0).data();
        require(data.num_cols() == 3, "lcel chain widens the grid");
        require(data.table_cells(0).col_span() == 3 &&
                    data.table_cells(0).end_col_offset_idx() == 3,
                "lcel chain extends the span to the grid edge");
        require(data.grid(1).cells(2).text().empty(), "shorter rows leave empty grid cells");
    }

    // A ucel under a column the next row no longer has is out of range and
    // must not extend anything (or crash).
    doc = map_otsl_or_fail("<fcel>A<nl><fcel>B<ucel><nl>");
    {
        const docv1::TableData& data = doc.tables(0).data();
        require(data.table_cells(0).row_span() == 1, "ucel in another column extends nothing");
    }

    // srow as the very first token starts the first row.
    doc = map_otsl_or_fail("<srow>Section<fcel>x<nl>");
    {
        const docv1::TableData& data = doc.tables(0).data();
        require(data.table_cells(0).row_section(), "leading srow is a section row");
        require(data.table_cells(0).text() == "Section", "leading srow text");
    }

    // ecel-only table: empty cells at every position.
    doc = map_otsl_or_fail("<ecel><ecel><nl><ecel><ecel><nl>");
    {
        const docv1::TableData& data = doc.tables(0).data();
        require(data.num_rows() == 2 && data.num_cols() == 2, "ecel-only grid");
        require(data.table_cells_size() == 4, "ecels are cells");
        for (const docv1::TableCell& cell : data.table_cells()) {
            require(cell.text().empty(), "ecel text is empty");
        }
    }

    // nl-heavy input holds no complete row: a mapping failure (PageRaw),
    // not an empty table.
    {
        docv1::Document failed;
        std::string error;
        require(!vlm::mapping::map_otsl("<otsl><nl><nl><nl></otsl>", page_context(), &failed,
                                        &error),
                "nl-only input fails mapping");
    }

    // A header token in a later row keeps its flag (deliberate advantage
    // over docling, which drops the flags entirely).
    doc = map_otsl_or_fail("<fcel>a<nl><ched>h<nl>");
    require(doc.tables(0).data().table_cells(1).column_header(),
            "ched outside the first row keeps its header flag");

    // A literal '<' inside cell text is text, not a token opener.
    doc = map_otsl_or_fail("<fcel>3 < 4<nl>");
    require(doc.tables(0).data().table_cells(0).text() == "3 < 4",
            "literal '<' kept in cell text: " + doc.tables(0).data().table_cells(0).text());
}

// ---------------------------------------------------------------------------
// Image-crop attacks (direct calls against crafted rasters and boxes).
// ---------------------------------------------------------------------------

void verify_image_crop_adversarial() {
    docv1::ImageRef image;

    // Color models stb must normalize to RGBA and re-encode cleanly.
    require(vlm::mapping::crop_png_image(base64_decode(kGray4x3), 0, 0, 2, 2, 4, 3, &image),
            "grayscale PNG crops");
    require(image.size().width() == 2 && image.size().height() == 2, "grayscale crop size");
    require(image.mimetype() == "image/png" &&
                image.uri().starts_with("data:image/png;base64,"),
            "grayscale crop re-encodes as a PNG data URI");
    require(vlm::mapping::crop_png_image(base64_decode(kPalette2x2), 0, 0, 2, 2, 2, 2, &image),
            "palette PNG crops");
    require(vlm::mapping::crop_png_image(base64_decode(kRgba2x2), 0, 0, 2, 2, 2, 2, &image),
            "alpha PNG crops");

    // 1x1 raster, full-box crop.
    require(vlm::mapping::crop_png_image(base64_decode(kRgb1x1), 0, 0, 1, 1, 1, 1, &image),
            "1x1 raster crops");

    // Boxes partially/fully outside the raster, zero-area boxes, and boxes
    // at the raster edge.
    require(vlm::mapping::crop_png_image(base64_decode(kGray4x3), 3, 2, 10, 10, 4, 3, &image),
            "partially outside box clamps");
    require(image.size().width() == 1 && image.size().height() == 1, "clamped crop size");
    require(!vlm::mapping::crop_png_image(base64_decode(kGray4x3), 10, 10, 20, 20, 4, 3, &image),
            "fully outside box yields no image");
    require(!vlm::mapping::crop_png_image(base64_decode(kGray4x3), 1, 1, 1, 2, 4, 3, &image),
            "zero-width box yields no image");
    require(vlm::mapping::crop_png_image(base64_decode(kGray4x3), 0, 0, 4, 3, 4, 3, &image),
            "box at the raster edge crops");

    // Declared page dims disagreeing with the decoded raster: the box is
    // scaled into real pixels before clamping.
    require(vlm::mapping::crop_png_image(base64_decode(kGray4x3), 0, 0, 500, 250, 1000, 1000,
                                         &image),
            "declared dims scale into raster pixels");
    require(image.size().width() == 2 && image.size().height() == 1, "scaled crop size");

    // Corrupt rasters: garbage bytes and a truncated PNG header.
    require(!vlm::mapping::crop_png_image("this is not a png at all", 0, 0, 2, 2, 4, 3, &image),
            "garbage bytes yield no image");
    const std::string truncated = base64_decode(kGray4x3).substr(0, 20);
    require(!vlm::mapping::crop_png_image(truncated, 0, 0, 2, 2, 4, 3, &image),
            "truncated PNG yields no image");
}

// ---------------------------------------------------------------------------
// Markdown attacks.
// ---------------------------------------------------------------------------

docv1::Document map_markdown_or_fail(const std::string& text) {
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_markdown(text, page_context(), &doc, &error),
            "markdown maps: " + error);
    return doc;
}

void verify_markdown_adversarial() {
    // An escaped pipe is cell content, not a column boundary.
    docv1::Document doc = map_markdown_or_fail("| a \\| b | c |\n| --- | --- |\n| 1 | 2 |");
    {
        const docv1::TableData& data = doc.tables(0).data();
        require(data.num_cols() == 2, "escaped pipe does not add a column");
        require(data.table_cells(0).text() == "a | b", "escaped pipe unescapes: " +
                                                            data.table_cells(0).text());
    }

    // Ragged rows: the grid stays rectangular at num_cols so consumers can
    // index grid(r).cells(c) uniformly (docling's grid invariant).
    doc = map_markdown_or_fail("| a | b | c |\n| 1 |");
    {
        const docv1::TableData& data = doc.tables(0).data();
        require(data.num_cols() == 3, "ragged table num_cols is the max");
        require(data.grid(1).cells_size() == 3, "ragged grid rows pad to num_cols");
        require(data.grid(1).cells(2).text().empty(), "padded cells are empty");
    }

    // ATX closing sequences come off the heading text; a '#' glued to the
    // text (C#) stays.
    doc = map_markdown_or_fail("# Title ##\n\n## C#\n\n# C++ #");
    require(doc.texts(0).section_header().base().text() == "Title",
            "closing hashes stripped: " + doc.texts(0).section_header().base().text());
    require(doc.texts(1).section_header().base().text() == "C#", " glued hash kept");
    require(doc.texts(2).section_header().base().text() == "C++", "trailing sequence stripped");

    // Seven hashes is not a heading.
    doc = map_markdown_or_fail("####### too deep");
    require(doc.texts(0).has_text(), "level 7 is a paragraph");

    // Fences: language with spaces, unclosed fence at EOF, CRLF endings.
    doc = map_markdown_or_fail("```c++ extra words\ncode line");
    require(doc.texts(0).code().code_language_raw() == "c++ extra words",
            "fence language keeps spaces");
    require(doc.texts(0).code().text() == "code line", "unclosed fence still closes at EOF");
    doc = map_markdown_or_fail("# Heading\r\n\r\nbody text\r\n");
    require(doc.texts(0).has_section_header() && doc.texts(1).text().base().text() == "body text",
            "CRLF line endings");

    // "10)" is an enumeration marker; "iii." is a paragraph.
    doc = map_markdown_or_fail("10) tenth\n\niii. roman");
    require(doc.texts(0).list_item().enumerated() &&
                doc.texts(0).list_item().marker() == "10)",
            "10) enumerates");
    require(doc.texts(1).has_text(), "iii. is a paragraph");
}

// ---------------------------------------------------------------------------
// HTML attacks.
// ---------------------------------------------------------------------------

void verify_html_adversarial() {
    docv1::Document doc;
    std::string error;

    // Mixed-case tags map like their lowercase spelling.
    require(vlm::mapping::map_html("<Table><tr><td>x</td></tr></Table>", page_context(), &doc,
                                   &error),
            "mixed-case table maps: " + error);
    require(doc.tables_size() == 1, "<Table> is a table, not a paragraph");
    doc.Clear();
    require(vlm::mapping::map_html("<Pre>code</Pre>", page_context(), &doc, &error),
            "mixed-case pre maps: " + error);
    require(doc.texts(0).has_code(), "<Pre> is code");

    // <pre> with markup inside keeps the text only.
    doc.Clear();
    require(vlm::mapping::map_html("<pre>a<b>c</b></pre>", page_context(), &doc, &error),
            "pre with markup maps: " + error);
    require(doc.texts(0).code().text() == "ac", "pre body strips inner tags");

    // Unclosed block tags hold no complete element: a mapping failure.
    doc.Clear();
    require(!vlm::mapping::map_html("<p>never closed", page_context(), &doc, &error),
            "unclosed blocks fail mapping");

    // The six supported entities unescape; others stay literal (documented
    // snippet-mapper scope).
    doc.Clear();
    require(vlm::mapping::map_html("<p>it&#39;s &copy; acme</p>", page_context(), &doc, &error),
            "entities map: " + error);
    require(doc.texts(0).text().base().text() == "it's &copy; acme",
            "&#39; unescapes, &copy; stays literal");
}

// ---------------------------------------------------------------------------
// VLM client attacks: hostile endpoint responses and endpoint URLs.
// ---------------------------------------------------------------------------

// A /v1/chat/completions that replays a scripted response sequence (last
// entry repeats). 503s carry a garbage Retry-After.
struct ScriptVlm {
    httplib::Server server;
    std::thread thread;
    int port = 0;
    std::mutex mutex;
    std::deque<std::pair<int, std::string>> script;
    std::atomic<long> attempts{0};

    void start() {
        server.Post("/v1/chat/completions",
                    [this](const httplib::Request&, httplib::Response& response) {
                        attempts++;
                        std::pair<int, std::string> step;
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            step = script.front();
                            if (script.size() > 1) {
                                script.pop_front();
                            }
                        }
                        response.status = step.first;
                        if (step.first == 503) {
                            response.set_header("Retry-After", "banana");
                        }
                        response.set_content(step.second, "application/json");
                    });
        port = server.bind_to_any_port("127.0.0.1");
        require(port > 0, "script VLM bound");
        thread = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    void stop() {
        server.stop();
        thread.join();
    }

    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port); }
};

vlm::VlmCall call_to(const std::string& endpoint) {
    return {.endpoint = endpoint,
            .model = "test-model",
            .prompt = "prompt",
            .stop = {},
            .max_tokens = 4096,
            .png = "fake-png-bytes",
            .timeout_seconds = 5};
}

std::string completion(const std::string& content_json, const std::string& extra = "") {
    return "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":" + content_json +
           "}" + extra + "}]}";
}

void verify_client_adversarial(ScriptVlm* fake) {
    // Malformed logprobs: tokens that are not objects, objects without a
    // logprob, non-numeric logprobs. The one valid token sets the mean.
    // (Walking these with const json::operator[] is undefined behavior.)
    fake->script = {{200, completion("\"text\"",
                                     ",\"logprobs\":{\"content\":[{\"token\":\"a\"},\"oops\","
                                     "{\"logprob\":-0.5},{\"logprob\":\"high\"}]}")}};
    vlm::VlmResult result = vlm::generate(call_to(fake->endpoint()));
    require(result.ok, "malformed logprobs do not fail the page: " + result.error);
    require(result.has_logprobs && std::fabs(result.mean_logprob + 0.5) < 1e-9,
            "mean logprob from the one valid logprob");

    // Structural garbage: empty choices, content as an array, an error
    // body with a 200. All clean failures, no crashes.
    fake->script = {{200, "{\"choices\":[]}"}};
    require(!vlm::generate(call_to(fake->endpoint())).ok, "empty choices fail");
    fake->script = {{200, completion("[{\"type\":\"text\",\"text\":\"hi\"}]")}};
    require(!vlm::generate(call_to(fake->endpoint())).ok, "array content fails");
    fake->script = {{200, "{\"error\":{\"message\":\"boom\"}}"}};
    require(!vlm::generate(call_to(fake->endpoint())).ok, "200 with an error body fails");

    // A garbage Retry-After does not break the retry policy.
    fake->attempts = 0;
    fake->script = {{503, "{\"error\":\"busy\"}"}, {200, completion("\"ok\"")}};
    result = vlm::generate(call_to(fake->endpoint()));
    require(result.ok && fake->attempts == 2, "garbage Retry-After still retries");

    // Escapes and unicode in content round-trip byte-exact.
    fake->script = {{200, completion("\"line \\n \\\"quoted\\\" caf\\u00e9 \\u2622\"")}};
    result = vlm::generate(call_to(fake->endpoint()));
    require(result.ok && result.text == "line \n \"quoted\" caf\xc3\xa9 \xe2\x98\xa2",
            "escapes and unicode round-trip");

    // Endpoint with a trailing slash must not produce //v1/chat/completions.
    fake->script = {{200, completion("\"ok\"")}};
    result = vlm::generate(call_to(fake->endpoint() + "/"));
    require(result.ok, "trailing-slash endpoint resolves: " + result.error);

    // The raw score is reported as given, not laundered: positive
    // (garbage) logprobs are not clamped away and extreme ones are not
    // collapsed. A clamp would hide an endpoint reporting nonsense.
    fake->script = {{200, completion("\"text\"", ",\"logprobs\":{\"content\":["
                                                 "{\"logprob\":3.0},{\"logprob\":2.0}]}")}};
    result = vlm::generate(call_to(fake->endpoint()));
    require(result.ok && result.has_logprobs && std::fabs(result.mean_logprob - 2.5) < 1e-9,
            "impossible positive logprobs survive as the raw mean");
    fake->script = {{200, completion("\"text\"", ",\"logprobs\":{\"content\":["
                                                 "{\"logprob\":-1e300}]}")}};
    result = vlm::generate(call_to(fake->endpoint()));
    require(result.ok && result.has_logprobs && result.mean_logprob == -1e300,
            "an extreme logprob is not floored");
}

// ---------------------------------------------------------------------------
// Preset resolution: override combinations and unknown enum values.
// ---------------------------------------------------------------------------

void verify_presets_adversarial() {
    std::string model, prompt;
    vlmv1::ResponseFormat format;
    std::vector<std::string> stop;
    int max_tokens = 0;
    vlmv1::ConvertOptions options;

    // RAW with no preset_raw resolves to nothing.
    options.set_preset(vlmv1::VLM_PRESET_RAW);
    require(!vlm::resolve_request(options, &model, &prompt, &format, &stop, &max_tokens),
            "RAW without preset_raw fails resolution");

    // RAW: preset_raw is the model, the DocTags defaults apply, and a
    // prompt override still wins.
    options.set_preset_raw("custom/vlm-9b");
    require(vlm::resolve_request(options, &model, &prompt, &format, &stop, &max_tokens),
            "RAW resolves with preset_raw");
    require(model == "custom/vlm-9b" && format == vlmv1::RESPONSE_FORMAT_DOCTAGS,
            "RAW model and default format");
    options.set_prompt("Transcribe.");
    require(vlm::resolve_request(options, &model, &prompt, &format, &stop, &max_tokens) &&
                prompt == "Transcribe.",
            "prompt override wins on RAW");

    // preset_raw on a named preset overrides only the model name.
    options = vlmv1::ConvertOptions();
    options.set_preset(vlmv1::VLM_PRESET_SMOLDOCLING);
    options.set_preset_raw("mirror/smoldocling");
    require(vlm::resolve_request(options, &model, &prompt, &format, &stop, &max_tokens),
            "named preset resolves");
    require(model == "mirror/smoldocling" && prompt == "Convert this page to docling." &&
                format == vlmv1::RESPONSE_FORMAT_DOCTAGS && max_tokens == 4096,
            "preset_raw overrides the model, keeps the preset's generation knobs");

    // An unknown preset enum (proto3 raw int) fails resolution.
    options = vlmv1::ConvertOptions();
    options.set_preset(static_cast<vlmv1::VlmPreset>(123));
    require(!vlm::resolve_request(options, &model, &prompt, &format, &stop, &max_tokens),
            "unknown preset enum fails resolution");

    // An unknown response_format survives resolution; the service rejects
    // it before any VLM call (asserted in verify_service_adversarial).
    options = vlmv1::ConvertOptions();
    options.set_response_format(static_cast<vlmv1::ResponseFormat>(123));
    require(vlm::resolve_request(options, &model, &prompt, &format, &stop, &max_tokens) &&
                static_cast<int>(format) == 123,
            "unknown response_format passes through resolution for the service to reject");
}

// ---------------------------------------------------------------------------
// Service-level attacks over gRPC loopback.
// ---------------------------------------------------------------------------

// Marker-driven fake VLM: FAIL → 503, SLOW → 2s delay, else canned DocTags.
struct FakeVlm {
    httplib::Server server;
    std::thread thread;
    int port = 0;
    std::atomic<long> calls{0};

    void start() {
        server.Post("/v1/chat/completions", [this](const httplib::Request& request,
                                                   httplib::Response& response) {
            calls++;
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            std::string url = body["messages"][0]["content"][1]["image_url"]["url"];
            const std::string png = base64_decode(url.substr(22));
            if (png.contains("FAIL")) {
                response.status = 503;
                response.set_content("{\"error\":\"model overloaded\"}", "application/json");
                return;
            }
            if (png.contains("SLOW")) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            nlohmann::json reply = {
                {"choices",
                 {{{"message",
                    {{"role", "assistant"},
                     {"content", "<doctag><text><loc_1><loc_1><loc_2><loc_2>x</text></doctag>"}}}}}}};
            response.set_content(reply.dump(), "application/json");
        });
        port = server.bind_to_any_port("127.0.0.1");
        require(port > 0, "fake VLM bound");
        thread = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    void stop() {
        server.stop();
        thread.join();
    }

    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port); }
};

struct TestServer {
    vlm::Config config;
    std::unique_ptr<vlm::VlmConvertServiceImpl> service;
    std::unique_ptr<grpc::Server> server;
    std::shared_ptr<grpc::Channel> channel;

    explicit TestServer(vlm::Config cfg) : config(std::move(cfg)) {
        service = std::make_unique<vlm::VlmConvertServiceImpl>(config);
        int port = 0;
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(service.get());
        server = builder.BuildAndStart();
        require(server != nullptr, "gRPC server started");
        channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                      grpc::InsecureChannelCredentials());
    }

    void stop() { server->Shutdown(); }
};

vlmv1::PageImage page(uint32_t page_no, const std::string& marker) {
    vlmv1::PageImage image;
    image.set_page_no(page_no);
    image.set_png(make_png(marker));
    image.set_width(1000);
    image.set_height(1000);
    return image;
}

struct Collected {
    int documents = 0;
    int raws = 0;
    bool got_complete = false;
    vlmv1::ConvertComplete complete;
    grpc::Status status;
};

Collected convert(const std::shared_ptr<grpc::Channel>& channel,
                  const vlmv1::ConvertOptions& options,
                  const std::vector<vlmv1::PageImage>& pages) {
    Collected out;
    auto stub = vlmv1::VlmConvertService::NewStub(channel);
    grpc::ClientContext context;
    auto stream = stub->ConvertPages(&context);
    vlmv1::ConvertPagesRequest request;
    *request.mutable_options() = options;
    stream->Write(request);
    for (const vlmv1::PageImage& p : pages) {
        request.Clear();
        *request.mutable_page_image() = p;
        stream->Write(request);
    }
    stream->WritesDone();
    vlmv1::ConvertPagesResponse event;
    while (stream->Read(&event)) {
        if (event.has_page_document()) {
            out.documents++;
        } else if (event.has_page_raw()) {
            out.raws++;
        } else if (event.has_complete()) {
            out.got_complete = true;
            out.complete = event.complete();
        }
    }
    out.status = stream->Finish();
    return out;
}

void verify_service_adversarial(FakeVlm* fake) {
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.concurrency = 4;
    TestServer server(config);

    // Unknown enum values on the wire (proto3 keeps the raw int): an
    // unknown response format is rejected up front, not discovered as a
    // per-page mapping failure after paying for the VLM calls.
    vlmv1::ConvertOptions options;
    options.set_response_format(static_cast<vlmv1::ResponseFormat>(123));
    Collected out = convert(server.channel, options, {page(1, "PAGE1")});
    require(out.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "unknown response_format is INVALID_ARGUMENT");
    options.clear_response_format();
    options.set_preset(static_cast<vlmv1::VlmPreset>(123));
    out = convert(server.channel, options, {page(1, "PAGE1")});
    require(out.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "unknown preset is INVALID_ARGUMENT");

    // Concurrency 1 with several pages: every page still converts.
    options = vlmv1::ConvertOptions();
    options.set_concurrency(1);
    out = convert(server.channel, options,
                  {page(1, "P1"), page(2, "P2"), page(3, "P3"), page(4, "P4"), page(5, "P5")});
    require(out.status.ok() && out.documents == 5 && out.complete.pages_ok() == 5,
            "concurrency 1 converts every page");

    // Duplicate page numbers pass through (the stream is a page sequence,
    // not a set) — both events arrive.
    out = convert(server.channel, options, {page(1, "P1"), page(1, "P1B")});
    require(out.status.ok() && out.documents == 2 && out.complete.pages_started() == 2,
            "duplicate page_no delivers both pages");

    // abort_on_error with the failure on the LAST page still aborts.
    options.set_abort_on_error(true);
    out = convert(server.channel, options, {page(1, "P1"), page(2, "FAIL-LAST")});
    require(out.status.error_code() == grpc::StatusCode::ABORTED,
            "abort_on_error fires on a last-page failure");
    server.stop();

    // max_pages boundary: exactly the cap converts, one over is rejected.
    vlm::Config capped = config;
    capped.max_pages = 2;
    TestServer server2(capped);
    options = vlmv1::ConvertOptions();
    out = convert(server2.channel, options, {page(1, "P1"), page(2, "P2")});
    require(out.status.ok() && out.complete.pages_started() == 2, "exactly max_pages converts");
    out = convert(server2.channel, options, {page(1, "P1"), page(2, "P2"), page(3, "P3")});
    require(out.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED,
            "one over max_pages is RESOURCE_EXHAUSTED");
    server2.stop();

    // Page bytes exactly at the cap pass validation and convert.
    vlm::Config tight = config;
    const vlmv1::PageImage exact = page(1, "EXACTCAP");  // 8-byte magic + 8-byte marker
    tight.max_page_bytes = exact.png().size();
    TestServer server3(tight);
    out = convert(server3.channel, options, {exact});
    require(out.status.ok() && out.documents == 1, "page exactly at the byte cap converts");
    server3.stop();

    // Client cancellation mid-stream: the service tears down (no hang, no
    // crash) and reports CANCELLED.
    {
        TestServer server4(config);
        auto stub = vlmv1::VlmConvertService::NewStub(server4.channel);
        grpc::ClientContext context;
        auto stream = stub->ConvertPages(&context);
        vlmv1::ConvertPagesRequest request;
        *request.mutable_options() = options;
        stream->Write(request);
        request.Clear();
        *request.mutable_page_image() = page(1, "SLOW");
        stream->Write(request);
        context.TryCancel();
        stream->WritesDone();
        vlmv1::ConvertPagesResponse event;
        while (stream->Read(&event)) {
        }
        const grpc::Status status = stream->Finish();
        require(status.error_code() == grpc::StatusCode::CANCELLED || status.ok(),
                "cancelled stream terminates: " + status.error_message());
        server4.stop();
    }
}

}  // namespace

int main() {
    vlm::set_retry_backoff_base_ms(0);
    ScriptVlm script_vlm;
    script_vlm.start();
    FakeVlm fake;
    fake.start();
    int failures = 0;
    auto run = [&](const char* name, const std::function<void()>& verify) {
        try {
            verify();
        } catch (const std::exception& error) {
            std::println(stderr, "{}: {}", name, error.what());
            failures++;
        }
    };
    run("doctags_literal_less_than", verify_doctags_literal_less_than);
    run("doctags_truncated_close", verify_doctags_truncated_close);
    run("doctags_stray_close_and_nesting", verify_doctags_stray_close_and_nesting);
    run("doctags_loc_boundaries", verify_doctags_loc_boundaries);
    run("doctags_huge_loc_crop", verify_doctags_huge_loc_crop);
    run("doctags_text_edge_cases", verify_doctags_text_edge_cases);
    run("otsl_adversarial", verify_otsl_adversarial);
    run("image_crop_adversarial", verify_image_crop_adversarial);
    run("markdown_adversarial", verify_markdown_adversarial);
    run("html_adversarial", verify_html_adversarial);
    run("client_adversarial", [&] { verify_client_adversarial(&script_vlm); });
    run("presets_adversarial", verify_presets_adversarial);
    run("service_adversarial", [&] { verify_service_adversarial(&fake); });
    script_vlm.stop();
    fake.stop();
    if (failures > 0) {
        std::println(stderr, "{} adversarial group(s) failed", failures);
        return 1;
    }
    std::println("adversarial-test passed");
    return 0;
}
