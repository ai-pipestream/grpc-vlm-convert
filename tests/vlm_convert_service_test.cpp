// End-to-end over gRPC loopback with an in-process fake VLM HTTP server:
// canned DocTags/markdown responses, the live-stream proof (a page's
// events arrive while the upload is still open), page-failure isolation,
// abort_on_error, FAILED_PRECONDITION with no endpoint, the caps and the
// input error matrix, and GetServiceInfo.

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

// The fake VLM: an OpenAI-compatible /v1/chat/completions that answers
// from markers embedded in the uploaded PNG bytes:
//   FAIL    → HTTP 503
//   MDPAGE  → canned markdown
//   RAWTEXT → prose without markup (mapping failure for DocTags)
//   PICTURES → canned DocTags with two full-page pictures
//   else    → canned DocTags naming the marker
struct FakeVlm {
    httplib::Server server;
    std::thread thread;
    int port = 0;
    std::atomic<long> calls{0};
    // The most recent request body, for wire-level assertions (stop
    // strings, max_tokens). Guarded: the handler runs on the server
    // thread.
    std::mutex mutex;
    nlohmann::json last_body;
    // The Authorization header of the most recent request ("" when absent).
    std::string last_authorization;
    // HOLD pages: how many are parked now, whether they may go, and how
    // many saw their caller hang up while parked.
    std::atomic<int> holding{0};
    std::atomic<bool> hold_released{false};
    std::atomic<long> held_disconnects{0};
    // Requests open right now, and the most that were ever open at once.
    std::atomic<int> inflight{0};
    std::atomic<int> max_inflight{0};
    // Runs as each request arrives, before it is answered (tests set it
    // between streams; guarded by `mutex`).
    std::function<void()> on_request;

    void set_on_request(std::function<void()> hook) {
        std::lock_guard<std::mutex> lock(mutex);
        on_request = std::move(hook);
    }

    nlohmann::json last_request() {
        std::lock_guard<std::mutex> lock(mutex);
        return last_body;
    }

    std::string authorization() {
        std::lock_guard<std::mutex> lock(mutex);
        return last_authorization;
    }

    void start() {
        server.Post("/v1/chat/completions", [this](const httplib::Request& request,
                                                   httplib::Response& response) {
            calls++;
            // Requests open at once, and the most ever seen.
            const int open = ++inflight;
            for (int seen = max_inflight.load();
                 open > seen && !max_inflight.compare_exchange_weak(seen, open);) {
            }
            struct Closer {
                std::atomic<int>& count;
                ~Closer() { count--; }
            } closer{inflight};
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            {
                std::lock_guard<std::mutex> lock(mutex);
                last_body = body;
                last_authorization = request.get_header_value("Authorization");
                if (on_request) {
                    on_request();
                }
            }
            std::string url = body["messages"][0]["content"][1]["image_url"]["url"];
            const std::string prefix = "data:image/png;base64,";
            require(url.starts_with(prefix), "image arrives as a data URL");
            const std::string png = base64_decode(url.substr(prefix.size()));

            if (png.contains("HOLD")) {
                // Parks like a model still generating, until released or
                // until the caller hangs up (which it records).
                holding++;
                while (!hold_released.load() && !request.is_connection_closed()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                if (!hold_released.load()) {
                    held_disconnects++;
                }
                holding--;
            }
            if (png.contains("SLOW")) {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }
            if (png.contains("FAIL")) {
                response.status = 503;
                response.set_content("{\"error\":\"model overloaded\"}", "application/json");
                return;
            }
            std::string content;
            if (png.contains("MDPAGE")) {
                content = "# Converted Page\n\nA markdown paragraph.\n";
            } else if (png.contains("RAWTEXT")) {
                content = "just plain words with no markup";
            } else if (png.contains("PICTURES")) {
                content =
                    "<doctag>"
                    "<picture><loc_0><loc_0><loc_500><loc_500></picture>"
                    "<picture><loc_0><loc_0><loc_500><loc_500></picture>"
                    "</doctag>";
            } else if (png.contains("WIDETABLE")) {
                // A table row past the column cap, as a looping model emits.
                content = "<doctag><otsl>";
                for (int i = 0; i < 300; i++) {
                    content += "<fcel>x";
                }
                content += "<nl></otsl></doctag>";
            } else {
                content =
                    "<doctag>"
                    "<section_header_level_1><loc_50><loc_100><loc_400><loc_150>Fake Heading"
                    "</section_header_level_1>"
                    "<text><loc_50><loc_200><loc_400><loc_260>Fake body text.</text>"
                    "</doctag>";
            }
            // Alternates come back only when the request asked for them,
            // so a fragment carrying them proves the key reached the wire.
            nlohmann::json alternates = nlohmann::json::array();
            if (body.is_object() && body.contains("top_logprobs")) {
                alternates = {{{"token", "a"}, {"logprob", -0.1}},
                              {{"token", "e"}, {"logprob", -3.0}}};
            }
            nlohmann::json reply = {
                {"model", "served-granite"},
                {"usage", {{"prompt_tokens", 1200}, {"completion_tokens", 512}}},
                {"choices",
                 {{{"message", {{"role", "assistant"}, {"content", content}}},
                   {"finish_reason", "length"},
                   {"logprobs",
                    {{"content",
                      {{{"token", "a"}, {"logprob", -0.1}, {"top_logprobs", alternates}},
                       {{"token", "b"}, {"logprob", -0.2}}}}}}}}},
            };
            response.set_content(reply.dump(), "application/json");
        });
        port = server.bind_to_any_port("127.0.0.1");
        require(port > 0, "fake VLM bound");
        thread = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    void stop() {
        hold_released = true;  // never park a held page on the teardown path
        server.stop();
        thread.join();
    }

    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port); }
};

bool wait_until(const std::function<bool()>& condition) {
    for (int i = 0; i < 300; i++) {  // 3 s budget, ms-scale in practice
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

struct Collected {
    std::vector<uint32_t> started;
    std::vector<vlmv1::PageDocument> documents;
    std::vector<vlmv1::PageRaw> raws;
    bool got_complete = false;
    bool complete_last = true;
    vlmv1::ConvertComplete complete;
    // Set when a PageDocument arrived before the upload finished — the
    // wire-level proof the server is not batching.
    bool document_before_upload_done = false;
    grpc::Status status;
};

void consume(Collected& out, const vlmv1::ConvertPagesResponse& event, bool upload_done) {
    if (out.got_complete) {
        out.complete_last = false;  // something followed the trailer
    }
    switch (event.event_case()) {
        case vlmv1::ConvertPagesResponse::kPageStarted:
            out.started.push_back(event.page_started().page_no());
            break;
        case vlmv1::ConvertPagesResponse::kPageDocument:
            if (!upload_done) {
                out.document_before_upload_done = true;
            }
            out.documents.push_back(event.page_document());
            break;
        case vlmv1::ConvertPagesResponse::kPageRaw:
            out.raws.push_back(event.page_raw());
            break;
        case vlmv1::ConvertPagesResponse::kComplete:
            out.got_complete = true;
            out.complete = event.complete();
            break;
        default:
            break;
    }
}

// Drives one ConvertPages stream. hold_back > 0 withholds that many tail
// pages until a PageDocument (or stream end) has been observed, proving
// conversion runs during the upload.
Collected convert(const std::shared_ptr<grpc::Channel>& channel,
                  const vlmv1::ConvertOptions& options,
                  const std::vector<vlmv1::PageImage>& pages, size_t hold_back = 0,
                  bool skip_options = false) {
    Collected out;
    auto stub = vlmv1::VlmConvertService::NewStub(channel);
    grpc::ClientContext context;
    auto stream = stub->ConvertPages(&context);

    vlmv1::ConvertPagesRequest request;
    if (!skip_options) {
        *request.mutable_options() = options;
        stream->Write(request);
    }
    size_t send_now = pages.size() > hold_back ? pages.size() - hold_back : 0;
    for (size_t i = 0; i < send_now; i++) {
        request.Clear();
        *request.mutable_page_image() = pages[i];
        if (!stream->Write(request)) {
            break;
        }
    }

    vlmv1::ConvertPagesResponse event;
    if (hold_back > 0) {
        // Read until the server proves it converts what it already has.
        // Whoever turns ConvertPages back into a batch deadlocks this
        // loop, visibly, under the ctest timeout.
        while (!out.document_before_upload_done && !out.raws.size() && stream->Read(&event)) {
            consume(out, event, /*upload_done=*/false);
        }
        for (size_t i = send_now; i < pages.size(); i++) {
            request.Clear();
            *request.mutable_page_image() = pages[i];
            stream->Write(request);
        }
    }
    stream->WritesDone();
    while (stream->Read(&event)) {
        consume(out, event, /*upload_done=*/true);
    }
    out.status = stream->Finish();
    return out;
}

vlmv1::PageImage page(uint32_t page_no, const std::string& marker) {
    vlmv1::PageImage image;
    image.set_page_no(page_no);
    image.set_png(make_png(marker));
    image.set_width(1000);
    image.set_height(1000);
    return image;
}

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

// A stream driven the way a real client drives it: the upload runs on its
// own thread (a server applying back-pressure blocks it) and the events are
// collected on another, so a test can look at what has arrived while the
// stream is still running. upload() may be called in steps.
struct LiveStream {
    std::unique_ptr<vlmv1::VlmConvertService::Stub> stub;
    grpc::ClientContext context;
    std::unique_ptr<grpc::ClientReaderWriter<vlmv1::ConvertPagesRequest,
                                             vlmv1::ConvertPagesResponse>>
        stream;
    std::mutex mutex;
    std::vector<uint32_t> started;
    int documents = 0;
    int raws = 0;
    std::jthread uploader;
    std::jthread reader;

    LiveStream(const std::shared_ptr<grpc::Channel>& channel, const vlmv1::ConvertOptions& options)
        : stub(vlmv1::VlmConvertService::NewStub(channel)) {
        stream = stub->ConvertPages(&context);
        vlmv1::ConvertPagesRequest request;
        *request.mutable_options() = options;
        stream->Write(request);
        reader = std::jthread([this] {
            vlmv1::ConvertPagesResponse event;
            while (stream->Read(&event)) {
                std::lock_guard<std::mutex> lock(mutex);
                if (event.has_page_started()) {
                    started.push_back(event.page_started().page_no());
                } else if (event.has_page_document()) {
                    documents++;
                } else if (event.has_page_raw()) {
                    raws++;
                }
            }
        });
    }

    // Sends the pages on the uploader thread (after any earlier upload),
    // then half-closes when `last` is set.
    void upload(std::vector<vlmv1::PageImage> pages, bool last) {
        if (uploader.joinable()) {
            uploader.join();
        }
        uploader = std::jthread([this, pages = std::move(pages), last] {
            vlmv1::ConvertPagesRequest request;
            for (const vlmv1::PageImage& image : pages) {
                request.Clear();
                *request.mutable_page_image() = image;
                if (!stream->Write(request)) {
                    return;
                }
            }
            if (last) {
                stream->WritesDone();
            }
        });
    }

    size_t started_count() {
        std::lock_guard<std::mutex> lock(mutex);
        return started.size();
    }

    grpc::Status finish() {
        if (uploader.joinable()) {
            uploader.join();
        }
        reader.join();
        return stream->Finish();
    }
};

void verify_streaming_and_failure_isolation(const std::shared_ptr<grpc::Channel>& channel) {
    vlmv1::ConvertOptions options;  // defaults: granite-docling, DocTags
    options.set_concurrency(2);
    std::vector<vlmv1::PageImage> pages = {page(1, "PAGE1"), page(2, "FAIL2"), page(3, "PAGE3")};
    Collected out = convert(channel, options, pages, /*hold_back=*/2);

    require(out.document_before_upload_done,
            "a PageDocument must arrive before the upload is finished");
    require(out.status.ok(), "stream with one failed page still completes OK: " +
                                 out.status.error_message());
    require(out.documents.size() == 2, "pages 1 and 3 convert");
    require(out.raws.size() == 1, "page 2 reports a failure event");
    require(out.raws[0].page_no() == 2, "the failed event is page 2");
    require(!out.raws[0].error().empty(), "the failure event carries the endpoint error");
    require(out.raws[0].error().contains("503"), "the error names HTTP 503");
    for (const vlmv1::PageDocument& document : out.documents) {
        require(document.page_no() == 1 || document.page_no() == 3, "converted pages are 1, 3");
        require(document.document().texts_size() == 2, "canned DocTags maps to two items");
        const auto& base = document.document().texts(0).section_header().base();
        require(base.text() == "Fake Heading", "mapped heading text");
        require(base.prov(0).bbox().coord_origin() == ai::pipestream::document::v1::
                    COORD_ORIGIN_TOPLEFT,
                "DocTags boxes are TOPLEFT");
        require(base.source(0).collector().collector() == "vlm-convert", "collector tagged");
        require(base.source(0).collector().model() == "ibm-granite/granite-docling-258M",
                "preset model tagged");
        require(!base.source(0).collector().version().empty(),
                "the collector stamps its own version");
        // The page-mean logprob is a raw score with its kind named, not a
        // per-item confidence claim: the same number on every item of the
        // page would say a heading and a hallucination scored alike.
        require(!base.source(0).collector().has_confidence(),
                "no per-item confidence is claimed from a page-wide mean");
        require(base.source(0).collector().has_raw_score() &&
                    std::fabs(base.source(0).collector().raw_score() + 0.15) < 1e-9,
                "the raw mean token logprob rides the source unrescaled");
        require(base.source(0).collector().raw_score_kind() == "page_mean_token_logprob",
                "the raw score names its statistic and its scope");
        require(base.source(0).collector().raw_score_samples() == 2,
                "the sample count says how many tokens the mean covers");
        // Generation provenance next to the collector source: the item
        // says which model answered, from where, how the answer stopped,
        // and what it cost.
        require(base.source_size() == 2 && base.source(1).has_generation(),
                "every item carries a GenerationSource beside the collector source");
        const auto& generation = base.source(1).generation();
        require(generation.model() == "served-granite",
                "the endpoint's own model name wins over the requested one");
        require(generation.endpoint().starts_with("http://127.0.0.1:"),
                "the answering endpoint is recorded");
        require(generation.finish_reason() == "length",
                "a truncated page says so, verbatim");
        require(generation.prompt_tokens() == 1200 && generation.completion_tokens() == 512,
                "token usage rides the fragment");
    }
    require(out.started.size() == 3, "PageStarted for every page");
    require(out.got_complete && out.complete_last, "ConvertComplete is the last event");
    require(out.complete.pages_started() == 3 && out.complete.pages_ok() == 2 &&
                out.complete.pages_failed() == 1,
            "trailer counts ok and failed pages");
}

void verify_markdown_and_raw_fallback(const std::shared_ptr<grpc::Channel>& channel) {
    vlmv1::ConvertOptions options;
    options.set_preset(vlmv1::VLM_PRESET_DEEPSEEK_OCR);  // markdown default
    Collected markdown = convert(channel, options, {page(1, "MDPAGE1")});
    require(markdown.status.ok(), "markdown page OK: " + markdown.status.error_message());
    require(markdown.documents.size() == 1, "markdown page converts");
    const auto& doc = markdown.documents[0].document();
    require(doc.texts_size() == 2, "heading + paragraph");
    require(doc.texts(0).has_section_header(), "markdown heading label");
    // Markdown provenance is the full page only — never invented boxes.
    const auto& box = doc.texts(0).section_header().base().prov(0).bbox();
    require(box.l() == 0 && box.t() == 0 && box.r() == 1000 && box.b() == 1000,
            "markdown provenance is the full page");

    // The model answering outside its declared format yields PageRaw with
    // the text, not a stream failure.
    vlmv1::ConvertOptions doctags;  // DocTags default
    Collected raw = convert(channel, doctags, {page(1, "RAWTEXT1")});
    require(raw.status.ok(), "mapping failure is not a stream failure");
    require(raw.documents.empty() && raw.raws.size() == 1, "PageRaw fallback");
    require(raw.raws[0].text().contains("plain words"),
            "PageRaw keeps the model text");
    require(!raw.raws[0].error().empty(), "PageRaw names the mapping failure");
    require(raw.complete.pages_failed() == 1, "trailer counts the failed page");
}

// Per-preset generation parameters reach the wire: granite-docling and
// smoldocling carry their docling stop strings and token budgets;
// presets without stop strings omit the parameter.
void verify_stop_and_max_tokens(const std::shared_ptr<grpc::Channel>& channel, FakeVlm* fake) {
    vlmv1::ConvertOptions granite;  // default preset
    Collected out = convert(channel, granite, {page(1, "PAGE1")});
    require(out.status.ok(), "granite page OK: " + out.status.error_message());
    nlohmann::json body = fake->last_request();
    require(body["max_tokens"] == 8192, "granite-docling max_tokens is 8192");
    require(body["stop"].is_array() && body["stop"].size() == 2 &&
                body["stop"][0] == "</doctag>" && body["stop"][1] == "<|end_of_text|>",
            "granite-docling stop strings on the wire");

    vlmv1::ConvertOptions smol;
    smol.set_preset(vlmv1::VLM_PRESET_SMOLDOCLING);
    out = convert(channel, smol, {page(1, "PAGE1")});
    require(out.status.ok(), "smoldocling page OK: " + out.status.error_message());
    body = fake->last_request();
    require(body["max_tokens"] == 4096, "smoldocling max_tokens is 4096");
    require(body["stop"].is_array() && body["stop"].size() == 2 &&
                body["stop"][0] == "</doctag>" && body["stop"][1] == "<end_of_utterance>",
            "smoldocling stop strings on the wire");

    vlmv1::ConvertOptions markdown;
    markdown.set_preset(vlmv1::VLM_PRESET_DEEPSEEK_OCR);
    out = convert(channel, markdown, {page(1, "MDPAGE1")});
    require(out.status.ok(), "markdown page OK: " + out.status.error_message());
    body = fake->last_request();
    require(body["max_tokens"] == 4096, "markdown presets keep the 4096 default");
    require(!body.contains("stop"), "presets without stop strings omit the parameter");
}

// Alternate readings: one JSON key out, hypotheses back on the fragment's
// body group. Nothing asks for them by default, and nothing carries them
// when nothing asked.
void verify_alternatives(const std::shared_ptr<grpc::Channel>& channel, FakeVlm* fake) {
    vlmv1::ConvertOptions silent;
    Collected out = convert(channel, silent, {page(1, "PAGE1")});
    require(out.status.ok(), "default page OK: " + out.status.error_message());
    require(!fake->last_request().contains("top_logprobs"),
            "no alternates are asked for by default");
    require(!out.documents[0].document().body().meta().has_alternatives(),
            "a page nobody asked alternates for carries none");

    vlmv1::ConvertOptions nbest;
    nbest.set_top_logprobs(2);
    out = convert(channel, nbest, {page(1, "PAGE1")});
    require(out.status.ok(), "n-best page OK: " + out.status.error_message());
    require(fake->last_request()["top_logprobs"] == 2, "top_logprobs reaches the wire");
    const auto& alternatives = out.documents[0].document().body().meta().alternatives();
    require(alternatives.hypotheses_size() == 2, "both alternates ride the fragment");
    require(alternatives.created_by() == "served-granite",
            "the alternates name the model that offered them");
    require(alternatives.hypotheses(0).text() == "a" &&
                alternatives.hypotheses(1).text() == "e",
            "alternates keep generation order");
    require(alternatives.hypotheses(1).raw_score_kind() == "token_logprob" &&
                std::fabs(alternatives.hypotheses(1).raw_score() + 3.0) < 1e-9,
            "an alternate carries its unrescaled score with the kind named");
    require(!alternatives.hypotheses(0).has_range(),
            "no range is claimed: nothing maps a token back to an item");
    // Page-scoped data stays on the page container, off the items.
    require(!out.documents[0]
                 .document()
                 .texts(0)
                 .section_header()
                 .base()
                 .meta()
                 .has_alternatives(),
            "items do not each claim the page's alternates");

    // Above the endpoint ceiling the RPC fails before a page is paid for.
    vlmv1::ConvertOptions too_many;
    too_many.set_top_logprobs(21);
    out = convert(channel, too_many, {page(1, "PAGE1")});
    require(out.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "top_logprobs above 20 is INVALID_ARGUMENT");
}

// What the mapper cut to a cap rides beside the fragment on the wire; a
// page with nothing cut carries no warning.
void verify_page_warnings(const std::shared_ptr<grpc::Channel>& channel) {
    vlmv1::ConvertOptions options;  // DocTags
    Collected out = convert(channel, options, {page(1, "WIDETABLE1"), page(2, "PAGE2")});
    require(out.status.ok() && out.documents.size() == 2, "both pages convert: " +
                                                            out.status.error_message());
    for (const vlmv1::PageDocument& document : out.documents) {
        if (document.page_no() == 2) {
            require(document.warnings_size() == 0, "an uncut page carries no warning");
            continue;
        }
        require(document.warnings_size() == 1, "the cut table is reported");
        const vlmv1::PageWarning& warning = document.warnings(0);
        require(warning.code() == vlmv1::PAGE_WARNING_CODE_TABLE_TRUNCATED &&
                    warning.ref() == "#/tables/0",
                "the warning names the code and the table");
        require(document.document().tables(0).data().num_cols() == 250,
                "the table keeps the columns within the cap");
    }
}

// With logprobs off (an endpoint that rejects them), calls omit the
// parameter, and a request for alternates, which need them, fails before a
// page is paid for.
void verify_logprobs_off(FakeVlm* fake) {
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.request_logprobs = false;
    TestServer server(config);
    Collected out = convert(server.channel, vlmv1::ConvertOptions(), {page(1, "PAGE1")});
    require(out.status.ok(), "a page converts without logprobs: " + out.status.error_message());
    require(!fake->last_request().contains("logprobs"), "the parameter is omitted");

    vlmv1::ConvertOptions nbest;
    nbest.set_top_logprobs(2);
    const long calls_before = fake->calls.load();
    out = convert(server.channel, nbest, {page(1, "PAGE1")});
    require(out.status.error_code() == grpc::StatusCode::FAILED_PRECONDITION,
            "alternates without logprobs is FAILED_PRECONDITION");
    require(fake->calls.load() == calls_before, "no page was paid for");
    server.stop();
}

void verify_abort_on_error(const std::shared_ptr<grpc::Channel>& channel) {    vlmv1::ConvertOptions options;
    options.set_abort_on_error(true);
    Collected out = convert(channel, options, {page(1, "PAGE1"), page(2, "FAIL2")});
    require(out.status.error_code() == grpc::StatusCode::ABORTED,
            "abort_on_error turns a page failure into ABORTED");
}

void verify_error_matrix(const std::shared_ptr<grpc::Channel>& channel) {
    vlmv1::ConvertOptions options;

    Collected out = convert(channel, options, {page(1, "PAGE1")}, 0, /*skip_options=*/true);
    require(out.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "missing options is INVALID_ARGUMENT");

    vlmv1::PageImage bad = page(0, "PAGE1");
    out = convert(channel, options, {bad});
    require(out.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "page_no 0 is INVALID_ARGUMENT");

    vlmv1::PageImage not_png = page(1, "whatever");
    not_png.set_png("plain bytes, not a png");
    out = convert(channel, options, {not_png});
    require(out.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "non-PNG page bytes are INVALID_ARGUMENT");

    // Overrides are off by default: even a malformed one is refused for
    // being an override (verify_api_key_and_overrides covers the opt-in).
    vlmv1::ConvertOptions bad_endpoint;
    bad_endpoint.set_endpoint("not-a-url");
    out = convert(channel, bad_endpoint, {page(1, "PAGE1")});
    require(out.status.error_code() == grpc::StatusCode::PERMISSION_DENIED,
            "an endpoint override without the operator opt-in is PERMISSION_DENIED");

    // Repeated options and PDF input, driven by hand.
    auto stub = vlmv1::VlmConvertService::NewStub(channel);
    {
        grpc::ClientContext context;
        auto stream = stub->ConvertPages(&context);
        vlmv1::ConvertPagesRequest request;
        *request.mutable_options() = options;
        stream->Write(request);
        stream->Write(request);  // options again
        stream->WritesDone();
        vlmv1::ConvertPagesResponse event;
        while (stream->Read(&event)) {
        }
        require(stream->Finish().error_code() == grpc::StatusCode::INVALID_ARGUMENT,
                "repeated options is INVALID_ARGUMENT");
    }
    {
        grpc::ClientContext context;
        auto stream = stub->ConvertPages(&context);
        vlmv1::ConvertPagesRequest request;
        *request.mutable_options() = options;
        stream->Write(request);
        request.Clear();
        request.mutable_pdf_chunk()->set_data("%PDF-1.4 fake");
        stream->Write(request);
        stream->WritesDone();
        vlmv1::ConvertPagesResponse event;
        while (stream->Read(&event)) {
        }
        require(stream->Finish().error_code() == grpc::StatusCode::UNIMPLEMENTED,
                "PDF input without a rasterizer is UNIMPLEMENTED");
    }
}

// The operator's API key reaches the operator's endpoint and nothing else;
// a request naming another endpoint is refused unless the operator opted
// in, and even then never carries the key.
void verify_api_key_and_overrides(FakeVlm* fake) {
    const std::string key = "vlm-key-5f2a9c";
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.vlm_api_key = vlm::Secret(key);
    {
        TestServer server(config);
        vlmv1::ConvertOptions options;
        Collected out = convert(server.channel, options, {page(1, "PAGE1")});
        require(out.status.ok() && out.documents.size() == 1,
                "keyed endpoint converts: " + out.status.error_message());
        require(fake->authorization() == "Bearer " + key,
                "the configured endpoint gets the bearer key");

        // Naming the configured endpoint is not an override, in any
        // spelling that reaches the same URL.
        for (const std::string& same :
             {fake->endpoint(), fake->endpoint() + "/", fake->endpoint() + "/v1",
              fake->endpoint() + "/v1/chat/completions"}) {
            options.set_endpoint(same);
            out = convert(server.channel, options, {page(1, "PAGE1")});
            require(out.status.ok(), "naming the configured endpoint as " + same +
                                         " is allowed: " + out.status.error_message());
            require(fake->authorization() == "Bearer " + key,
                    "the key still goes to the configured endpoint");
        }

        // Any other endpoint is an override, refused before a call is made:
        // another port, another host, the same server under another query.
        const long calls_before = fake->calls.load();
        for (const std::string& other :
             {std::string("http://127.0.0.1:1"),
              "http://127.0.0.2:" + std::to_string(fake->port),
              fake->endpoint() + "?tenant=b"}) {
            options.set_endpoint(other);
            out = convert(server.channel, options, {page(1, "PAGE1")});
            require(out.status.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                    "an override without the opt-in is PERMISSION_DENIED: " + other);
            require(out.status.error_message().contains("GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE"),
                    "the refusal names the opt-in");
        }
        require(fake->calls.load() == calls_before, "a refused override reaches no endpoint");

        // GetServiceInfo never carries the key.
        auto stub = vlmv1::VlmConvertService::NewStub(server.channel);
        grpc::ClientContext context;
        vlmv1::GetServiceInfoResponse info;
        require(stub->GetServiceInfo(&context, vlmv1::GetServiceInfoRequest(), &info).ok(),
                "GetServiceInfo OK");
        require(!info.DebugString().contains(key), "GetServiceInfo does not carry the key");
        server.stop();
    }

    config.allow_endpoint_override = true;
    {
        TestServer server(config);
        // Same server, different URL (a query the fake ignores): an
        // override, so no key.
        vlmv1::ConvertOptions options;
        options.set_endpoint(fake->endpoint() + "?tenant=b");
        Collected out = convert(server.channel, options, {page(1, "PAGE1")});
        require(out.status.ok(), "an allowed override converts: " + out.status.error_message());
        require(fake->authorization().empty(), "an override never receives the operator's key");

        vlmv1::ConvertOptions malformed;
        malformed.set_endpoint("not-a-url");
        out = convert(server.channel, malformed, {page(1, "PAGE1")});
        require(out.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
                "a malformed allowed override is INVALID_ARGUMENT");
        server.stop();
    }
}

// A client that cancels (or whose deadline passes) while its page is still
// in the model gets the VLM call cut: the endpoint sees the hang-up and the
// RPC thread is released, instead of waiting out the call and its retries.
void verify_cancel_reaches_inflight_call(FakeVlm* fake) {
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.concurrency = 1;
    config.vlm_timeout_seconds = 60;
    TestServer server(config);
    fake->hold_released = false;
    const long disconnects_before = fake->held_disconnects.load();

    auto stub = vlmv1::VlmConvertService::NewStub(server.channel);
    grpc::ClientContext context;
    auto stream = stub->ConvertPages(&context);
    vlmv1::ConvertPagesRequest request;
    *request.mutable_options() = vlmv1::ConvertOptions();
    stream->Write(request);
    request.Clear();
    *request.mutable_page_image() = page(1, "HOLD-1");
    stream->Write(request);
    const bool parked = wait_until([&] { return fake->holding.load() > 0; });

    context.TryCancel();
    vlmv1::ConvertPagesResponse event;
    while (stream->Read(&event)) {
    }
    const grpc::Status status = stream->Finish();
    const bool cut = wait_until([&] { return fake->held_disconnects.load() > disconnects_before; });
    // Release before any assertion can throw, so a failure never leaves
    // the server waiting on a parked page.
    fake->hold_released = true;
    require(parked, "the page reached the model");
    require(status.error_code() == grpc::StatusCode::CANCELLED, "the stream is cancelled");
    require(cut, "the cancel reached the VLM call in flight");
    server.stop();
}

// Pages of one fixed size, so a byte budget translates into a page count.
std::vector<vlmv1::PageImage> sized_pages(uint32_t first, uint32_t count,
                                          const std::string& first_marker) {
    std::vector<vlmv1::PageImage> pages;
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t page_no = first + i;
        std::string marker = i == 0 ? first_marker : "PAGE-";
        marker += std::to_string(page_no % 10);
        pages.push_back(page(page_no, marker));
    }
    return pages;
}

// One stream holds at most max_stream_buffered_bytes of pages read but not
// yet answered. With a page parked in the model and room for two, the
// server admits two and reads nothing more until the budget frees, however
// much the client has sent; then every page still converts.
void verify_stream_backpressure(FakeVlm* fake) {
    const std::vector<vlmv1::PageImage> pages = sized_pages(1, 6, "HOLD-");
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.concurrency = 1;
    config.max_stream_buffered_bytes = 2 * pages[0].png().size();
    TestServer server(config);
    fake->hold_released = false;

    LiveStream live(server.channel, vlmv1::ConvertOptions());
    live.upload(pages, /*last=*/true);
    const bool parked = wait_until([&] { return fake->holding.load() > 0; });
    // Two fit; then give an unbounded reader room to run ahead.
    wait_until([&] { return live.started_count() >= 2; });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const size_t started_while_parked = live.started_count();
    fake->hold_released = true;
    const grpc::Status status = live.finish();

    require(parked, "page 1 reached the model");
    require(started_while_parked == 2,
            "the stream admits only what its budget holds while page 1 is parked: " +
                std::to_string(started_while_parked) + " started");
    require(status.ok() && live.documents == 6, "every page converts once the budget frees: " +
                                                    status.error_message());
    require(server.service->buffered_bytes().in_use() == 0,
            "every page's bytes are back in the process budget");
    server.stop();
}

// The process-wide budget holds across streams: with a parked stream
// filling it, a second stream's first page waits, then converts.
void verify_process_backpressure(FakeVlm* fake) {
    const std::vector<vlmv1::PageImage> first = sized_pages(1, 3, "HOLD-");
    const std::vector<vlmv1::PageImage> second = sized_pages(11, 2, "PAGE-");
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.concurrency = 1;
    config.max_buffered_bytes = 2 * first[0].png().size();
    TestServer server(config);
    fake->hold_released = false;

    LiveStream a(server.channel, vlmv1::ConvertOptions());
    a.upload(first, /*last=*/true);
    const bool parked = wait_until([&] { return fake->holding.load() > 0; });
    wait_until([&] { return a.started_count() >= 2; });
    LiveStream b(server.channel, vlmv1::ConvertOptions());
    b.upload(second, /*last=*/true);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const size_t a_started = a.started_count();
    const size_t b_started = b.started_count();
    fake->hold_released = true;
    const grpc::Status a_status = a.finish();
    const grpc::Status b_status = b.finish();

    require(parked, "stream A's first page reached the model");
    require(a_started == 2, "stream A fills the process budget: " + std::to_string(a_started));
    require(b_started == 0, "stream B waits for the process budget: " +
                                std::to_string(b_started));
    require(a_status.ok() && a.documents == 3, "stream A converts: " + a_status.error_message());
    require(b_status.ok() && b.documents == 2, "stream B converts: " + b_status.error_message());
    require(server.service->buffered_bytes().in_use() == 0, "the process budget drains");
    server.stop();
}

// GRPC_VLM_MAX_INFLIGHT bounds the calls every stream together has open on
// the endpoint, whatever each stream's own concurrency.
void verify_process_inflight_cap(FakeVlm* fake) {
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.concurrency = 2;
    config.max_inflight = 1;
    TestServer server(config);
    fake->max_inflight = 0;

    vlmv1::ConvertOptions options;
    options.set_concurrency(2);
    LiveStream a(server.channel, options);
    LiveStream b(server.channel, options);
    a.upload({page(1, "SLOW-1"), page(2, "SLOW-2")}, /*last=*/true);
    b.upload({page(1, "SLOW-3"), page(2, "SLOW-4")}, /*last=*/true);
    const grpc::Status a_status = a.finish();
    const grpc::Status b_status = b.finish();

    require(a_status.ok() && a.documents == 2, "stream A converts: " + a_status.error_message());
    require(b_status.ok() && b.documents == 2, "stream B converts: " + b_status.error_message());
    require(fake->max_inflight.load() == 1, "never more than one call open on the endpoint: " +
                                                std::to_string(fake->max_inflight.load()));
    require(server.service->vlm_slots().in_use() == 0, "every slot is given back");
    server.stop();
}

// A real 8-bit grayscale PNG of width x height mid-gray pixels, stored
// without compression so the test needs no encoder. stb checks neither
// chunk CRCs nor the zlib checksum, so those are zero.
std::string gray_png(uint32_t width, uint32_t height) {
    std::string png("\x89PNG\r\n\x1a\n", 8);
    auto be32 = [](std::string* out, uint32_t value) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            *out += static_cast<char>((value >> shift) & 0xFF);
        }
    };
    auto chunk = [&](const std::string& type, const std::string& data) {
        be32(&png, static_cast<uint32_t>(data.size()));
        png += type + data;
        be32(&png, 0);
    };
    std::string header;
    be32(&header, width);
    be32(&header, height);
    header += std::string("\x08\x00\x00\x00\x00", 5);  // 8-bit gray, no interlace
    chunk("IHDR", header);

    std::string row(1 + width, '\x80');
    row[0] = '\0';  // filter: none
    std::string raw;
    for (uint32_t y = 0; y < height; y++) {
        raw += row;
    }
    std::string zlib("\x78\x01", 2);
    for (size_t offset = 0; offset < raw.size(); offset += 65535) {
        const size_t length = std::min<size_t>(65535, raw.size() - offset);
        zlib += static_cast<char>(offset + length == raw.size() ? 1 : 0);  // stored block
        zlib += static_cast<char>(length & 0xFF);
        zlib += static_cast<char>(length >> 8);
        zlib += static_cast<char>(~length & 0xFF);
        zlib += static_cast<char>((~length >> 8) & 0xFF);
        zlib += raw.substr(offset, length);
    }
    zlib += std::string(4, '\0');  // adler32, unchecked
    chunk("IDAT", zlib);
    chunk("IEND", "");
    return png;
}

// The in-flight slot covers mapping as well as the call: a page's answer
// is mapped (a picture page decodes and crops its whole raster) before the
// next call may take the slot, so GRPC_VLM_MAX_INFLIGHT also bounds the
// rasters being decoded at once.
void verify_inflight_slot_covers_mapping(FakeVlm* fake) {
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.concurrency = 2;
    config.max_inflight = 1;
    TestServer server(config);
    std::mutex mutex;
    std::vector<long> finished_at_arrival;
    fake->set_on_request([&] {
        std::lock_guard<std::mutex> lock(mutex);
        finished_at_arrival.push_back(server.service->pages_finished.load());
    });

    vlmv1::PageImage pictures;
    pictures.set_page_no(1);
    pictures.set_png(gray_png(1500, 1500) + "PICTURES");
    pictures.set_width(1500);
    pictures.set_height(1500);
    vlmv1::ConvertOptions options;
    options.set_concurrency(2);
    Collected out = convert(server.channel, options, {pictures, page(2, "PAGE-2"),
                                                      page(3, "PAGE-3")});
    fake->set_on_request(nullptr);

    require(out.status.ok() && out.documents.size() == 3,
            "the pages convert: " + out.status.error_message());
    const auto picture_page =
        std::ranges::find_if(out.documents, [](const vlmv1::PageDocument& document) {
            return document.page_no() == 1;
        });
    require(picture_page != out.documents.end() &&
                picture_page->document().pictures_size() == 2 &&
                picture_page->document().pictures(0).has_image(),
            "the picture page was cropped from its decoded raster");
    std::lock_guard<std::mutex> lock(mutex);
    require(finished_at_arrival == std::vector<long>({0, 1, 2}),
            "each call starts only once the page before it was mapped");
    require(server.service->vlm_slots().in_use() == 0, "every slot is given back");
    server.stop();
}

// Once a stream is going to fail, pages nobody will receive are not paid
// for: abort_on_error stops dispatching after the first failed page, and a
// bad page mid-stream cuts the call in flight and skips the queue.
void verify_failure_stops_dispatch(FakeVlm* fake) {
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.concurrency = 1;
    TestServer server(config);

    vlmv1::ConvertOptions abort_options;
    abort_options.set_abort_on_error(true);
    long calls_before = fake->calls.load();
    Collected out = convert(server.channel, abort_options,
                            {page(1, "FAIL-1"), page(2, "PAGE-2"), page(3, "PAGE-3"),
                             page(4, "PAGE-4")});
    require(out.status.error_code() == grpc::StatusCode::ABORTED, "the stream aborts");
    require(fake->calls.load() - calls_before == 6,
            "only the failed page's 6 attempts reach the endpoint: " +
                std::to_string(fake->calls.load() - calls_before));

    fake->hold_released = false;
    const long disconnects_before = fake->held_disconnects.load();
    calls_before = fake->calls.load();
    LiveStream live(server.channel, vlmv1::ConvertOptions());
    live.upload({page(1, "HOLD-1"), page(2, "PAGE-2"), page(3, "PAGE-3")}, /*last=*/false);
    const bool parked = wait_until([&] { return fake->holding.load() > 0; });
    vlmv1::PageImage not_png = page(4, "PAGE-4");
    not_png.set_png("plain bytes, not a png");
    live.upload({not_png}, /*last=*/true);
    const grpc::Status status = live.finish();
    const bool cut = wait_until([&] { return fake->held_disconnects.load() > disconnects_before; });
    fake->hold_released = true;
    require(parked, "page 1 reached the model");
    require(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT, "the bad page fails it");
    require(cut, "the call in flight is cut");
    require(fake->calls.load() - calls_before == 1, "queued pages never reach the endpoint: " +
                                                        std::to_string(fake->calls.load() -
                                                                       calls_before));
    server.stop();
}

void verify_no_endpoint() {
    vlm::Config config;
    config.endpoint = "";
    TestServer server(config);
    vlmv1::ConvertOptions options;
    Collected out = convert(server.channel, options, {page(1, "PAGE1")});
    require(out.status.error_code() == grpc::StatusCode::FAILED_PRECONDITION,
            "no endpoint configured is FAILED_PRECONDITION");
    server.stop();
}

void verify_page_byte_cap() {
    vlm::Config config;
    config.max_page_bytes = 16;
    config.endpoint = "http://127.0.0.1:1";  // never reached: the cap fires first
    TestServer server(config);
    vlmv1::ConvertOptions options;
    Collected out = convert(server.channel, options, {page(1, "PAGE1-WELL-OVER-THE-CAP")});
    require(out.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED,
            "over-cap page is RESOURCE_EXHAUSTED");
    server.stop();
}

// The operator's inline crop cap reaches the mapper: a page whose crops
// would outgrow it keeps its pictures, attaches what fits, and says what
// it skipped, instead of sending a PageDocument the client cannot receive.
void verify_page_crop_byte_cap(FakeVlm* fake) {
    // A real 4x3 grayscale PNG (stb decodes it for the crops) with the
    // fake's marker after IEND, where decoders stop reading.
    const std::string gray = base64_decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAAAAACRn/EaAAAAF0lEQVR4nGNg4BKRY9AwsnFjCIhKyQMADI8C"
        "lWcdFq8AAAAASUVORK5CYII=");
    docv1::ImageRef full_page;
    require(vlm::mapping::crop_png_image(gray, 0, 0, 4, 3, 4, 3, &full_page),
            "the fixture raster crops");
    vlm::Config config;
    config.endpoint = fake->endpoint();
    config.max_page_crop_bytes = full_page.uri().size();  // one full-page crop fits
    TestServer server(config);
    vlmv1::PageImage image;
    image.set_page_no(1);
    image.set_png(gray + "PICTURES");
    image.set_width(4);
    image.set_height(3);
    Collected out = convert(server.channel, vlmv1::ConvertOptions(), {image});
    require(out.status.ok() && out.documents.size() == 1,
            "the page converts: " + out.status.error_message());
    const vlmv1::PageDocument& document = out.documents[0];
    require(document.document().pictures_size() == 2, "both pictures are kept");
    require(document.document().pictures(0).has_image() &&
                !document.document().pictures(1).has_image(),
            "only the crop within the configured cap is attached");
    require(document.warnings_size() == 1 &&
                document.warnings(0).code() == vlmv1::PAGE_WARNING_CODE_PICTURE_IMAGES_SKIPPED,
            "the skipped crop is reported as a typed warning");
    server.stop();
}

void verify_service_info(const std::shared_ptr<grpc::Channel>& channel,
                         const std::string& endpoint) {
    auto stub = vlmv1::VlmConvertService::NewStub(channel);
    grpc::ClientContext context;
    vlmv1::GetServiceInfoRequest request;
    vlmv1::GetServiceInfoResponse info;
    require(stub->GetServiceInfo(&context, request, &info).ok(), "GetServiceInfo OK");
    require(!info.version().empty(), "version reported");
    require(info.endpoint() == endpoint, "endpoint reported");
    require(info.presets_size() == 9, "all built-in presets reported by default");
    require(info.concurrency() > 0 && info.max_page_bytes() > 0 && info.max_pages() > 0,
            "limits reported");
    require(info.ui().title() == "VLM Convert" && info.ui().path() == "/ui/vlm-convert" &&
                info.ui().description() == "Calls an external VLM server for document conversion",
            "shared-shell ui advertisement reported");

    // A configured preset list maps known names to enums and forwards the
    // rest as raw names.
    vlm::Config config;
    config.endpoint = endpoint;
    config.presets = {"granite-docling", "unlimited-ocr"};
    TestServer server(config);
    auto stub2 = vlmv1::VlmConvertService::NewStub(server.channel);
    grpc::ClientContext context2;
    vlmv1::GetServiceInfoResponse info2;
    require(stub2->GetServiceInfo(&context2, request, &info2).ok(), "GetServiceInfo OK (2)");
    require(info2.presets_size() == 1 &&
                info2.presets(0) == vlmv1::VLM_PRESET_GRANITE_DOCLING,
            "known preset name maps to the enum");
    require(info2.raw_presets_size() == 1 && info2.raw_presets(0) == "unlimited-ocr",
            "unknown preset names are reported raw");
    server.stop();

    // The RPC is unauthenticated: an endpoint whose path and query carry a
    // tenant token is reported as scheme, host and port only.
    vlm::Config secret_path;
    secret_path.endpoint = endpoint + "/tenant/PATH-SECRET?key=QUERY-SECRET";
    TestServer server3(secret_path);
    auto stub3 = vlmv1::VlmConvertService::NewStub(server3.channel);
    grpc::ClientContext context3;
    vlmv1::GetServiceInfoResponse info3;
    require(stub3->GetServiceInfo(&context3, request, &info3).ok(), "GetServiceInfo OK (3)");
    require(info3.endpoint() == endpoint, "the reported endpoint is the origin: " +
                                              info3.endpoint());
    require(!info3.DebugString().contains("SECRET"), "no part of the token is reported");
    server3.stop();
}

}  // namespace

int main() {
    // The FAIL pages now retry per docling's policy; pin the backoff base
    // to zero so persistent-failure tests don't sleep ~3s per page.
    vlm::set_retry_backoff_base_ms(0);
    FakeVlm fake;
    fake.start();
    try {
        vlm::Config config;
        config.endpoint = fake.endpoint();
        config.concurrency = 2;
        config.vlm_timeout_seconds = 30;
        TestServer server(config);

        const long calls_before = fake.calls.load();
        verify_streaming_and_failure_isolation(server.channel);
        require(fake.calls.load() - calls_before > 3,
                "the persistently-503 page was retried: more calls than pages");
        verify_markdown_and_raw_fallback(server.channel);
        verify_stop_and_max_tokens(server.channel, &fake);
        verify_alternatives(server.channel, &fake);
        verify_page_warnings(server.channel);
        verify_logprobs_off(&fake);
        verify_abort_on_error(server.channel);
        verify_error_matrix(server.channel);
        verify_api_key_and_overrides(&fake);
        verify_cancel_reaches_inflight_call(&fake);
        verify_stream_backpressure(&fake);
        verify_process_backpressure(&fake);
        verify_process_inflight_cap(&fake);
        verify_inflight_slot_covers_mapping(&fake);
        verify_failure_stops_dispatch(&fake);
        verify_no_endpoint();
        verify_page_byte_cap();
        verify_page_crop_byte_cap(&fake);
        verify_service_info(server.channel, fake.endpoint());

        require(server.service->converted.load() > 0, "converted counter moved");
        require(server.service->rejected.load() > 0, "rejected counter moved");
        require(server.service->pages_ok.load() > 0 && server.service->pages_failed.load() > 0,
                "page counters moved");
        server.stop();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        fake.stop();
        return 1;
    }
    fake.stop();
    std::println("vlm-convert-service-test passed");
    return 0;
}
