// End-to-end over gRPC loopback with an in-process fake VLM HTTP server:
// canned DocTags/markdown responses, the live-stream proof (a page's
// events arrive while the upload is still open), page-failure isolation,
// abort_on_error, FAILED_PRECONDITION with no endpoint, the caps and the
// input error matrix, and GetServiceInfo.

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "fixture.h"
#include "service/vlm_convert_service.h"

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
//   else    → canned DocTags naming the marker
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
            const std::string prefix = "data:image/png;base64,";
            require(url.compare(0, prefix.size(), prefix) == 0, "image arrives as a data URL");
            const std::string png = base64_decode(url.substr(prefix.size()));

            if (png.find("FAIL") != std::string::npos) {
                response.status = 503;
                response.set_content("{\"error\":\"model overloaded\"}", "application/json");
                return;
            }
            std::string content;
            if (png.find("MDPAGE") != std::string::npos) {
                content = "# Converted Page\n\nA markdown paragraph.\n";
            } else if (png.find("RAWTEXT") != std::string::npos) {
                content = "just plain words with no markup";
            } else {
                content =
                    "<doctag>"
                    "<section_header_level_1><loc_50><loc_100><loc_400><loc_150>Fake Heading"
                    "</section_header_level_1>"
                    "<text><loc_50><loc_200><loc_400><loc_260>Fake body text.</text>"
                    "</doctag>";
            }
            nlohmann::json reply = {
                {"choices",
                 {{{"message", {{"role", "assistant"}, {"content", content}}},
                   {"logprobs",
                    {{"content",
                      {{{"token", "a"}, {"logprob", -0.1}},
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
        server.stop();
        thread.join();
    }

    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port); }
};

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
    require(out.raws[0].error().find("503") != std::string::npos, "the error names HTTP 503");
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
        require(base.source(0).collector().has_confidence(),
                "logprobs become CollectorSource confidence");
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
    require(raw.raws[0].text().find("plain words") != std::string::npos,
            "PageRaw keeps the model text");
    require(!raw.raws[0].error().empty(), "PageRaw names the mapping failure");
    require(raw.complete.pages_failed() == 1, "trailer counts the failed page");
}

void verify_abort_on_error(const std::shared_ptr<grpc::Channel>& channel) {
    vlmv1::ConvertOptions options;
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

    vlmv1::ConvertOptions bad_endpoint;
    bad_endpoint.set_endpoint("not-a-url");
    out = convert(channel, bad_endpoint, {page(1, "PAGE1")});
    require(out.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "malformed endpoint override is INVALID_ARGUMENT");

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

void verify_service_info(const std::shared_ptr<grpc::Channel>& channel,
                         const std::string& endpoint) {
    auto stub = vlmv1::VlmConvertService::NewStub(channel);
    grpc::ClientContext context;
    vlmv1::GetServiceInfoRequest request;
    vlmv1::GetServiceInfoResponse info;
    require(stub->GetServiceInfo(&context, request, &info).ok(), "GetServiceInfo OK");
    require(!info.version().empty(), "version reported");
    require(info.endpoint() == endpoint, "endpoint reported");
    require(info.presets_size() == 8, "all built-in presets reported by default");
    require(info.concurrency() > 0 && info.max_page_bytes() > 0 && info.max_pages() > 0,
            "limits reported");

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
}

}  // namespace

int main() {
    FakeVlm fake;
    fake.start();
    try {
        vlm::Config config;
        config.endpoint = fake.endpoint();
        config.concurrency = 2;
        config.vlm_timeout_seconds = 30;
        TestServer server(config);

        verify_streaming_and_failure_isolation(server.channel);
        verify_markdown_and_raw_fallback(server.channel);
        verify_abort_on_error(server.channel);
        verify_error_matrix(server.channel);
        verify_no_endpoint();
        verify_page_byte_cap();
        verify_service_info(server.channel, fake.endpoint());

        require(server.service->converted.load() > 0, "converted counter moved");
        require(server.service->rejected.load() > 0, "rejected counter moved");
        require(server.service->pages_ok.load() > 0 && server.service->pages_failed.load() > 0,
                "page counters moved");
        server.stop();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        fake.stop();
        return 1;
    }
    fake.stop();
    std::cout << "vlm-convert-service-test passed\n";
    return 0;
}
