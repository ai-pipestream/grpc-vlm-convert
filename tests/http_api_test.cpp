// HTTP front-end tests over loopback with the in-process fake VLM: the
// sync /v1/convert happy path (event order, canonical proto3 JSON field
// names), the 400 error matrix, the async /v1/convert/stream NDJSON
// per-event flush (a PageDocument line must arrive while the fake VLM
// still holds back a later page), abort_on_error on both endpoints,
// healthz, the transport body cap, a caller hanging up mid-conversion, and
// the bearer token.

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <google/protobuf/util/json_util.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include "ai/pipestream/vlm/v1/vlm_convert_http.pb.h"
#include "config.h"
#include "endpoint_health.h"
#include "fixture.h"
#include "http_gateway.h"
#include "service/vlm_convert_service.h"
#include "vlm_client.h"

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

// The fake VLM, same marker dialect as the gRPC e2e test plus HOLD: the
// page parks inside the handler until release(), so the test can prove a
// faster page's NDJSON line reached the client while this one was still
// converting.
struct FakeVlm {
    httplib::Server server;
    std::jthread thread;
    int port = 0;
    std::atomic<bool> hold_released{false};
    std::atomic<bool> holding{false};
    // Calls received, and HOLD pages that saw their caller hang up.
    std::atomic<long> calls{0};
    std::atomic<long> held_disconnects{0};

    void start() {
        server.Post("/v1/chat/completions", [this](const httplib::Request& request,
                                                   httplib::Response& response) {
            calls++;
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            std::string url = body["messages"][0]["content"][1]["image_url"]["url"];
            const std::string prefix = "data:image/png;base64,";
            require(url.starts_with(prefix), "image arrives as a data URL");
            const std::string png = base64_decode(url.substr(prefix.size()));

            if (png.contains("HOLD")) {
                // Parks until released or until the caller hangs up.
                holding = true;
                while (!hold_released.load() && !request.is_connection_closed()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                if (!hold_released.load()) {
                    held_disconnects++;
                }
                holding = false;
            }
            if (png.contains("FAIL")) {
                response.status = 503;
                response.set_content("{\"error\":\"model overloaded\"}", "application/json");
                return;
            }
            // RAWTEXT pages answer outside the DocTags format, so they fail
            // to map and come back as a PageRaw.
            const std::string content =
                png.contains("RAWTEXT")
                    ? "just plain words with no markup"
                    : "<doctag>"
                      "<section_header_level_1><loc_50><loc_100><loc_400><loc_150>Fake Heading"
                      "</section_header_level_1>"
                      "<text><loc_50><loc_200><loc_400><loc_260>Fake body text.</text>"
                      "</doctag>";
            nlohmann::json reply = {
                {"choices", {{{"message", {{"role", "assistant"}, {"content", content}}}}}},
            };
            response.set_content(reply.dump(), "application/json");
        });
        port = server.bind_to_any_port("127.0.0.1");
        require(port > 0, "fake VLM bound");
        thread = std::jthread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    void release() { hold_released = true; }

    void stop() {
        release();  // never park a held page on the teardown path
        server.stop();
        thread.join();
    }

    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port); }
};

struct HttpTestServer {
    vlm::Config config;
    vlm::VlmConvertServiceImpl service;
    vlm::HttpGateway gateway;
    httplib::Client client;

    explicit HttpTestServer(vlm::Config cfg)
        : config(std::move(cfg)), service(config), gateway(config, service),
          client("127.0.0.1", [&] {
              require(gateway.start("127.0.0.1", 0), "HTTP gateway bound");
              return gateway.port();
          }()) {}

    void stop() { gateway.stop(); }
};

vlmv1::PageImage page(uint32_t page_no, const std::string& marker) {
    vlmv1::PageImage image;
    image.set_page_no(page_no);
    image.set_png(make_png(marker));
    image.set_width(1000);
    image.set_height(1000);
    return image;
}

nlohmann::json proto_json(const google::protobuf::Message& message) {
    std::string json;
    require(google::protobuf::util::MessageToJsonString(message, &json).ok(),
            "test message serializes");
    return nlohmann::json::parse(json);
}

// The request body, built as the message it is and printed by protobuf.
nlohmann::json convert_body(const vlmv1::ConvertOptions& options,
                            const std::vector<vlmv1::PageImage>& pages) {
    vlmv1::ConvertHttpRequest body;
    *body.mutable_options() = options;
    for (const vlmv1::PageImage& image : pages) {
        *body.add_pages() = image;
    }
    return proto_json(body);
}

// A response body, read back as the message it is: a body that does not
// parse as a ConvertHttpResponse is not one.
vlmv1::ConvertHttpResponse response_of(const std::string& body) {
    vlmv1::ConvertHttpResponse response;
    const auto status = google::protobuf::util::JsonStringToMessage(body, &response);
    require(status.ok(), "the body is a ConvertHttpResponse: " + std::string(status.message()) +
                             " in " + body.substr(0, 200));
    return response;
}

void verify_healthz(httplib::Client& client) {
    auto result = client.Get("/healthz");
    require(result && result->status == 200, "healthz is 200");
    require(result->body == "ok", "healthz body is ok");
}

// /healthz answers for the VLM endpoint, not for the listener: while the
// endpoint is not serving, an orchestrator reading it must not route here.
void verify_healthz_follows_the_endpoint(FakeVlm* fake) {
    vlm::Config config;
    config.endpoint = fake->endpoint();
    HttpTestServer server(config);
    std::atomic<bool> serving{false};
    server.gateway.set_health([&] {
        return serving.load()
                   ? vlm::EndpointHealth{.serving = true, .detail = "ok"}
                   : vlm::EndpointHealth{.serving = false, .detail = "endpoint unreachable"};
    });
    auto result = server.client.Get("/healthz");
    require(result && result->status == 503, "healthz is 503 while the endpoint is down");
    require(result->body.contains("endpoint unreachable"), "healthz says why: " + result->body);
    serving = true;
    result = server.client.Get("/healthz");
    require(result && result->status == 200 && result->body == "ok",
            "healthz is 200 ok once the endpoint serves");
    server.stop();
}

void verify_sync_happy_path(httplib::Client& client) {
    vlmv1::ConvertOptions options;
    options.set_concurrency(2);
    auto result = client.Post("/v1/convert", convert_body(options, {page(1, "PAGE1"), page(2, "PAGE2")}).dump(),
                              "application/json");
    require(result && result->status == 200, "sync convert is 200");
    const nlohmann::json body = nlohmann::json::parse(result->body);
    require(!body.contains("error"), "no error on the happy path");
    require(body["events"].is_array(), "events array present");
    const auto& events = body["events"];
    require(events.size() == 5, "2 started + 2 documents + complete");

    // PageStarted(1) is always first — the read loop pushes it before the
    // job reaches a worker. The rest of the middle is completion order:
    // page 1's document may legitimately precede page 2's PageStarted, so
    // the middle three are checked as a set.
    require(events[0].contains("pageStarted") && events[0]["pageStarted"]["pageNo"] == 1,
            "first event is pageStarted page 1 (proto3 JSON field names)");
    bool saw_started2 = false, saw_doc1 = false, saw_doc2 = false;
    for (size_t i = 1; i < 4; i++) {
        const auto& event = events[i];
        if (event.contains("pageStarted")) {
            require(event["pageStarted"]["pageNo"] == 2, "the other started event is page 2");
            saw_started2 = true;
            continue;
        }
        require(event.contains("pageDocument"), "middle events are started or documents");
        const auto& document = event["pageDocument"];
        require(document.contains("pageNo") && document.contains("document"),
                "pageDocument carries pageNo + document");
        require(document["document"]["texts"].size() == 2, "canned DocTags maps to two items");
        require(document["document"]["texts"][0]["sectionHeader"]["base"]["text"] ==
                    "Fake Heading",
                "mapped heading text with camelCase proto3 names");
        if (document["pageNo"] == 1) {
            saw_doc1 = true;
        }
        if (document["pageNo"] == 2) {
            saw_doc2 = true;
        }
    }
    require(saw_started2 && saw_doc1 && saw_doc2,
            "page 2 started and both pages produced a PageDocument");

    // ConvertComplete trailer is last, with the counts. Canonical proto3
    // JSON omits zero-valued fields, so pagesFailed 0 reads as absent.
    require(events[4].contains("complete"), "last event is the complete trailer");
    const auto& complete = events[4]["complete"];
    require(complete["pagesStarted"] == 2 && complete["pagesOk"] == 2 &&
                complete.value("pagesFailed", 0) == 0,
            "trailer counts");
}

void verify_sync_garbage_json(httplib::Client& client) {
    auto result = client.Post("/v1/convert", "this is not json", "application/json");
    require(result && result->status == 400, "garbage JSON is 400");
    const vlmv1::ConvertHttpResponse body = response_of(result->body);
    require(body.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_INVALID_ARGUMENT,
            "garbage JSON names INVALID_ARGUMENT");
}

void verify_sync_non_png(httplib::Client& client) {
    vlmv1::PageImage not_png = page(1, "whatever");
    not_png.set_png("plain bytes, not a png");
    vlmv1::ConvertOptions options;
    auto result =
        client.Post("/v1/convert", convert_body(options, {not_png}).dump(), "application/json");
    require(result && result->status == 400, "non-PNG page is 400");
    const vlmv1::ConvertHttpResponse body = response_of(result->body);
    require(body.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_INVALID_ARGUMENT,
            "non-PNG names INVALID_ARGUMENT");
    require(body.error().message().contains("PNG"), "the message says why");
}

// An endpoint override without the operator's opt-in is PERMISSION_DENIED,
// which the error matrix maps to 403.
void verify_sync_override_refused(httplib::Client& client) {
    vlmv1::ConvertOptions options;
    options.set_endpoint("http://127.0.0.1:1");
    auto result = client.Post("/v1/convert", convert_body(options, {page(1, "PAGE1")}).dump(),
                              "application/json");
    require(result && result->status == 403, "a refused override is 403");
    const vlmv1::ConvertHttpResponse body = response_of(result->body);
    require(body.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_PERMISSION_DENIED,
            "403 names PERMISSION_DENIED");
}

// Collects an NDJSON response incrementally; lines() splits what has
// arrived so far.
struct Ndjson {
    std::mutex mutex;
    std::string buffer;
    int status = 0;

    std::vector<std::string> lines() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::string> out;
        size_t start = 0;
        for (;;) {
            size_t newline = buffer.find('\n', start);
            if (newline == std::string::npos) {
                break;
            }
            if (newline > start) {
                out.push_back(buffer.substr(start, newline - start));
            }
            start = newline + 1;
        }
        return out;
    }

    bool saw(const std::string& needle) {
        std::lock_guard<std::mutex> lock(mutex);
        return buffer.contains(needle);
    }
};

// Drives /v1/convert/stream on a thread; content arrives through the
// receiver as the server flushes each line. jthread so a failed require
// mid-test unwinds into main's catch instead of aborting in ~thread.
std::jthread post_stream(httplib::Client& client, const nlohmann::json& body,
                         Ndjson& sink) {
    const std::string payload = body.dump();  // the thread outlives the caller's json
    return std::jthread([&client, payload, &sink] {
        httplib::Request request;
        request.method = "POST";
        request.path = "/v1/convert/stream";
        request.body = payload;
        request.set_header("Content-Type", "application/json");
        request.content_receiver = [&sink](const char* data, size_t length, uint64_t, uint64_t) {
            std::lock_guard<std::mutex> lock(sink.mutex);
            sink.buffer.append(data, length);
            return true;
        };
        auto result = client.send(request);
        if (result) {
            sink.status = result->status;
        }
    });
}

bool wait_until(const std::function<bool()>& condition) {
    for (int i = 0; i < 1000; i++) {  // 5s budget, ms-scale in practice
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

bool wait_for(Ndjson& sink, const std::string& needle) {
    return wait_until([&] { return sink.saw(needle); });
}

void verify_stream_flushes_per_event(httplib::Client& client, FakeVlm& fake) {
    vlmv1::ConvertOptions options;
    options.set_concurrency(2);
    Ndjson sink;
    std::jthread worker =
        post_stream(client, convert_body(options, {page(1, "PAGE1"), page(2, "HOLD2")}), sink);

    // Page 2 parks inside the fake VLM until release(), so waiting for the
    // park first is race-free. Page 1's PageDocument line must then be on
    // the wire while page 2 is still held — the per-event flush proof.
    require(wait_until([&] { return fake.holding.load(); }), "page 2 parks in the fake VLM");
    require(wait_for(sink, "\"pageDocument\""), "a PageDocument line arrives while page 2 is held");
    require(fake.holding.load(), "page 2 is still parked when page 1's line arrived");
    for (const std::string& line : sink.lines()) {
        require(nlohmann::json::parse(line, nullptr, false).is_object(),
                "every NDJSON line is a JSON object");
    }

    fake.release();
    worker.join();
    require(sink.status == 200, "stream endpoint commits to 200");
    require(wait_for(sink, "\"complete\""), "the complete trailer line arrives after release");
    const std::vector<std::string> lines = sink.lines();
    require(!lines.empty(), "lines collected");
    const nlohmann::json last = nlohmann::json::parse(lines.back());
    require(last.contains("complete"), "the trailer is the last line");
    require(last["complete"]["pagesStarted"] == 2 && last["complete"]["pagesOk"] == 2,
            "trailer counts after release");
    bool saw_doc2 = false;
    for (const std::string& line : lines) {
        const nlohmann::json event = nlohmann::json::parse(line);
        if (event.contains("pageDocument") && event["pageDocument"]["pageNo"] == 2) {
            saw_doc2 = true;
        }
    }
    require(saw_doc2, "the held page converts after release");
}

// Body-shape errors: the body is one ConvertHttpRequest in proto3 JSON,
// read by protobuf in one pass, so options must be an object, pages an
// array, and a field the message does not have is refused rather than
// read past. An empty body reaches the pipeline's own rejections.
void verify_envelope_shapes(httplib::Client& client) {
    for (const char* malformed :
         {"{\"options\":5,\"pages\":[]}", "{\"options\":{},\"pages\":5}", "[]", "\"text\"",
          "{\"options\":{\"concurrency\":\"many\"},\"pages\":[]}"}) {
        auto result = client.Post("/v1/convert", malformed, "application/json");
        require(result && result->status == 400, std::string("400 for ") + malformed);
        const vlmv1::ConvertHttpResponse body = response_of(result->body);
        require(body.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_INVALID_ARGUMENT &&
                    body.error().message().starts_with("request body is not a convert request"),
                std::string("the message says the body is not a request: ") + malformed);
        require(body.events().empty(), "a refused body produced no events");
    }

    // A field the request does not have is a mistake in the request (a
    // misspelt "pages" would otherwise convert nothing and say so only by
    // omission).
    nlohmann::json misspelt = convert_body(vlmv1::ConvertOptions(), {page(1, "PAGE1")});
    misspelt["page"] = misspelt["pages"];
    auto result = client.Post("/v1/convert", misspelt.dump(), "application/json");
    require(result && result->status == 400, "an unknown field is 400");
    require(response_of(result->body).error().message().contains("page"),
            "the message names the unknown field: " + result->body);

    // An absent options object becomes an empty-payload first message,
    // so the pipeline's own first-message rule fires.
    result = client.Post("/v1/convert", "{}", "application/json");
    require(result && result->status == 400, "empty envelope is 400");
    vlmv1::ConvertHttpResponse body = response_of(result->body);
    require(body.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_INVALID_ARGUMENT &&
                body.error().message().contains("ConvertOptions"),
            "empty envelope trips the first-message rule");

    // Options present but no pages reaches the no-pages rejection.
    result = client.Post("/v1/convert", "{\"options\":{},\"pages\":[]}",
                         "application/json");
    require(result && result->status == 400, "pageless request is 400");
    body = response_of(result->body);
    require(body.error().message().contains("no pages"),
            "pageless request reaches the pipeline's no-pages rejection");

    // The stream route refuses the same bodies the same way.
    result = client.Post("/v1/convert/stream", "{\"options\":5}", "application/json");
    require(result && result->status == 400 &&
                response_of(result->body).error().code() ==
                    vlmv1::CONVERT_HTTP_ERROR_CODE_INVALID_ARGUMENT,
            "the stream route answers a malformed body with the same error body");
}

// Both routes answer with messages: the whole /v1/convert body reads back
// as one ConvertHttpResponse whose events are the stream's own, and each
// /v1/convert/stream line as one ConvertPagesResponse.
void verify_bodies_are_messages(httplib::Client& client) {
    vlmv1::ConvertOptions options;
    auto result = client.Post("/v1/convert",
                              convert_body(options, {page(1, "PAGE1"), page(2, "RAWTEXT2")}).dump(),
                              "application/json");
    require(result && result->status == 200, "sync convert is 200");
    const vlmv1::ConvertHttpResponse body = response_of(result->body);
    require(!body.has_error(), "no error on a stream that ended OK");
    require(body.events_size() == 5 && body.events(4).has_complete(),
            "two started, a document, a raw and the trailer, trailer last");
    int documents = 0, raws = 0;
    for (const vlmv1::ConvertPagesResponse& event : body.events()) {
        documents += event.has_page_document() ? 1 : 0;
        if (event.has_page_raw()) {
            raws++;
            require(event.page_raw().error_code() == vlmv1::PAGE_ERROR_CODE_UNMAPPABLE_OUTPUT,
                    "the typed page error survives the JSON framing");
        }
    }
    require(documents == 1 && raws == 1, "one page mapped and one did not");
    require(body.events(4).complete().pages_ok() == 1 &&
                body.events(4).complete().pages_failed() == 1,
            "trailer counts");

    Ndjson sink;
    std::jthread worker = post_stream(client, convert_body(options, {page(1, "PAGE1")}), sink);
    worker.join();
    const std::vector<std::string> lines = sink.lines();
    require(lines.size() == 3, "started, document, trailer");
    for (const std::string& line : lines) {
        vlmv1::ConvertPagesResponse event;
        require(google::protobuf::util::JsonStringToMessage(line, &event).ok() &&
                    event.event_case() != vlmv1::ConvertPagesResponse::EVENT_NOT_SET,
                "every line of a stream that ended OK is one event: " + line.substr(0, 120));
    }
}

// A stream on which the endpoint served nothing is 503 with the events it
// did produce, not a 200 whose events all happen to be failures.
void verify_endpoint_failure_is_503(httplib::Client& client) {
    vlmv1::ConvertOptions options;
    auto result = client.Post("/v1/convert", convert_body(options, {page(1, "FAIL1")}).dump(),
                              "application/json");
    require(result && result->status == 503, "an endpoint that served nothing is 503, got " +
                                                 std::to_string(result ? result->status : 0));
    const vlmv1::ConvertHttpResponse body = response_of(result->body);
    require(body.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_UNAVAILABLE,
            "the error names UNAVAILABLE");
    require(body.events_size() == 3 && body.events(1).has_page_raw() &&
                body.events(1).page_raw().error_code() ==
                    vlmv1::PAGE_ERROR_CODE_ENDPOINT_HTTP_ERROR &&
                body.events(1).page_raw().endpoint_error().http_status() == 503 &&
                body.events(2).has_complete(),
            "the page failure and the trailer ride along");

    Ndjson sink;
    std::jthread worker = post_stream(client, convert_body(options, {page(1, "FAIL1")}), sink);
    worker.join();
    const std::vector<std::string> lines = sink.lines();
    require(lines.size() == 4, "started, raw, trailer, then the error line");
    require(response_of(lines.back()).error().code() ==
                vlmv1::CONVERT_HTTP_ERROR_CODE_UNAVAILABLE,
            "the stream route ends with the UNAVAILABLE error line");
}

// An event protobuf cannot print is this server's failure. It must not
// pass for the caller hanging up (CANCELLED), and on the buffered route it
// must not come back as a 200 missing an event.
void verify_unprintable_event_is_internal(FakeVlm& fake) {
    vlm::Config config;
    config.endpoint = fake.endpoint();
    HttpTestServer server(std::move(config));
    // Refuses any message that holds a PageDocument, prints the rest.
    server.gateway.set_json_printer(
        [](const google::protobuf::Message& message, std::string* json) {
            auto holds_document = [](const vlmv1::ConvertPagesResponse& event) {
                return event.has_page_document();
            };
            if (message.GetDescriptor() == vlmv1::ConvertPagesResponse::descriptor() &&
                holds_document(static_cast<const vlmv1::ConvertPagesResponse&>(message))) {
                return false;
            }
            if (message.GetDescriptor() == vlmv1::ConvertHttpResponse::descriptor()) {
                for (const auto& event :
                     static_cast<const vlmv1::ConvertHttpResponse&>(message).events()) {
                    if (holds_document(event)) {
                        return false;
                    }
                }
            }
            return google::protobuf::util::MessageToJsonString(message, json).ok();
        });
    const nlohmann::json request = convert_body(vlmv1::ConvertOptions(), {page(1, "PAGE1")});

    auto result = server.client.Post("/v1/convert", request.dump(), "application/json");
    require(result && result->status == 500, "an unprintable body is 500, got " +
                                                 std::to_string(result ? result->status : 0));
    const vlmv1::ConvertHttpResponse body = response_of(result->body);
    require(body.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_INTERNAL &&
                body.error().message().contains("serialization"),
            "the buffered route names INTERNAL: " + result->body);

    Ndjson sink;
    std::jthread worker = post_stream(server.client, request, sink);
    worker.join();
    const std::vector<std::string> lines = sink.lines();
    require(!lines.empty(), "the stream route still answers");
    const vlmv1::ConvertHttpResponse last = response_of(lines.back());
    require(last.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_INTERNAL,
            "the stream route ends with INTERNAL, not CANCELLED: " + lines.back());
    server.stop();
}

// The body is capped at the transport: past http_max_body_bytes httplib
// answers 413 having read no more than the cap, before the JSON parser or
// the pipeline holds any of it. A body at the cap still reaches the handler.
void verify_body_cap(FakeVlm& fake) {
    vlm::Config config;
    config.endpoint = fake.endpoint();
    config.http_max_body_bytes = 64 * 1024;
    HttpTestServer server(std::move(config));
    auto result = server.client.Post("/v1/convert", std::string(64 * 1024 + 1, 'x'),
                                     "application/json");
    require(result && result->status == 413, "a body over the cap is 413");
    result = server.client.Post("/v1/convert", std::string(64 * 1024, 'x'), "application/json");
    require(result && result->status == 400, "a body at the cap reaches the handler");
    require(response_of(result->body).error().code() ==
                vlmv1::CONVERT_HTTP_ERROR_CODE_INVALID_ARGUMENT,
            "the at-cap body fails as JSON, not at the transport");
    server.stop();
}

// A caller that hangs up stops the conversion, on either route: the page
// in the model is cut and the queued pages are never sent.
void verify_disconnect_stops_pipeline(FakeVlm& fake, const std::string& route) {
    vlm::Config config;
    config.endpoint = fake.endpoint();
    config.concurrency = 1;
    HttpTestServer server(std::move(config));
    fake.hold_released = false;
    const long calls_before = fake.calls.load();
    const long disconnects_before = fake.held_disconnects.load();

    vlmv1::ConvertOptions options;
    options.set_concurrency(1);
    const std::string body =
        convert_body(options, {page(1, "HOLD1"), page(2, "PAGE2"), page(3, "PAGE3")}).dump();
    httplib::Client caller("127.0.0.1", server.gateway.port());
    std::jthread request([&] { caller.Post(route, body, "application/json"); });
    const bool parked = wait_until([&] { return fake.holding.load(); });
    caller.stop();  // hang up mid-conversion
    request.join();
    const bool cut = wait_until([&] { return fake.held_disconnects.load() > disconnects_before; });
    fake.release();
    require(parked, route + ": page 1 reached the model");
    require(cut, route + ": the hang-up reached the VLM call in flight");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    require(fake.calls.load() - calls_before == 1,
            route + ": queued pages are never sent after the caller left: " +
                std::to_string(fake.calls.load() - calls_before));
    server.stop();
}

// With a token configured every convert route needs it, the check happens
// before the body is read, and health stays open.
void verify_token(FakeVlm& fake) {
    vlm::Config config;
    config.endpoint = fake.endpoint();
    config.http_token = vlm::Secret("t0ken-6d1e");
    HttpTestServer server(std::move(config));
    const std::string body = convert_body(vlmv1::ConvertOptions(), {page(1, "PAGE1")}).dump();

    auto result = server.client.Post("/v1/convert", body, "application/json");
    require(result && result->status == 401, "no token is 401");
    const vlmv1::ConvertHttpResponse refusal = response_of(result->body);
    require(refusal.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_UNAUTHENTICATED &&
                refusal.events().empty(),
            "401 names UNAUTHENTICATED in the convert response body");
    require(!result->body.contains("t0ken"), "the 401 never echoes the token");

    result = server.client.Post("/v1/convert", httplib::Headers{{"Authorization", "Bearer nope"}},
                                body, "application/json");
    require(result && result->status == 401, "a wrong token is 401");

    result = server.client.Post("/v1/convert/stream", body, "application/json");
    require(result && result->status == 401, "the stream route needs the token too");

    result = server.client.Post("/v1/convert",
                                httplib::Headers{{"Authorization", "Bearer t0ken-6d1e"}}, body,
                                "application/json");
    require(result && result->status == 200, "the right token converts");

    result = server.client.Get("/healthz");
    require(result && result->status == 200, "health needs no token");
    server.stop();
}

void verify_abort_on_error_sync(httplib::Client& client) {
    vlmv1::ConvertOptions options;
    options.set_abort_on_error(true);
    options.set_concurrency(2);
    auto result = client.Post("/v1/convert",
                              convert_body(options, {page(1, "PAGE1"), page(2, "FAIL2")}).dump(),
                              "application/json");
    require(result && result->status == 500, "abort_on_error maps ABORTED to 500");
    const vlmv1::ConvertHttpResponse body = response_of(result->body);
    require(body.error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_ABORTED,
            "the error names ABORTED");
    require(!body.events().empty(), "events collected before the abort still come back");
    bool saw_raw = false;
    for (const vlmv1::ConvertPagesResponse& event : body.events()) {
        if (event.has_page_raw() && event.page_raw().page_no() == 2) {
            saw_raw = true;
        }
    }
    require(saw_raw, "the failed page's PageRaw rides along");
}

void verify_abort_on_error_stream(httplib::Client& client) {
    vlmv1::ConvertOptions options;
    options.set_abort_on_error(true);
    options.set_concurrency(2);
    Ndjson sink;
    std::jthread worker =
        post_stream(client, convert_body(options, {page(1, "PAGE1"), page(2, "FAIL2")}), sink);
    worker.join();
    require(wait_for(sink, "\"error\""), "the error line arrives");
    const std::vector<std::string> lines = sink.lines();
    require(!lines.empty(), "lines collected");
    require(response_of(lines.back()).error().code() == vlmv1::CONVERT_HTTP_ERROR_CODE_ABORTED,
            "mid-stream failure ends the NDJSON with an error line");
}

}  // namespace

int main() {
    vlm::set_retry_backoff_base_ms(0);
    FakeVlm fake;
    fake.start();
    try {
        vlm::Config config;
        config.endpoint = fake.endpoint();
        config.concurrency = 2;
        config.vlm_timeout_seconds = 30;
        HttpTestServer server(std::move(config));

        verify_healthz(server.client);
        verify_healthz_follows_the_endpoint(&fake);
        verify_sync_happy_path(server.client);
        verify_sync_garbage_json(server.client);
        verify_sync_non_png(server.client);
        verify_sync_override_refused(server.client);
        verify_stream_flushes_per_event(server.client, fake);
        verify_envelope_shapes(server.client);
        verify_bodies_are_messages(server.client);
        verify_endpoint_failure_is_503(server.client);
        verify_abort_on_error_sync(server.client);
        verify_abort_on_error_stream(server.client);
        server.stop();

        verify_body_cap(fake);
        verify_disconnect_stops_pipeline(fake, "/v1/convert");
        verify_disconnect_stops_pipeline(fake, "/v1/convert/stream");
        verify_token(fake);
        verify_unprintable_event_is_internal(fake);
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        fake.stop();
        return 1;
    }
    fake.stop();
    std::println("http-api-test passed");
    return 0;
}
