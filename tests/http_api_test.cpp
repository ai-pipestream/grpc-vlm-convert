// HTTP front-end tests over loopback with the in-process fake VLM: the
// sync /v1/convert happy path (event order, canonical proto3 JSON field
// names), the 400 error matrix, the async /v1/convert/stream NDJSON
// per-event flush (a PageDocument line must arrive while the fake VLM
// still holds back a later page), abort_on_error on both endpoints, and
// healthz.

#include <atomic>
#include <chrono>
#include <condition_variable>
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

#include "config.h"
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
    std::mutex hold_mutex;
    std::condition_variable hold_cv;
    bool hold_released = false;
    std::atomic<bool> holding{false};

    void start() {
        server.Post("/v1/chat/completions", [this](const httplib::Request& request,
                                                   httplib::Response& response) {
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            std::string url = body["messages"][0]["content"][1]["image_url"]["url"];
            const std::string prefix = "data:image/png;base64,";
            require(url.starts_with(prefix), "image arrives as a data URL");
            const std::string png = base64_decode(url.substr(prefix.size()));

            if (png.contains("HOLD")) {
                holding = true;
                std::unique_lock<std::mutex> lock(hold_mutex);
                hold_cv.wait(lock, [&] { return hold_released; });
                holding = false;
            }
            if (png.contains("FAIL")) {
                response.status = 503;
                response.set_content("{\"error\":\"model overloaded\"}", "application/json");
                return;
            }
            const std::string content =
                "<doctag>"
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

    void release() {
        {
            std::lock_guard<std::mutex> lock(hold_mutex);
            hold_released = true;
        }
        hold_cv.notify_all();
    }

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

nlohmann::json convert_body(const vlmv1::ConvertOptions& options,
                            const std::vector<vlmv1::PageImage>& pages) {
    nlohmann::json body;
    body["options"] = proto_json(options);
    body["pages"] = nlohmann::json::array();
    for (const vlmv1::PageImage& image : pages) {
        body["pages"].push_back(proto_json(image));
    }
    return body;
}

void verify_healthz(httplib::Client& client) {
    auto result = client.Get("/healthz");
    require(result && result->status == 200, "healthz is 200");
    require(result->body == "ok", "healthz body is ok");
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
    const nlohmann::json body = nlohmann::json::parse(result->body);
    require(body["error"]["code"] == "INVALID_ARGUMENT", "garbage JSON names INVALID_ARGUMENT");
}

void verify_sync_non_png(httplib::Client& client) {
    vlmv1::PageImage not_png = page(1, "whatever");
    not_png.set_png("plain bytes, not a png");
    vlmv1::ConvertOptions options;
    auto result =
        client.Post("/v1/convert", convert_body(options, {not_png}).dump(), "application/json");
    require(result && result->status == 400, "non-PNG page is 400");
    const nlohmann::json body = nlohmann::json::parse(result->body);
    require(body["error"]["code"] == "INVALID_ARGUMENT", "non-PNG names INVALID_ARGUMENT");
    require(body["error"]["message"].get<std::string>().contains("PNG"),
            "the message says why");
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

// Envelope-shape errors from the JSON layer: options must be an object
// (proto3 JSON), pages must be an array, and an empty envelope reaches
// the pipeline's own "no pages" rejection.
void verify_envelope_shapes(httplib::Client& client) {
    auto result = client.Post("/v1/convert", "{\"options\":5,\"pages\":[]}",
                              "application/json");
    require(result && result->status == 400, "non-object options is 400");
    nlohmann::json body = nlohmann::json::parse(result->body);
    require(body["error"]["code"] == "INVALID_ARGUMENT", "non-object options names the code");
    require(body["error"]["message"].get<std::string>().starts_with("options"),
            "the message points at options");

    result = client.Post("/v1/convert", "{\"pages\":{}}", "application/json");
    require(result && result->status == 400, "non-array pages is 400");
    body = nlohmann::json::parse(result->body);
    require(body["error"]["message"] == "pages must be an array", "the message points at pages");

    // An absent options object becomes an empty-payload first message,
    // so the pipeline's own first-message rule fires.
    result = client.Post("/v1/convert", "{}", "application/json");
    require(result && result->status == 400, "empty envelope is 400");
    body = nlohmann::json::parse(result->body);
    require(body["error"]["code"] == "INVALID_ARGUMENT" &&
                body["error"]["message"].get<std::string>().contains("ConvertOptions"),
            "empty envelope trips the first-message rule");

    // Options present but no pages reaches the no-pages rejection.
    result = client.Post("/v1/convert", "{\"options\":{},\"pages\":[]}",
                         "application/json");
    require(result && result->status == 400, "pageless request is 400");
    body = nlohmann::json::parse(result->body);
    require(body["error"]["message"].get<std::string>().contains("no pages"),
            "pageless request reaches the pipeline's no-pages rejection");
}

// cpp-httplib 0.53 introduced a 100MB default request-body cap (0.20 had
// none); the gateway removes it so the configurable app-level page caps
// stay authoritative. A body just over 100MB must reach the JSON parser
// (400 from the handler), never die at the transport (413).
void verify_transport_uncapped(httplib::Client& client) {
    client.set_write_timeout(60, 0);
    const std::string big(100 * 1024 * 1024 + 1, 'x');
    auto result = client.Post("/v1/convert", big, "application/json");
    require(result && result->status == 400, "an over-100MB body reaches the handler");
    const nlohmann::json body = nlohmann::json::parse(result->body);
    require(body["error"]["code"] == "INVALID_ARGUMENT",
            "the oversized body fails as JSON, not as a transport cap");
}

void verify_abort_on_error_sync(httplib::Client& client) {
    vlmv1::ConvertOptions options;
    options.set_abort_on_error(true);
    options.set_concurrency(2);
    auto result = client.Post("/v1/convert",
                              convert_body(options, {page(1, "PAGE1"), page(2, "FAIL2")}).dump(),
                              "application/json");
    require(result && result->status == 500, "abort_on_error maps ABORTED to 500");
    const nlohmann::json body = nlohmann::json::parse(result->body);
    require(body["error"]["code"] == "ABORTED", "the error names ABORTED");
    require(body["events"].is_array() && !body["events"].empty(),
            "events collected before the abort still come back");
    bool saw_raw = false;
    for (const auto& event : body["events"]) {
        if (event.contains("pageRaw") && event["pageRaw"]["pageNo"] == 2) {
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
    const nlohmann::json last = nlohmann::json::parse(lines.back());
    require(last.contains("error") && last["error"]["code"] == "ABORTED",
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
        verify_sync_happy_path(server.client);
        verify_sync_garbage_json(server.client);
        verify_sync_non_png(server.client);
        verify_stream_flushes_per_event(server.client, fake);
        verify_envelope_shapes(server.client);
        verify_transport_uncapped(server.client);
        verify_abort_on_error_sync(server.client);
        verify_abort_on_error_stream(server.client);

        server.stop();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        fake.stop();
        return 1;
    }
    fake.stop();
    std::println("http-api-test passed");
    return 0;
}
