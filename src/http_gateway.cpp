#include "http_gateway.h"

#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include <google/protobuf/util/json_util.h>
#include <nlohmann/json.hpp>

#include "service/vlm_convert_service.h"

namespace vlm {

namespace {

namespace vlmv1 = ai::pipestream::vlm::v1;

// The gRPC status vocabulary, surfaced as strings in the JSON error body.
const char* status_code_name(grpc::StatusCode code) {
    switch (code) {
        case grpc::StatusCode::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case grpc::StatusCode::FAILED_PRECONDITION:
            return "FAILED_PRECONDITION";
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return "RESOURCE_EXHAUSTED";
        case grpc::StatusCode::UNIMPLEMENTED:
            return "UNIMPLEMENTED";
        case grpc::StatusCode::ABORTED:
            return "ABORTED";
        case grpc::StatusCode::CANCELLED:
            return "CANCELLED";
        default:
            return "INTERNAL";
    }
}

// The ConvertPages error matrix, mapped onto HTTP: the request-shape
// errors are 4xx, everything server-side collapses to 500.
int http_status_for(grpc::StatusCode code) {
    switch (code) {
        case grpc::StatusCode::INVALID_ARGUMENT:
            return 400;
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return 413;
        case grpc::StatusCode::UNIMPLEMENTED:
            return 501;
        default:
            return 500;
    }
}

std::string error_json(grpc::StatusCode code, const std::string& message) {
    nlohmann::json error = {{"code", status_code_name(code)}, {"message", message}};
    return error.dump();
}

// Envelope → the exact request sequence a gRPC client would stream:
// options first (an empty-payload message when absent, so the pipeline's
// own "first message must be ConvertOptions" fires), then one PageImage
// per array element. Message parsing is protobuf's canonical proto3 JSON.
bool parse_convert_body(const std::string& body,
                        std::vector<vlmv1::ConvertPagesRequest>* requests, std::string* error) {
    const nlohmann::json envelope = nlohmann::json::parse(body, nullptr, false);
    if (envelope.is_discarded() || !envelope.is_object()) {
        *error = "request body must be a JSON object";
        return false;
    }
    vlmv1::ConvertPagesRequest first;
    if (envelope.contains("options")) {
        const auto status =
            google::protobuf::util::JsonStringToMessage(envelope["options"].dump(),
                                                        first.mutable_options());
        if (!status.ok()) {
            *error = "options: " + std::string(status.message());
            return false;
        }
    }
    requests->push_back(std::move(first));
    if (envelope.contains("pages")) {
        if (!envelope["pages"].is_array()) {
            *error = "pages must be an array";
            return false;
        }
        size_t index = 0;
        for (const nlohmann::json& element : envelope["pages"]) {
            vlmv1::ConvertPagesRequest next;
            const auto status = google::protobuf::util::JsonStringToMessage(
                element.dump(), next.mutable_page_image());
            if (!status.ok()) {
                *error = "pages[" + std::to_string(index) + "]: " + std::string(status.message());
                return false;
            }
            requests->push_back(std::move(next));
            index++;
        }
    }
    return true;
}

// Drives ConvertPagesCore over a fully-buffered request sequence. `read`
// replays the messages in order; the pipeline still runs its worker pool,
// so events come out in completion order exactly as on the gRPC wire.
grpc::Status run_pipeline(VlmConvertServiceImpl& service,
                          std::vector<vlmv1::ConvertPagesRequest>* requests,
                          const VlmConvertServiceImpl::ConvertWrite& write,
                          const std::function<bool()>& cancelled) {
    auto next = requests->begin();
    auto read = [&](vlmv1::ConvertPagesRequest* request) {
        if (next == requests->end()) {
            return false;
        }
        *request = std::move(*next);
        ++next;
        return true;
    };
    return service.ConvertPagesCore(read, write, cancelled);
}

}  // namespace

HttpGateway::HttpGateway(const Config& config, VlmConvertServiceImpl& service)
    : config_(config), service_(service) {
    // cpp-httplib 0.53 caps request bodies at 100MB by default (0.20 had
    // no cap). The page caps here are max_page_bytes (configurable up to
    // 1GB) times the envelope, enforced per page by the pipeline — keep
    // the transport uncapped so the app-level limits stay authoritative.
    server_.set_payload_max_length((std::numeric_limits<size_t>::max)());
    server_.Get("/healthz", [](const httplib::Request& /*request*/,
                               httplib::Response& response) {
        response.set_content("ok", "text/plain");
    });
    server_.Post("/v1/convert",
                 [this](const httplib::Request& request, httplib::Response& response) {
                     handle_convert(request, response);
                 });
    server_.Post("/v1/convert/stream",
                 [this](const httplib::Request& request, httplib::Response& response) {
                     handle_convert_stream(request, response);
                 });
}

HttpGateway::~HttpGateway() { stop(); }

bool HttpGateway::start(const std::string& host, int port) {
    if (port == 0) {
        port_ = server_.bind_to_any_port(host);
    } else {
        port_ = server_.bind_to_port(host, port) ? port : -1;
    }
    if (port_ < 0) {
        return false;
    }
    thread_ = std::thread([this] { server_.listen_after_bind(); });
    server_.wait_until_ready();
    return true;
}

void HttpGateway::stop() {
    if (thread_.joinable()) {
        server_.stop();
        thread_.join();
    }
}

void HttpGateway::handle_convert(const httplib::Request& request,
                                 httplib::Response& response) {
    std::vector<vlmv1::ConvertPagesRequest> requests;
    std::string error;
    if (!parse_convert_body(request.body, &requests, &error)) {
        response.status = 400;
        response.set_content("{\"events\":[],\"error\":" +
                                 error_json(grpc::StatusCode::INVALID_ARGUMENT, error) + "}",
                             "application/json");
        return;
    }

    std::vector<std::string> events;
    bool serialize_failed = false;
    auto write = [&](const vlmv1::ConvertPagesResponse& event) {
        std::string json;
        if (!google::protobuf::util::MessageToJsonString(event, &json).ok()) {
            serialize_failed = true;
            return false;
        }
        events.push_back(std::move(json));
        return true;
    };
    grpc::Status status = run_pipeline(service_, &requests, write, nullptr);
    if (serialize_failed && status.ok()) {
        status = grpc::Status(grpc::StatusCode::INTERNAL, "event JSON serialization failed");
    }

    std::string body = "{\"events\":[";
    for (size_t i = 0; i < events.size(); i++) {
        if (i > 0) {
            body += ',';
        }
        body += events[i];
    }
    body += ']';
    if (!status.ok()) {
        // abort_on_error lands here too: the events collected so far ride
        // along with the error, as the gRPC stream would have delivered
        // them before its ABORTED trailer.
        body += ",\"error\":" + error_json(status.error_code(), status.error_message());
    }
    body += '}';
    response.status = status.ok() ? 200 : http_status_for(status.error_code());
    response.set_content(body, "application/json");
}

void HttpGateway::handle_convert_stream(const httplib::Request& request,
                                        httplib::Response& response) {
    auto requests = std::make_shared<std::vector<vlmv1::ConvertPagesRequest>>();
    std::string error;
    if (!parse_convert_body(request.body, requests.get(), &error)) {
        response.status = 400;
        response.set_content("{\"error\":" +
                                 error_json(grpc::StatusCode::INVALID_ARGUMENT, error) + "}",
                             "application/json");
        return;
    }

    // Chunked NDJSON: every event becomes one line the moment the
    // pipeline produces it — the same live per-page view gRPC clients
    // get. A mid-stream failure ends the body with one {"error": ...}
    // line; the HTTP status is already committed to 200 by then.
    response.set_chunked_content_provider(
        "application/x-ndjson",
        [this, requests](size_t /*offset*/, httplib::DataSink& sink) {
            auto write_line = [&sink](const std::string& json) {
                return sink.write(json.data(), json.size()) && sink.write("\n", 1);
            };
            auto write = [&](const vlmv1::ConvertPagesResponse& event) {
                std::string json;
                return google::protobuf::util::MessageToJsonString(event, &json).ok() &&
                       write_line(json);
            };
            const grpc::Status status =
                run_pipeline(service_, requests.get(), write,
                             [&sink] { return !sink.is_writable(); });
            if (!status.ok()) {
                write_line("{\"error\":" +
                           error_json(status.error_code(), status.error_message()) + "}");
            }
            sink.done();
            return false;
        });
}

}  // namespace vlm
