#pragma once

#include <functional>
#include <string>
#include <thread>

#include <google/protobuf/message.h>
#include <grpcpp/support/status.h>
#include <httplib.h>

#include "config.h"
#include "endpoint_health.h"

namespace vlm {

class VlmConvertServiceImpl;

// HTTP/JSON front end for VlmConvertService, on its own port alongside
// gRPC. Both convert endpoints drive the same ConvertPagesCore pipeline
// the gRPC transport uses — only the framing differs. Every body is a
// message of vlm_convert_http.proto in protobuf's canonical proto3 JSON,
// parsed or printed whole by one protobuf call (JsonStringToMessage /
// MessageToJsonString): no JSON is assembled, re-dumped or mapped by
// hand.
//
//   POST /v1/convert         — one ConvertHttpRequest in, one
//                              ConvertHttpResponse out (the whole stream,
//                              in order, and the error it ended with)
//   POST /v1/convert/stream  — same request; chunked NDJSON out, one
//                              ConvertPagesResponse per line as it
//                              happens, then one ConvertHttpResponse
//                              carrying the error if the stream failed
//   GET  /healthz            — 200 "ok" while the VLM endpoint serves,
//                              503 with the reason while it does not
//
// It binds loopback unless the operator names another host, and then a
// bearer token is required (both checked at startup, see Config). Request
// bodies are capped (413 past http_max_body_bytes), and a caller that
// hangs up on /v1/convert stops its pipeline.
class HttpGateway {
  public:
    HttpGateway(const Config& config, VlmConvertServiceImpl& service);
    ~HttpGateway();

    // Binds and starts the listener thread. Port 0 picks an ephemeral
    // port (tests); port() reports the bound port. False when the bind
    // fails.
    bool start(const std::string& host, int port);
    void stop();

    int port() const { return port_; }

    // Where /healthz reads the VLM endpoint's state from (main wires the
    // EndpointMonitor in). Set before start(). Without one /healthz
    // answers for the listener alone.
    void set_health(std::function<EndpointHealth()> health) { health_ = std::move(health); }

    // Test hook: replaces protobuf's JSON printer, so a test can make a
    // message unprintable. False from it means the message could not be
    // printed. Set before start().
    using JsonPrinter = std::function<bool(const google::protobuf::Message&, std::string*)>;
    void set_json_printer(JsonPrinter printer) { printer_ = std::move(printer); }

  private:
    void handle_convert(const httplib::Request& request, httplib::Response& response);
    void handle_convert_stream(const httplib::Request& request, httplib::Response& response);
    // Prints one message as canonical proto3 JSON; false when it cannot be.
    bool print_json(const google::protobuf::Message& message, std::string* json) const;
    // The JSON of a ConvertHttpResponse that carries only this error.
    std::string error_json(grpc::StatusCode code, const std::string& message) const;

    const Config& config_;
    VlmConvertServiceImpl& service_;
    std::function<EndpointHealth()> health_;
    JsonPrinter printer_;
    httplib::Server server_;
    std::thread thread_;
    int port_ = -1;
};

}  // namespace vlm
