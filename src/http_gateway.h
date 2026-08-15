#pragma once

#include <string>
#include <thread>

#include <httplib.h>

#include "config.h"

namespace vlm {

class VlmConvertServiceImpl;

// HTTP/JSON front end for VlmConvertService, on its own port alongside
// gRPC. Both convert endpoints drive the same ConvertPagesCore pipeline
// the gRPC transport uses — only the framing differs: envelope plumbing
// with nlohmann/json, message bodies with protobuf's canonical proto3
// JSON (MessageToJsonString / JsonStringToMessage), never hand-mapped
// fields.
//
//   POST /v1/convert         — one JSON object in, one {"events": [...]}
//                              JSON object out (the whole stream, in order)
//   POST /v1/convert/stream  — same request; chunked NDJSON out, one
//                              ConvertPagesResponse per line as it happens
//   GET  /healthz            — 200 "ok"
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

  private:
    void handle_convert(const httplib::Request& request, httplib::Response& response);
    void handle_convert_stream(const httplib::Request& request, httplib::Response& response);

    const Config& config_;
    VlmConvertServiceImpl& service_;
    httplib::Server server_;
    std::thread thread_;
    int port_ = -1;
};

}  // namespace vlm
