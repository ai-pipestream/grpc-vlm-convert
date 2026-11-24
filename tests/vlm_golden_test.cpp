// Golden test against a real VLM endpoint: one real page raster through
// the configured preset, asserting labels + ordering + counts rather than
// full message equality (model output drifts between versions).
//
// Needs GRPC_VLM_TEST_ENDPOINT and GRPC_VLM_TEST_PNG (a real page raster
// file); skips with 77 otherwise, so CI without a VLM server stays green.
//
// A skip is only honest where no VLM server is expected. On a host that
// has one, GRPC_VLM_TEST_REQUIRE_ENDPOINT=1 turns a missing endpoint or
// raster into a failure, so a suite that ran no real model cannot pass
// there. GRPC_VLM_TEST_REQUIRE_DEVICE (e.g. "cuda", "xpu") additionally
// asks the endpoint's GET /health which device the model is on, as the
// north-micro-vision server reports it, and fails unless it starts with
// that name: a model that fell back to CPU answers the golden page just
// as well, only slower.

#include <grpcpp/grpcpp.h>

#include <fstream>
#include <string>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "fixture.h"
#include "service/vlm_convert_service.h"
#include "vlm_client.h"

namespace vlmv1 = ai::pipestream::vlm::v1;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// What to do when the test cannot run: skip where a VLM server is
// optional, fail where GRPC_VLM_TEST_REQUIRE_ENDPOINT says one must be
// there.
int cannot_run(const std::string& why) {
    if (env_or_null("GRPC_VLM_TEST_REQUIRE_ENDPOINT") != nullptr) {
        std::println(stderr,
                     "FAIL: {}, and GRPC_VLM_TEST_REQUIRE_ENDPOINT is set: this host must run "
                     "the golden test, not skip it",
                     why);
        return 1;
    }
    return skip(why);
}

// The device the endpoint's model runs on, as its GET /health reports it
// ("cuda:0", "xpu", "cpu"). Throws when the endpoint has no such route or
// does not say.
std::string endpoint_device(const std::string& endpoint) {
    const std::string origin = vlm::endpoint_origin(endpoint);
    httplib::Client client(origin);
    client.set_connection_timeout(10);
    client.set_read_timeout(30);
    const auto reply = client.Get("/health");
    require(reply && reply->status == 200,
            "GET " + origin + "/health answers 200 (GRPC_VLM_TEST_REQUIRE_DEVICE needs it)");
    const nlohmann::json health = nlohmann::json::parse(reply->body, nullptr, false);
    require(health.is_object() && health.contains("device") && health["device"].is_string(),
            "the endpoint's /health names the model's device");
    return health["device"].get<std::string>();
}

}  // namespace

int main() try {
    const char* endpoint = env_or_null("GRPC_VLM_TEST_ENDPOINT");
    const char* png_path = env_or_null("GRPC_VLM_TEST_PNG");
    if (endpoint == nullptr) {
        return cannot_run("no VLM endpoint at $GRPC_VLM_TEST_ENDPOINT");
    }
    if (png_path == nullptr || slurp(png_path).empty()) {
        return cannot_run("no page raster at $GRPC_VLM_TEST_PNG");
    }
    const std::string png = slurp(png_path);
    if (const char* wanted = env_or_null("GRPC_VLM_TEST_REQUIRE_DEVICE")) {
        const std::string device = endpoint_device(endpoint);
        require(device.starts_with(wanted),
                "the model runs on " + device + ", not on the required " + wanted);
    }

    vlm::Config config;
    config.endpoint = endpoint;
    config.vlm_timeout_seconds = 600;
    vlm::VlmConvertServiceImpl service(config);
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    require(server != nullptr, "server started");
    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                       grpc::InsecureChannelCredentials());
    auto stub = vlmv1::VlmConvertService::NewStub(channel);

    grpc::ClientContext context;
    auto stream = stub->ConvertPages(&context);
    vlmv1::ConvertPagesRequest request;
    // Default preset is Granite-Docling (DocTags); GRPC_VLM_TEST_PRESET_RAW
    // overrides the model name on the wire for other endpoints.
    const char* preset_raw = env_or_null("GRPC_VLM_TEST_PRESET_RAW");
    if (preset_raw != nullptr) {
        request.mutable_options()->set_preset_raw(preset_raw);
    }
    stream->Write(request);
    request.Clear();
    request.mutable_page_image()->set_page_no(1);
    request.mutable_page_image()->set_png(png);
    request.mutable_page_image()->set_width(1024);
    request.mutable_page_image()->set_height(1024);
    stream->Write(request);
    stream->WritesDone();

    bool got_started = false, got_document = false, got_complete = false;
    vlmv1::ConvertComplete complete;
    vlmv1::ConvertPagesResponse event;
    while (stream->Read(&event)) {
        if (event.has_page_started()) {
            got_started = true;
        } else if (event.has_page_document()) {
            got_document = true;
            const auto& doc = event.page_document().document();
            require(doc.texts_size() + doc.pictures_size() + doc.tables_size() > 0,
                    "real page maps to at least one item");
            require(doc.pages().count(1) == 1, "fragment carries the page entry");
        } else if (event.has_page_raw()) {
            require(false, "real endpoint page failed: " + event.page_raw().error());
        } else if (event.has_complete()) {
            got_complete = true;
            complete = event.complete();
        }
    }
    require(stream->Finish().ok(), "golden stream completes OK");
    require(got_started && got_document && got_complete,
            "started → document → complete event order");
    require(complete.pages_ok() == 1 && complete.pages_failed() == 0,
            "trailer counts the one page");
    server->Shutdown();
    std::println("vlm-golden-test passed");
    return 0;
} catch (const std::exception& error) {
    std::println(stderr, "{}", error.what());
    return 1;
}
