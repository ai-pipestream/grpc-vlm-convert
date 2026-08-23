// Golden test against a real VLM endpoint: one real page raster through
// the configured preset, asserting labels + ordering + counts rather than
// full message equality (model output drifts between versions).
//
// Needs GRPC_VLM_TEST_ENDPOINT and GRPC_VLM_TEST_PNG (a real page raster
// file); skips with 77 otherwise, so CI without a VLM server stays green.

#include <grpcpp/grpcpp.h>

#include <fstream>
#include <string>

#include "config.h"
#include "fixture.h"
#include "service/vlm_convert_service.h"

namespace vlmv1 = ai::pipestream::vlm::v1;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
    const char* endpoint = env_or_null("GRPC_VLM_TEST_ENDPOINT");
    const char* png_path = env_or_null("GRPC_VLM_TEST_PNG");
    if (endpoint == nullptr) {
        return skip("no VLM endpoint at $GRPC_VLM_TEST_ENDPOINT");
    }
    if (png_path == nullptr || slurp(png_path).empty()) {
        return skip("no page raster at $GRPC_VLM_TEST_PNG");
    }
    const std::string png = slurp(png_path);

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
}
