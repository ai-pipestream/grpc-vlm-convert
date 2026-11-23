// Nemotron Parse 3.5 double and triple pass through one page: which calls
// are made with which settings (docling's _predict_nemotron and
// _nemotron_inputs), what a failed stage leaves, and the service path end
// to end against a fake endpoint that answers each stage by its request.

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "fixture.h"
#include "mapping/nemotron_multipass.h"
#include "service/nemotron_passes.h"
#include "service/vlm_convert_service.h"

namespace vlmv1 = ai::pipestream::vlm::v1;

namespace {

const std::string kPageTable = R"(\begin{tabular}{cc}a & b \\ c & d \\\end{tabular})";
const std::string kCropTable = R"(\begin{tabular}{cc}a & b \\ c & D \\\end{tabular})";
const std::string kPageText =
    "<x_0.1><y_0.1>Title<x_0.9><y_0.15><class_Title>\n"
    "<x_0.1><y_0.2>" + kPageTable + "<x_0.6><y_0.5><class_Table>";
// The crop's own view: the table fills most of it.
const std::string kCropText =
    "<x_0.05><y_0.05>" + kCropTable + "<x_0.95><y_0.95><class_Table>";
const std::string kMergedText =
    "<x_0.10000000><y_0.10000000>Title<x_0.90000000><y_0.15000000><class_Title>\n"
    "<x_0.10000000><y_0.20000000>" + kCropTable + "<x_0.60000000><y_0.50000000><class_Table>";

// The size a PNG's IHDR declares.
std::pair<uint32_t, uint32_t> png_size(std::string_view png) {
    auto be32 = [&](size_t at) {
        uint32_t value = 0;
        for (size_t i = 0; i < 4; i++) {
            value = (value << 8) | static_cast<unsigned char>(png[at + i]);
        }
        return value;
    };
    return png.size() >= 24 ? std::pair{be32(16), be32(20)} : std::pair{0u, 0u};
}

bool is_crop(const vlm::VlmCall& call) { return call.sampling.temperature == 0.1; }
bool is_detection(const vlm::VlmCall& call) {
    return call.prompt == vlm::mapping::kNemotronDetectPrompt;
}

vlm::VlmResult answer(std::string text, std::string finish = "stop") {
    vlm::VlmResult result;
    result.ok = true;
    result.text = std::move(text);
    result.finish_reason = std::move(finish);
    result.has_usage = true;
    result.prompt_tokens = 100;
    result.completion_tokens = 10;
    result.has_logprobs = true;
    result.mean_logprob = -0.1;
    result.scored_tokens = 10;
    return result;
}

vlm::VlmResult failure(std::string error) {
    vlm::VlmResult result;
    result.error = std::move(error);
    return result;
}

vlm::VlmResult cut_short() {
    vlm::VlmResult result;
    result.cancelled = true;
    return result;
}

// A CallBatch that answers each stage from the fields below and records
// every batch it was handed.
struct FakeStages {
    vlm::VlmResult page = answer(kPageText);
    vlm::VlmResult detection = answer("<x_0.1><y_0.2><x_0.6><y_0.5><class_Table>");
    vlm::VlmResult crop = answer(kCropText);
    std::vector<std::vector<vlm::VlmCall>> batches;
    std::vector<std::pair<uint32_t, uint32_t>> crop_sizes;

    vlm::CallBatch runner() {
        return [this](const std::vector<vlm::VlmCall>& calls) {
            batches.push_back(calls);
            std::vector<vlm::VlmResult> results;
            for (const vlm::VlmCall& call : calls) {
                if (is_crop(call)) {
                    crop_sizes.push_back(png_size(call.png));
                    results.push_back(crop);
                } else {
                    results.push_back(is_detection(call) ? detection : page);
                }
            }
            return results;
        };
    }
};

vlm::VlmCall page_call(const std::string& png) {
    return {.endpoint = "http://vlm:8080", .model = "nemotron", .prompt = "page prompt",
            .stop = {"</s>"}, .max_tokens = 4096, .top_logprobs = 0, .logprobs = true,
            .png = png};
}

void verify_double_pass_settings_and_merge() {
    const std::string png = gray_png(1000, 800);
    FakeStages fake;
    const vlm::NemotronPassResult out =
        vlm::run_nemotron_passes(page_call(png), {.triple = false}, fake.runner());
    require(fake.batches.size() == 2, "double: a page call, then one crop batch");
    const vlm::VlmCall& page = fake.batches[0][0];
    require(page.prompt == "page prompt" && page.max_tokens == 8744 &&
                page.sampling.temperature == 0.2 && page.sampling.repetition_penalty == 1.05 &&
                page.sampling.top_k == 50 && page.sampling.top_p == 1.0 &&
                page.sampling.skip_special_tokens == false && !page.logprobs &&
                page.stop == std::vector<std::string>{"</s>"},
            "double: the page call carries docling's fullpage settings");
    const vlm::VlmCall& crop = fake.batches[1][0];
    require(crop.prompt == "page prompt" && crop.max_tokens == 8744 &&
                crop.sampling.repetition_penalty == 1.0,
            "double: the crop call carries docling's crop settings");
    // The table at (0.1, 0.2)-(0.6, 0.5) plus NVIDIA's margins.
    require(fake.crop_sizes == std::vector<std::pair<uint32_t, uint32_t>>{{564, 284}},
            "double: the crop is the region and its margin");
    require(out.result.text == kMergedText, "double: the crop's table replaces the page's");
    require(out.shortfall.empty(), "double: nothing fell short");
    require(out.result.prompt_tokens == 200 && out.result.completion_tokens == 20,
            "double: usage sums both calls");
    require(!out.result.has_logprobs && out.result.scored_tokens == 0,
            "double: the merged text carries no logprobs");
}

void verify_triple_pass_crops_the_detected_regions() {
    const std::string png = gray_png(1000, 800);
    FakeStages fake;
    // The detector puts the table a little lower than the page call did.
    fake.detection = answer("<x_0.1><y_0.25><x_0.6><y_0.5><class_Table>");
    const vlm::NemotronPassResult out = vlm::run_nemotron_passes(
        page_call(png), {.triple = true, .union_regions = false}, fake.runner());
    require(fake.batches.size() == 3, "triple: page, detection, one crop batch");
    const vlm::VlmCall& detect = fake.batches[1][0];
    require(is_detection(detect) && detect.max_tokens == 4096 &&
                detect.sampling.temperature == 0.2,
            "triple: the detection call carries the no-text prompt and settings");
    require(fake.crop_sizes == std::vector<std::pair<uint32_t, uint32_t>>{{564, 240}},
            "triple: the crop follows the detector's region");
    require(out.result.prompt_tokens == 300, "triple: usage sums all three calls");
}

void verify_failed_stages_keep_the_page() {
    const std::string png = gray_png(1000, 800);
    {
        FakeStages fake;
        fake.page = answer(kPageText, "length");
        const auto out = vlm::run_nemotron_passes(page_call(png), {}, fake.runner());
        require(fake.batches.size() == 1 && out.result.text == kPageText &&
                    out.result.finish_reason == "length",
                "a page cut at max_tokens is not cropped");
        // The crop passes were asked for and did not run: that is a
        // shortfall like a failed detector, never a silent single pass.
        require(out.shortfall.contains("page call") && out.shortfall.contains("length") &&
                    out.shortfall.contains("no region was cropped"),
                "a page pass cut at max_tokens says the crop passes were skipped: " +
                    out.shortfall);
    }
    {
        FakeStages fake;
        fake.page = answer("  \n");
        const auto out =
            vlm::run_nemotron_passes(page_call(png), {.triple = true}, fake.runner());
        require(fake.batches.size() == 1 && out.shortfall.contains("no text") &&
                    out.shortfall.contains("no region was cropped"),
                "a blank page pass says the later passes were skipped: " + out.shortfall);
    }
    {
        FakeStages fake;
        fake.page = failure("HTTP 503");
        const auto out = vlm::run_nemotron_passes(page_call(png), {}, fake.runner());
        require(!out.result.ok && out.shortfall.empty(),
                "a failed page call is a failed page, with nothing to fall short of");
    }
    {
        FakeStages fake;
        fake.detection = failure("HTTP 503");
        const auto out =
            vlm::run_nemotron_passes(page_call(png), {.triple = true}, fake.runner());
        require(fake.batches.size() == 2 && out.result.text == kPageText,
                "triple: a failed detector crops nothing");
        require(out.shortfall.contains("detection call failed (HTTP 503)"),
                "triple: the failed detector is reported: " + out.shortfall);
    }
    {
        FakeStages fake;
        fake.crop = failure("HTTP 500");
        const auto out = vlm::run_nemotron_passes(page_call(png), {}, fake.runner());
        require(out.result.text == kPageText && out.result.ok, "a failed crop keeps the page");
        require(out.shortfall == "1 of 1 region crops failed; those regions keep the page pass text",
                "a failed crop is reported: " + out.shortfall);
    }
    {
        FakeStages fake;
        fake.crop = cut_short();
        const auto out = vlm::run_nemotron_passes(page_call(png), {}, fake.runner());
        require(out.result.cancelled, "a cancelled crop cancels the page");
    }
    {
        FakeStages fake;
        const std::string not_png = make_png("no pixels");
        const auto out = vlm::run_nemotron_passes(page_call(not_png), {}, fake.runner());
        require(fake.batches.size() == 1 && out.result.text == kPageText &&
                    out.shortfall.contains("could not be decoded"),
                "an undecodable raster keeps the page and says so");
    }
}

void verify_crops_go_in_batches() {
    const std::string png = gray_png(1000, 800);
    FakeStages fake;
    fake.page = answer(
        "<x_0.1><y_0.1>" + kPageTable + "<x_0.3><y_0.2><class_Table>\n"
        "<x_0.1><y_0.4>" + kPageTable + "<x_0.3><y_0.5><class_Table>\n"
        "<x_0.1><y_0.7>" + kPageTable + "<x_0.3><y_0.8><class_Table>");
    vlm::run_nemotron_passes(page_call(png), {.crop_batch_size = 2}, fake.runner());
    require(fake.batches.size() == 3 && fake.batches[1].size() == 2 && fake.batches[2].size() == 1,
            "three crops in batches of two");
}

void verify_options() {
    std::string error;
    vlmv1::ConvertOptions options;
    const auto v35 = vlmv1::RESPONSE_FORMAT_NEMOTRON_PARSE_V3_5;
    require(!vlm::nemotron_passes_for(options, v35, &error) && error.empty(),
            "unset is the single pass");
    options.mutable_nemotron_parse()->set_passes(vlmv1::NEMOTRON_PARSE_PASSES_TRIPLE);
    options.mutable_nemotron_parse()->set_triple_regions(
        vlmv1::NEMOTRON_PARSE_TRIPLE_REGIONS_DETECTOR);
    auto passes = vlm::nemotron_passes_for(options, v35, &error);
    require(passes && passes->triple && !passes->union_regions && passes->crop_batch_size == 8,
            "triple, detector regions, docling's batch size");
    options.mutable_nemotron_parse()->set_crop_batch_size(500);
    require(vlm::nemotron_passes_for(options, v35, &error)->crop_batch_size ==
                vlm::mapping::kMaxNemotronCrops,
            "a batch never exceeds the crop cap");

    vlm::nemotron_passes_for(options, vlmv1::RESPONSE_FORMAT_DOCTAGS, &error);
    require(error.contains("needs response format RESPONSE_FORMAT_NEMOTRON_PARSE_V3_5"),
            "multi-pass with another format is refused: " + error);
    error.clear();
    options.set_top_logprobs(3);
    vlm::nemotron_passes_for(options, v35, &error);
    require(error.contains("top_logprobs"), "multi-pass with alternates is refused: " + error);
    error.clear();
    options.set_top_logprobs(0);
    options.mutable_nemotron_parse()->set_passes(static_cast<vlmv1::NemotronParsePasses>(9));
    vlm::nemotron_passes_for(options, v35, &error);
    require(error.contains("unknown nemotron_parse.passes value: 9"), "unknown passes: " + error);
}

// An endpoint that answers the page and crop stages by request.
struct StageEndpoint {
    httplib::Server server;
    std::thread thread;
    int port = 0;
    std::string page_text = kPageText;
    std::mutex mutex;
    std::vector<nlohmann::json> bodies;
    std::atomic<int> open{0};
    std::atomic<int> most_open{0};

    void start() {
        server.Post("/v1/chat/completions", [this](const httplib::Request& request,
                                                   httplib::Response& response) {
            const int now = ++open;
            for (int seen = most_open.load();
                 now > seen && !most_open.compare_exchange_weak(seen, now);) {
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const nlohmann::json body = nlohmann::json::parse(request.body);
            {
                std::lock_guard<std::mutex> lock(mutex);
                bodies.push_back(body);
            }
            const bool crop = body.value("temperature", 0.0) == 0.1;
            const nlohmann::json reply = {
                {"choices",
                 {{{"message", {{"content", crop ? kCropText : page_text}}},
                   {"finish_reason", "stop"}}}}};
            response.set_content(reply.dump(), "application/json");
            open--;
        });
        port = server.bind_to_any_port("127.0.0.1");
        thread = std::thread([this] { server.listen_after_bind(); });
    }

    ~StageEndpoint() {
        server.stop();
        if (thread.joinable()) {
            thread.join();
        }
    }
};

vlmv1::ConvertOptions nemotron_options(vlmv1::NemotronParsePasses passes) {
    vlmv1::ConvertOptions options;
    options.set_preset(vlmv1::VLM_PRESET_RAW);
    options.set_preset_raw("nemotron-parse");
    options.set_response_format(vlmv1::RESPONSE_FORMAT_NEMOTRON_PARSE_V3_5);
    options.mutable_nemotron_parse()->set_passes(passes);
    return options;
}

vlmv1::PageImage gray_page(uint32_t width, uint32_t height) {
    vlmv1::PageImage image;
    image.set_page_no(1);
    image.set_width(width);
    image.set_height(height);
    image.set_png(gray_png(1000, 800));
    return image;
}

// One page through the service; returns the page's final event.
vlmv1::ConvertPagesResponse convert_one(vlm::VlmConvertServiceImpl& service,
                                        const vlmv1::ConvertOptions& options,
                                        const vlmv1::PageImage& image) {
    std::vector<vlmv1::ConvertPagesRequest> requests(2);
    *requests[0].mutable_options() = options;
    *requests[1].mutable_page_image() = image;
    size_t next = 0;
    vlmv1::ConvertPagesResponse last;
    const grpc::Status status = service.ConvertPagesCore(
        [&](vlmv1::ConvertPagesRequest* request) {
            if (next == requests.size()) {
                return false;
            }
            *request = requests[next++];
            return true;
        },
        [&](const vlmv1::ConvertPagesResponse& event) {
            if (event.has_page_document() || event.has_page_raw()) {
                last = event;
            }
            return true;
        },
        nullptr);
    require(status.ok(), "nemotron stream: " + status.error_message());
    return last;
}

// One double pass page through the service; returns its PageDocument.
vlmv1::PageDocument convert_double_pass(vlm::VlmConvertServiceImpl& service) {
    const vlmv1::ConvertPagesResponse event = convert_one(
        service, nemotron_options(vlmv1::NEMOTRON_PARSE_PASSES_DOUBLE), gray_page(1000, 800));
    require(event.has_page_document(), "double pass: the page maps");
    return event.page_document();
}

void verify_service_runs_the_passes() {
    StageEndpoint endpoint;
    endpoint.start();
    vlm::Config config;
    config.endpoint = "http://127.0.0.1:" + std::to_string(endpoint.port);
    vlm::VlmConvertServiceImpl service(config);

    const vlmv1::PageDocument page = convert_double_pass(service);
    require(page.document().tables_size() == 1, "double pass: one table");
    require(page.document().tables(0).DebugString().contains("\"D\""),
            "double pass: the table holds the crop's cell text");
    require(page.warnings_size() == 0, "double pass: nothing fell short");
    std::lock_guard<std::mutex> lock(endpoint.mutex);
    require(endpoint.bodies.size() == 2, "double pass: page call and crop call on the wire");
    const nlohmann::json& page_body = endpoint.bodies[0];
    require(page_body["temperature"] == 0.2 && page_body["repetition_penalty"] == 1.05 &&
                page_body["top_k"] == 50 && page_body["top_p"] == 1.0 &&
                page_body["skip_special_tokens"] == false && page_body["max_tokens"] == 8744 &&
                !page_body.contains("logprobs"),
            "double pass: the page call's sampling reaches the wire: " + page_body.dump());
    require(endpoint.bodies[1]["temperature"] == 0.1, "double pass: then the crop call");
}

// A page's crops run beside each other only on free in-flight slots.
void verify_crops_stay_within_the_inflight_cap() {
    StageEndpoint endpoint;
    endpoint.page_text =
        "<x_0.1><y_0.1>" + kPageTable + "<x_0.3><y_0.2><class_Table>\n"
        "<x_0.1><y_0.4>" + kPageTable + "<x_0.3><y_0.5><class_Table>\n"
        "<x_0.1><y_0.7>" + kPageTable + "<x_0.3><y_0.8><class_Table>\n"
        "<x_0.6><y_0.7>" + kPageTable + "<x_0.8><y_0.8><class_Table>";
    endpoint.start();
    vlm::Config config;
    config.endpoint = "http://127.0.0.1:" + std::to_string(endpoint.port);
    config.max_inflight = 2;
    vlm::VlmConvertServiceImpl service(config);

    convert_double_pass(service);
    require(endpoint.bodies.size() == 5, "four crops after the page call");
    require(endpoint.most_open.load() <= 2,
            "crops never pass the in-flight cap: " + std::to_string(endpoint.most_open.load()));
    require(service.vlm_slots().in_use() == 0, "every slot is given back");
}


// The single pass sends what docling's API engine sends for the
// nemotron_parse_v3_5 preset, and its page prompt when the request has
// none.
void verify_single_pass_matches_docling() {
    StageEndpoint endpoint;
    endpoint.start();
    vlm::Config config;
    config.endpoint = "http://127.0.0.1:" + std::to_string(endpoint.port);
    vlm::VlmConvertServiceImpl service(config);

    vlmv1::ConvertOptions options = nemotron_options(vlmv1::NEMOTRON_PARSE_PASSES_UNSPECIFIED);
    require(convert_one(service, options, gray_page(1000, 800)).has_page_document(),
            "single pass: the page maps");
    options.set_prompt("my prompt");
    convert_one(service, options, gray_page(1000, 800));

    std::lock_guard<std::mutex> lock(endpoint.mutex);
    require(endpoint.bodies.size() == 2, "single pass: one call per page");
    nlohmann::json settings = endpoint.bodies[0];
    settings.erase("messages");
    const nlohmann::json docling = {
        {"model", "nemotron-parse"}, {"max_tokens", 8744}, {"logprobs", true},
        {"temperature", 0.0},        {"top_k", 1},         {"repetition_penalty", 1.1},
        {"skip_special_tokens", false}};
    require(settings == docling,
            "single pass: docling's API settings reach the wire: " + settings.dump());
    auto prompt = [&](size_t call) {
        return endpoint.bodies[call]["messages"][0]["content"][0]["text"].get<std::string>();
    };
    require(prompt(0) == vlm::mapping::kNemotronPagePrompt,
            "single pass: docling's page prompt by default: " + prompt(0));
    require(prompt(1) == "my prompt", "single pass: a request's own prompt is kept");
}

// A page sent without its size takes it from the PNG header, so its boxes
// match the page sent with one; without a readable header it fails before
// the model is called.
void verify_page_size_from_png_header() {
    StageEndpoint endpoint;
    endpoint.start();
    vlm::Config config;
    config.endpoint = "http://127.0.0.1:" + std::to_string(endpoint.port);
    vlm::VlmConvertServiceImpl service(config);
    const vlmv1::ConvertOptions options =
        nemotron_options(vlmv1::NEMOTRON_PARSE_PASSES_UNSPECIFIED);

    auto table_box = [&](const vlmv1::PageImage& image) {
        const vlmv1::ConvertPagesResponse event = convert_one(service, options, image);
        require(event.has_page_document() && event.page_document().document().tables_size() == 1,
                "sized page: one table");
        return event.page_document().document().tables(0).prov(0).bbox();
    };
    const auto declared = table_box(gray_page(1000, 800));
    const auto from_header = table_box(gray_page(0, 0));
    require(declared.r() > declared.l() && declared.b() > declared.t(),
            "declared size: the box has area");
    require(from_header.DebugString() == declared.DebugString(),
            "no size: the box matches the declared size's: " + from_header.DebugString());

    vlmv1::PageImage unreadable = gray_page(0, 0);
    unreadable.set_png(make_png("NOT-A-HEADER"));
    const size_t calls = endpoint.bodies.size();
    const vlmv1::ConvertPagesResponse event = convert_one(service, options, unreadable);
    require(event.has_page_raw() && event.page_raw().error().contains("no width or height"),
            "no size and no header: the page fails: " + event.DebugString());
    require(endpoint.bodies.size() == calls, "no size and no header: the model is not called");
}

}  // namespace

int main() {
    try {
        verify_double_pass_settings_and_merge();
        verify_triple_pass_crops_the_detected_regions();
        verify_failed_stages_keep_the_page();
        verify_crops_go_in_batches();
        verify_options();
        verify_service_runs_the_passes();
        verify_crops_stay_within_the_inflight_cap();
        verify_single_pass_matches_docling();
        verify_page_size_from_png_header();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("nemotron-passes-test passed");
    return 0;
}
