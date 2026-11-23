#include "nemotron_passes.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <optional>
#include <string_view>
#include <thread>

#include "mapping/builder.h"
#include "mapping/nemotron_multipass.h"
#include "mapping/region_image.h"

namespace vlm {
namespace {

namespace vlmv1 = ai::pipestream::vlm::v1;

enum class Stage { kPage, kDetection, kCrop };

// docling's max_new_tokens for Nemotron Parse 3.5.
constexpr int kNemotronMaxTokens = 8744;

// docling's per-stage generation settings (_nemotron_inputs).
VlmCall stage_call(const VlmCall& page, Stage stage) {
    VlmCall call = page;
    call.max_tokens = stage == Stage::kDetection ? 4096 : kNemotronMaxTokens;
    call.sampling = {
        .temperature = stage == Stage::kCrop ? 0.1 : 0.2,
        .top_p = 1.0,
        .top_k = 50,
        .repetition_penalty = stage == Stage::kCrop ? 1.0 : 1.05,
        .skip_special_tokens = false,
    };
    // The merged page drops them, so no stage asks for them.
    call.logprobs = false;
    call.top_logprobs = 0;
    if (stage == Stage::kDetection) {
        call.prompt = std::string(mapping::kNemotronDetectPrompt);
    }
    return call;
}

// docling's _nemotron_complete: an answer that ended on its own and says
// something.
bool complete(const VlmResult& result) {
    constexpr std::string_view kEnded[] = {"", "stop", "eos", "end_of_sequence", "unspecified"};
    return result.ok && std::ranges::find(kEnded, result.finish_reason) != std::end(kEnded) &&
           !mapping::trim(result.text).empty();
}

void add_usage(VlmResult* total, const VlmResult& part) {
    if (part.has_usage) {
        total->has_usage = true;
        total->prompt_tokens += part.prompt_tokens;
        total->completion_tokens += part.completion_tokens;
    }
}

NemotronPassResult finished(VlmResult result, std::string shortfall = {}) {
    result.has_logprobs = false;
    result.mean_logprob = 0.0;
    result.scored_tokens = 0;
    result.alternatives.clear();
    return {std::move(result), std::move(shortfall)};
}

NemotronPassResult cancelled() {
    VlmResult result;
    result.cancelled = true;
    result.error = "VLM call cancelled";
    return {std::move(result), {}};
}

}  // namespace

void apply_nemotron_page_settings(VlmCall* call, bool prompt_given) {
    if (!prompt_given) {
        call->prompt = std::string(mapping::kNemotronPagePrompt);
    }
    call->max_tokens = kNemotronMaxTokens;
    call->sampling = {
        .temperature = 0.0,
        .top_p = std::nullopt,
        .top_k = 1,
        .repetition_penalty = 1.1,
        .skip_special_tokens = false,
    };
}

std::optional<NemotronPasses> nemotron_passes_for(const vlmv1::ConvertOptions& options,
                                                  vlmv1::ResponseFormat format,
                                                  std::string* error) {
    const vlmv1::NemotronParseOptions& asked = options.nemotron_parse();
    if (!vlmv1::NemotronParsePasses_IsValid(asked.passes())) {
        *error = "unknown nemotron_parse.passes value: " + std::to_string(asked.passes());
        return std::nullopt;
    }
    if (!vlmv1::NemotronParseTripleRegions_IsValid(asked.triple_regions())) {
        *error = "unknown nemotron_parse.triple_regions value: " +
                 std::to_string(asked.triple_regions());
        return std::nullopt;
    }
    const bool triple = asked.passes() == vlmv1::NEMOTRON_PARSE_PASSES_TRIPLE;
    if (!triple && asked.passes() != vlmv1::NEMOTRON_PARSE_PASSES_DOUBLE) {
        return std::nullopt;
    }
    const std::string name = vlmv1::NemotronParsePasses_Name(asked.passes());
    if (format != vlmv1::RESPONSE_FORMAT_NEMOTRON_PARSE_V3_5) {
        *error = "nemotron_parse.passes " + name + " needs response format " +
                 "RESPONSE_FORMAT_NEMOTRON_PARSE_V3_5, not " + vlmv1::ResponseFormat_Name(format);
        return std::nullopt;
    }
    if (options.top_logprobs() > 0) {
        *error = "nemotron_parse.passes " + name +
                 " cannot be combined with top_logprobs: the merged page text is not text the "
                 "model generated, so it has no token alternates";
        return std::nullopt;
    }
    return NemotronPasses{
        .triple = triple,
        .union_regions =
            asked.triple_regions() != vlmv1::NEMOTRON_PARSE_TRIPLE_REGIONS_DETECTOR,
        // More than one batch per crop would change nothing.
        .crop_batch_size = asked.crop_batch_size() == 0
                               ? 8
                               : std::min<size_t>(asked.crop_batch_size(),
                                                  mapping::kMaxNemotronCrops),
    };
}

NemotronPassResult run_nemotron_passes(const VlmCall& page, const NemotronPasses& passes,
                                       const CallBatch& run) {
    VlmResult result = run({stage_call(page, Stage::kPage)}).at(0);
    if (result.cancelled) {
        return cancelled();
    }
    if (!complete(result)) {
        return finished(std::move(result));  // nothing to crop from
    }

    std::optional<std::string> detection;
    if (passes.triple) {
        const VlmResult detected = run({stage_call(page, Stage::kDetection)}).at(0);
        if (detected.cancelled) {
            return cancelled();
        }
        add_usage(&result, detected);
        // A failed detector must not turn the triple pass into a double.
        if (!complete(detected)) {
            return finished(std::move(result), "the detection call failed (" +
                                                   (detected.ok ? "finish reason " +
                                                                      detected.finish_reason
                                                                : detected.error) +
                                                   "); no region was cropped");
        }
        detection = detected.text;
    }

    const mapping::RegionImages raster(page.png);
    if (!raster.decoded()) {
        return finished(std::move(result),
                        "the page raster could not be decoded; no region was cropped");
    }
    const std::vector<mapping::NemotronCrop> crops = mapping::plan_nemotron_crops(
        raster.width(), raster.height(), result.text, detection ? &*detection : nullptr,
        passes.union_regions);

    std::vector<mapping::NemotronBlock> candidates;
    size_t failed = 0;
    const size_t batch_size = std::max<size_t>(passes.crop_batch_size, 1);
    for (size_t start = 0; start < crops.size(); start += batch_size) {
        const size_t end = std::min(crops.size(), start + batch_size);
        // The calls view these images, so they live until the batch ends.
        std::vector<std::string> images;
        for (size_t i = start; i < end; i++) {
            const mapping::NemotronCrop& crop = crops[i];
            images.push_back(raster.crop(crop.left, crop.top, crop.right, crop.bottom));
        }
        std::vector<VlmCall> calls;
        std::vector<size_t> sent;
        for (size_t i = 0; i < images.size(); i++) {
            if (images[i].empty()) {
                failed++;
                continue;
            }
            VlmCall call = stage_call(page, Stage::kCrop);
            call.png = images[i];
            calls.push_back(std::move(call));
            sent.push_back(start + i);
        }
        if (calls.empty()) {
            continue;
        }
        const std::vector<VlmResult> answers = run(calls);
        for (size_t i = 0; i < answers.size(); i++) {
            const VlmResult& answer = answers[i];
            if (answer.cancelled) {
                return cancelled();
            }
            add_usage(&result, answer);
            if (!complete(answer)) {
                failed++;
                continue;
            }
            for (mapping::NemotronBlock& block :
                 mapping::remap_nemotron_crop(answer.text, crops[sent[i]])) {
                candidates.push_back(std::move(block));
            }
        }
    }
    result.text = mapping::merge_nemotron_crops(result.text, candidates);
    std::string shortfall;
    if (failed > 0) {
        shortfall = std::to_string(failed) + " of " + std::to_string(crops.size()) +
                    " region crops failed; those regions keep the page pass text";
    }
    return finished(std::move(result), std::move(shortfall));
}

std::vector<VlmResult> run_within_slots(const std::vector<VlmCall>& calls, Budget& slots) {
    std::vector<VlmResult> results(calls.size());
    std::vector<Lease> extra;
    while (extra.size() + 1 < calls.size()) {
        Lease slot = slots.try_acquire(1);
        if (!slot) {
            break;
        }
        extra.push_back(std::move(slot));
    }
    std::atomic<size_t> next{0};
    auto drain = [&] {
        for (size_t i = next++; i < calls.size(); i = next++) {
            try {
                results[i] = generate(calls[i]);
            } catch (const std::exception& error) {
                // A throw out of a helper thread would end the process.
                results[i].error = std::string("VLM call failed: ") + error.what();
            }
        }
    };
    std::vector<std::thread> helpers;
    helpers.reserve(extra.size());
    for (size_t i = 0; i < extra.size(); i++) {
        helpers.emplace_back(drain);
    }
    drain();
    for (std::thread& helper : helpers) {
        helper.join();
    }
    return results;
}

}  // namespace vlm
