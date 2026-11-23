#pragma once

// Nemotron Parse 3.5 double and triple pass for one page, as docling runs
// them (VlmConvertModel._predict_nemotron): a page call, a detection call
// for triple, then one call per Table or Chart crop, merged back into the
// page text by mapping/nemotron_multipass.

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "ai/pipestream/vlm/v1/vlm_convert.pb.h"
#include "budget.h"
#include "vlm_client.h"

namespace vlm {

struct NemotronPasses {
    // False is the double pass: crops come from the page call's regions.
    bool triple = false;
    // Triple pass: also crop page call regions the detector missed.
    bool union_regions = true;
    // Crop calls sent together.
    size_t crop_batch_size = 8;
};

// The passes a request asks for, nullopt for the single pass. When the
// options cannot be honored it returns nullopt and sets `error`: an enum
// value outside the proto, a double or triple pass for a format other than
// Nemotron Parse 3.5, or one combined with top_logprobs.
std::optional<NemotronPasses> nemotron_passes_for(
    const ai::pipestream::vlm::v1::ConvertOptions& options,
    ai::pipestream::vlm::v1::ResponseFormat format, std::string* error);

// docling's single page call for Nemotron Parse 3.5 through an API
// engine (the nemotron_parse_v3_5 preset's api_overrides): greedy decoding
// with top-k 1, repetition penalty 1.1, special tokens kept and 8744
// tokens. Without a prompt in the request the page gets docling's page
// prompt. Every Nemotron Parse 3.5 page starts from this call; a double or
// triple pass then sets each stage's own sampling.
void apply_nemotron_page_settings(VlmCall* call, bool prompt_given);

// Runs every call and returns their results in the same order.
using CallBatch = std::function<std::vector<VlmResult>(const std::vector<VlmCall>&)>;

struct NemotronPassResult {
    // The page call's result with the crops merged into its text and the
    // token usage of every call summed. It carries no logprobs or
    // alternates: the merged text is not one the model generated.
    VlmResult result;
    // Why some regions kept their page pass text although a pass was
    // asked for (a detection or crop call that failed, or a raster that
    // could not be cropped); empty when every call came back complete. A
    // crop the structure check refused is not a shortfall: keeping the
    // page block is the check doing its job.
    std::string shortfall;
};

// `page` is the call a single pass would make; the stages reuse its
// endpoint, model, prompt, image and stop strings with docling's sampling
// settings.
NemotronPassResult run_nemotron_passes(const VlmCall& page, const NemotronPasses& passes,
                                       const CallBatch& run);

// A CallBatch for the service: the first call runs on the slot the page
// already holds, and up to one more call per in-flight slot that is free
// right now runs beside it. It never waits for a slot, so a busy server
// runs a page's crops one at a time instead of stalling.
std::vector<VlmResult> run_within_slots(const std::vector<VlmCall>& calls, Budget& slots);

}  // namespace vlm
