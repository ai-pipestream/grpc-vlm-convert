#pragma once

#include <string>
#include <vector>

#include "ai/pipestream/vlm/v1/vlm_convert.pb.h"

namespace vlm {

// One model preset: the name the endpoint knows the model by, the default
// prompt, the default response format, and the generation caps. Prompts,
// formats, stop strings, and token budgets follow Docling's VlmPipeline
// presets (docling/datamodel/stage_model_specs.py); the model itself is
// never loaded here — the name is just forwarded on the wire.
struct PresetSpec {
    ai::pipestream::vlm::v1::VlmPreset preset;
    const char* name;  // reported to clients and matched in GRPC_VLM_PRESETS
    const char* model;  // forwarded as "model" to the endpoint
    const char* prompt;  // default when ConvertOptions.prompt is empty
    ai::pipestream::vlm::v1::ResponseFormat format;
    // OpenAI "stop" parameter; empty means the parameter is omitted.
    std::vector<const char*> stop;
    // OpenAI "max_tokens" parameter.
    int max_tokens;
};

// All built-in presets, in enum order (UNSPECIFIED and RAW excluded).
const std::vector<PresetSpec>& all_presets();

// Looks up a built-in preset; nullptr for UNSPECIFIED / RAW / unknown.
const PresetSpec* find_preset(ai::pipestream::vlm::v1::VlmPreset preset);

// Looks up a preset by its report name ("granite-docling", ...); nullptr
// when the name is outside the built-in vocabulary.
const PresetSpec* find_preset_by_name(const std::string& name);

// Resolves the effective model name, prompt, response format, stop strings,
// and max_tokens for one request. preset_raw wins when preset is
// VLM_PRESET_RAW; explicit request fields (prompt, response_format)
// override preset defaults. Returns false when no model name can be
// resolved.
bool resolve_request(const ai::pipestream::vlm::v1::ConvertOptions& options,
                     std::string* model, std::string* prompt,
                     ai::pipestream::vlm::v1::ResponseFormat* format,
                     std::vector<std::string>* stop, int* max_tokens);

}  // namespace vlm
