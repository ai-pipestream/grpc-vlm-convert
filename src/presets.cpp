#include "presets.h"

namespace vlm {

namespace {

namespace vlmv1 = ai::pipestream::vlm::v1;

const std::vector<PresetSpec> kPresets = {
    {vlmv1::VLM_PRESET_SMOLDOCLING, "smoldocling", "HuggingFaceTB/SmolDocling-256M-preview",
     "Convert this page to docling.", vlmv1::RESPONSE_FORMAT_DOCTAGS},
    {vlmv1::VLM_PRESET_GRANITE_DOCLING, "granite-docling", "ibm-granite/granite-docling-258M",
     "Convert this page to DocTags.", vlmv1::RESPONSE_FORMAT_DOCTAGS},
    {vlmv1::VLM_PRESET_GOT_OCR_2, "got-ocr-2", "stepfun-ai/GOT-OCR-2.0-hf",
     "Convert this page to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN},
    {vlmv1::VLM_PRESET_GRANITE_VISION, "granite-vision", "ibm-granite/granite-vision-3.2-2b",
     "Convert this page to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN},
    {vlmv1::VLM_PRESET_DEEPSEEK_OCR, "deepseek-ocr", "deepseek-ai/DeepSeek-OCR",
     "Convert this document to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN},
    {vlmv1::VLM_PRESET_NANONETS_OCR2, "nanonets-ocr2", "nanonets/Nanonets-OCR2-3B",
     "Extract the text from the above document as if you were reading it naturally.",
     vlmv1::RESPONSE_FORMAT_MARKDOWN},
    {vlmv1::VLM_PRESET_GLM_OCR, "glm-ocr", "zai-org/GLM-OCR",
     "Convert this page to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN},
    {vlmv1::VLM_PRESET_LIGHTON_OCR, "lighton-ocr", "lightonai/LightOnOCR-1B",
     "Convert this page to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN},
};

}  // namespace

const std::vector<PresetSpec>& all_presets() {
    return kPresets;
}

const PresetSpec* find_preset(vlmv1::VlmPreset preset) {
    for (const PresetSpec& spec : kPresets) {
        if (spec.preset == preset) {
            return &spec;
        }
    }
    return nullptr;
}

const PresetSpec* find_preset_by_name(const std::string& name) {
    for (const PresetSpec& spec : kPresets) {
        if (name == spec.name) {
            return &spec;
        }
    }
    return nullptr;
}

bool resolve_request(const vlmv1::ConvertOptions& options, std::string* model,
                     std::string* prompt, vlmv1::ResponseFormat* format) {
    const PresetSpec* spec = nullptr;
    if (options.preset() == vlmv1::VLM_PRESET_RAW) {
        *model = options.preset_raw();
        if (model->empty()) {
            return false;
        }
        *prompt = "Convert this page to DocTags.";
        *format = vlmv1::RESPONSE_FORMAT_DOCTAGS;
    } else {
        vlmv1::VlmPreset preset = options.preset() == vlmv1::VLM_PRESET_UNSPECIFIED
                                      ? vlmv1::VLM_PRESET_GRANITE_DOCLING
                                      : options.preset();
        spec = find_preset(preset);
        if (spec == nullptr) {
            return false;
        }
        *model = spec->model;
        *prompt = spec->prompt;
        *format = spec->format;
    }
    // preset_raw on a non-RAW preset overrides the model name on the wire.
    if (!options.preset_raw().empty() && options.preset() != vlmv1::VLM_PRESET_RAW) {
        *model = options.preset_raw();
    }
    if (!options.prompt().empty()) {
        *prompt = options.prompt();
    }
    if (options.response_format() != vlmv1::RESPONSE_FORMAT_UNSPECIFIED) {
        *format = options.response_format();
    }
    return true;
}

}  // namespace vlm
