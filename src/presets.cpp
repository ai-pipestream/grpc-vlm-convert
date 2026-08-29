#include "presets.h"

#include <algorithm>

namespace vlm {

namespace {

namespace vlmv1 = ai::pipestream::vlm::v1;

const std::vector<PresetSpec> kPresets = {
    {vlmv1::VLM_PRESET_SMOLDOCLING, "smoldocling", "HuggingFaceTB/SmolDocling-256M-preview",
     "Convert this page to docling.", vlmv1::RESPONSE_FORMAT_DOCTAGS,
     {"</doctag>", "<end_of_utterance>"}, 4096},
    {vlmv1::VLM_PRESET_GRANITE_DOCLING, "granite-docling", "ibm-granite/granite-docling-258M",
     "Convert this page to DocTags.", vlmv1::RESPONSE_FORMAT_DOCTAGS,
     {"</doctag>", "<|end_of_text|>"}, 8192},
    {vlmv1::VLM_PRESET_GOT_OCR_2, "got-ocr-2", "stepfun-ai/GOT-OCR-2.0-hf",
     "Convert this page to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN, {}, 4096},
    {vlmv1::VLM_PRESET_GRANITE_VISION, "granite-vision", "ibm-granite/granite-vision-3.2-2b",
     "Convert this page to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN, {}, 4096},
    {vlmv1::VLM_PRESET_DEEPSEEK_OCR, "deepseek-ocr", "deepseek-ai/DeepSeek-OCR",
     "Convert this document to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN, {}, 4096},
    {vlmv1::VLM_PRESET_NANONETS_OCR2, "nanonets-ocr2", "nanonets/Nanonets-OCR2-3B",
     "Extract the text from the above document as if you were reading it naturally.",
     vlmv1::RESPONSE_FORMAT_MARKDOWN, {}, 4096},
    {vlmv1::VLM_PRESET_GLM_OCR, "glm-ocr", "zai-org/GLM-OCR",
     "Convert this page to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN, {}, 4096},
    {vlmv1::VLM_PRESET_LIGHTON_OCR, "lighton-ocr", "lightonai/LightOnOCR-1B",
     "Convert this page to markdown.", vlmv1::RESPONSE_FORMAT_MARKDOWN, {}, 4096},
    // A general VLM rather than an OCR fine-tune, so the prompt says what
    // a page conversion must keep; its multimodal context is 8k tokens.
    {vlmv1::VLM_PRESET_NORTH_MICRO_VISION, "north-micro-vision",
     "CohereLabs/North-Micro-Vision-Instruct",
     "Convert this page to markdown. Keep the reading order, every heading, "
     "paragraph, list and table (tables as markdown tables), and transcribe "
     "the text exactly; do not summarize or describe.",
     vlmv1::RESPONSE_FORMAT_MARKDOWN, {}, 8192},
};

}  // namespace

const std::vector<PresetSpec>& all_presets() {
    return kPresets;
}

const PresetSpec* find_preset(vlmv1::VlmPreset preset) {
    const auto spec = std::ranges::find(kPresets, preset, &PresetSpec::preset);
    return spec != kPresets.end() ? &*spec : nullptr;
}

const PresetSpec* find_preset_by_name(const std::string& name) {
    const auto spec = std::ranges::find(kPresets, name, &PresetSpec::name);
    return spec != kPresets.end() ? &*spec : nullptr;
}

bool resolve_request(const vlmv1::ConvertOptions& options, std::string* model,
                     std::string* prompt, vlmv1::ResponseFormat* format,
                     std::vector<std::string>* stop, int* max_tokens) {
    const PresetSpec* spec = nullptr;
    if (options.preset() == vlmv1::VLM_PRESET_RAW) {
        *model = options.preset_raw();
        if (model->empty()) {
            return false;
        }
        *prompt = "Convert this page to DocTags.";
        *format = vlmv1::RESPONSE_FORMAT_DOCTAGS;
        stop->clear();
        *max_tokens = 4096;
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
        stop->assign(spec->stop.begin(), spec->stop.end());
        *max_tokens = spec->max_tokens;
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
