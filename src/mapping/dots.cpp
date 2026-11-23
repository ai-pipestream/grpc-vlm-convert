// dots.ocr / dots.mocr layout JSON (docling dots_utils.py, v2.142.1): a
// JSON array of {"bbox": [x1, y1, x2, y2], "category": ..., "text": ...}
// elements in reading order. Boxes are pixels of the image the model saw
// after Qwen2-VL resizing, rescaled to the page. Text is Markdown with
// inline HTML; tables are HTML; a fenced block is code.
//
// Docling skips an element without a usable bbox and cuts an unfinished
// tail off silently. This mapper keeps such an element's text on the
// full-page box and reports every entry and byte it reads past as a typed
// PageWarning.
//
// Docling degrades malformed output to an empty page. This mapper does
// not: an answer that is prose, a refusal, JSON broken beyond repair, or a
// layout of elements it builds nothing from is a mapping failure, so the
// caller gets the model's text back as PageRaw instead of a blank page
// that reads as converted. Only "[]", the model saying the page is empty,
// and a blank answer (reported with a warning) are an empty page.

#include <charconv>
#include <cmath>
#include <optional>

#include <nlohmann/json.hpp>

#include "layout.h"

namespace vlm::mapping {

namespace {

using Ref = LayoutWriter::Ref;
using Box = std::optional<docv1::BoundingBox>;

// _clean_json: keep from the first '['; a truncated array is cut after
// its last complete object and closed. `dropped` is how many bytes of an
// unfinished element that cut away (a separator alone is not content).
std::string clean_json(const std::string& raw, size_t* dropped) {
    *dropped = 0;
    const size_t open = raw.find('[');
    if (open == std::string::npos) {
        return "[]";
    }
    std::string text = raw.substr(open);
    size_t end = text.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        end--;
    }
    if (end == 0 || text[end - 1] != ']') {
        const size_t brace = text.rfind('}', end == 0 ? 0 : end - 1);
        if (brace == std::string::npos) {
            return "[]";
        }
        for (size_t at = brace + 1; at < end; at++) {
            const auto c = static_cast<unsigned char>(text[at]);
            // A separator, the closing bracket and a code fence after it
            // are the array's own punctuation, not an element.
            if (std::isspace(c) == 0 && c != ',' && c != ']' && c != '`') {
                *dropped = end - at;
                break;
            }
        }
        return text.substr(0, brace + 1) + "]";
    }
    return text;
}

// True when the array the model wrote is itself empty and closed ("[]",
// fenced or not): the model declaring a page with no layout elements.
// clean_json also answers "[]" for prose and for an array cut off before
// its first complete object, which is why this reads the raw text.
bool declares_empty_layout(const std::string& raw) {
    size_t at = raw.find('[');
    if (at == std::string::npos) {
        return false;
    }
    at++;
    while (at < raw.size() && std::isspace(static_cast<unsigned char>(raw[at])) != 0) {
        at++;
    }
    return at < raw.size() && raw[at] == ']';
}

// Python float() of one JSON value: numbers, booleans and numeric strings.
std::optional<double> python_float(const nlohmann::json& value) {
    if (value.is_boolean()) {
        return value.get<bool>() ? 1.0 : 0.0;
    }
    if (value.is_number()) {
        return value.get<double>();
    }
    if (!value.is_string()) {
        return std::nullopt;
    }
    std::string text = trim(value.get<std::string>());
    if (!text.empty() && text.front() == '+') {
        text.erase(0, 1);
    }
    if (text.empty()) {
        return std::nullopt;
    }
    double out = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    if (ec != std::errc() || ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return out;
}

// _FENCED_CODE_PATTERN, a full match:
// ^```(?P<hint>[^`\r\n]*)\r?\n(?P<text>.*?)(?:\r?\n)?```[ \t]*$
bool fenced_code(const std::string& text, std::string* hint, std::string* body) {
    if (!text.starts_with("```")) {
        return false;
    }
    size_t i = 3;
    while (i < text.size() && text[i] != '`' && text[i] != '\r' && text[i] != '\n') {
        i++;
    }
    const std::string raw_hint = text.substr(3, i - 3);
    if (i < text.size() && text[i] == '\r') {
        i++;
    }
    if (i >= text.size() || text[i] != '\n') {
        return false;
    }
    i++;
    size_t end = text.size();
    while (end > i && (text[end - 1] == ' ' || text[end - 1] == '\t')) {
        end--;
    }
    if (end < i + 3 || text.compare(end - 3, 3, "```") != 0) {
        return false;
    }
    std::string inner = text.substr(i, end - 3 - i);
    if (inner.ends_with("\r\n")) {
        inner.resize(inner.size() - 2);
    } else if (inner.ends_with("\n")) {
        inner.pop_back();
    }
    *hint = trim(raw_hint);
    *body = std::move(inner);
    return true;
}

std::optional<docv1::DocItemLabel> category_label(const std::string& category) {
    static const std::map<std::string, docv1::DocItemLabel> kLabels = {
        {"Text", docv1::DOC_ITEM_LABEL_TEXT},
        {"Title", docv1::DOC_ITEM_LABEL_TITLE},
        {"Section-header", docv1::DOC_ITEM_LABEL_SECTION_HEADER},
        {"Table", docv1::DOC_ITEM_LABEL_TABLE},
        {"Picture", docv1::DOC_ITEM_LABEL_PICTURE},
        {"Caption", docv1::DOC_ITEM_LABEL_CAPTION},
        {"Footnote", docv1::DOC_ITEM_LABEL_FOOTNOTE},
        {"Page-header", docv1::DOC_ITEM_LABEL_PAGE_HEADER},
        {"Page-footer", docv1::DOC_ITEM_LABEL_PAGE_FOOTER},
        {"List-item", docv1::DOC_ITEM_LABEL_LIST_ITEM},
        {"Formula", docv1::DOC_ITEM_LABEL_FORMULA},
    };
    const auto it = kLabels.find(category);
    if (it == kLabels.end()) {
        return std::nullopt;
    }
    return it->second;
}

// A heading or list item whose text docling fills from inline markup or,
// without any, from the unescaped text.
void fill_inline(LayoutWriter& w, Ref item, const std::string& text) {
    if (!add_inline_text(w, text, docv1::DOC_ITEM_LABEL_TEXT, std::nullopt, item)) {
        w.set_text(item, unescape_markdown_literals(text), nullptr, "");
    }
}

}  // namespace

bool map_dots_json(const std::string& text, const PageContext& page, docv1::Document* out,
                   std::string* error, std::vector<vlmv1::PageWarning>* warnings) {
    LayoutWriter w(page, out, warnings);
    if (trim(text).empty()) {
        return w.finish(error, "dots.ocr", EmptyPage::kBlankReply);
    }
    size_t dropped_tail = 0;
    const nlohmann::json elements =
        nlohmann::json::parse(clean_json(text, &dropped_tail), nullptr, false);
    if (elements.is_discarded() || !elements.is_array()) {
        *error = "dots.ocr response is not a JSON array of layout elements";
        return false;
    }
    // What the mapper reads past, reported beside the fragment: entries it
    // builds nothing from, and elements kept without their own box.
    size_t skipped = 0;
    size_t unplaced = 0;
    // Boxes are on the Qwen2-VL input grid of the page raster.
    const auto [model_w, model_h] = qwen2vl_image_size(page.width, page.height);
    const double sx = model_w > 0 ? page.width / model_w : 1.0;
    const double sy = model_h > 0 ? page.height / model_h : 1.0;

    std::optional<Ref> list;
    std::vector<size_t> starts;
    std::vector<size_t> caption_blocks;
    for (size_t index = 0; index < elements.size(); index++) {
        starts.push_back(w.body_size());
        const nlohmann::json& element = elements[index];
        if (!element.is_object()) {
            skipped++;
            continue;
        }
        const std::string category = element.contains("category") &&
                                             element["category"].is_string()
                                         ? element["category"].get<std::string>()
                                         : "";
        if (category == "Caption") {
            caption_blocks.push_back(index);
        }
        const std::string body = element.contains("text") && element["text"].is_string()
                                     ? element["text"].get<std::string>()
                                     : "";
        // docling skips an element whose bbox is missing or is not four
        // finite numbers. One with text is kept here instead, on the
        // full-page box, and counted: its words are the page's content. One
        // with no text has nothing to keep.
        std::vector<double> coords;
        if (element.contains("bbox") && element["bbox"].is_array() &&
            element["bbox"].size() == 4) {
            for (const nlohmann::json& value : element["bbox"]) {
                const std::optional<double> number = python_float(value);
                if (!number || !std::isfinite(*number)) {
                    break;
                }
                coords.push_back(*number);
            }
        }
        Box box;
        if (coords.size() == 4) {
            box = w.scaled_box(coords[0], coords[1], coords[2], coords[3], sx, sy);
        } else if (trim(body).empty()) {
            skipped++;
            continue;
        } else {
            unplaced++;
        }
        const docv1::DocItemLabel label =
            category_label(category).value_or(docv1::DOC_ITEM_LABEL_TEXT);

        if (category != "List-item") {
            list.reset();
        }
        if (category == "Table") {
            if (!add_html_table(w, body, box, LayoutWriter::kBody)) {
                w.table(box, {});
            }
        } else if (category == "Picture") {
            w.picture(box);
        } else if (category == "Title" || category == "Section-header") {
            int level = 0;
            const std::string clean = strip_markdown_heading(body, &level);
            const Ref item = category == "Title" ? w.title(box, "")
                                                 : w.heading(level > 0 ? level : 2, box, "");
            w.set_orig(item, body);
            fill_inline(w, item, clean);
        } else if (category == "List-item") {
            if (!list) {
                list = w.list_group();
            }
            const Ref item = w.list_item(box, "", false, "", *list);
            w.set_orig(item, body);
            fill_inline(w, item, body);
        } else {
            std::string hint;
            std::string code;
            if (fenced_code(body, &hint, &code)) {
                const Ref item = w.code(box, code, hint);
                w.set_orig(item, body);
            } else if (category == "Formula") {
                w.formula(box, body);
            } else if (!add_inline_text(w, body, label, box, std::nullopt)) {
                const Ref item = w.styled_text(label, box, unescape_markdown_literals(body),
                                               nullptr, "");
                w.set_orig(item, body);
            }
        }
    }
    w.link_adjacent_captions(starts, caption_blocks, /*code_owners=*/true);
    if (dropped_tail > 0) {
        add_warning(warnings, vlmv1::PAGE_WARNING_CODE_OUTPUT_TRUNCATED,
                    "the layout array breaks off inside an element; the last " +
                        std::to_string(dropped_tail) + " byte(s) of the answer were not mapped",
                    "");
    }
    if (skipped > 0) {
        add_warning(warnings, vlmv1::PAGE_WARNING_CODE_ELEMENTS_SKIPPED,
                    std::to_string(skipped) + " of " + std::to_string(elements.size()) +
                        " layout entries were not mapped: not an element object, or an "
                        "element with neither a usable bbox nor text",
                    "");
    }
    if (unplaced > 0) {
        add_warning(warnings, vlmv1::PAGE_WARNING_CODE_LOCATION_UNREADABLE,
                    std::to_string(unplaced) +
                        " element(s) whose bbox is missing or is not four numbers keep their "
                        "text on the full-page box",
                    "");
    }
    return w.finish(error, "dots.ocr",
                    declares_empty_layout(text) ? EmptyPage::kDeclaredEmpty
                                                : EmptyPage::kMappingFailure);
}

}  // namespace vlm::mapping
