// DocLang XML (docling-core DocLangDocDeserializer, as docling v2.142.1's
// VLM pipeline calls it): the first <doclang>...</doclang> fragment of the
// response is read. Element heads carry <location value=.../> quartets on
// the <default_resolution> grid (512 by default), <layer value=.../> and
// <label value=.../>; bodies are text runs, formatting tags, lists split
// by <ldiv/>, OTSL tables, pictures and float groups with their caption
// and footnote satellites.
//
// Not carried (each documented in docs/design.md): <thread> continuation
// across elements (each piece stays its own item), <track>, <page_break>
// (one response is one page), <custom> metadata, archive <src> images,
// rich table cells (a cell's elements are flattened to text), and the
// field_* form vocabulary (walked as plain content).

#include <cctype>
#include <charconv>
#include <optional>
#include <set>

#include "layout.h"

namespace vlm::mapping {

namespace {

using Ref = LayoutWriter::Ref;
using Box = std::optional<docv1::BoundingBox>;

const std::set<std::string> kHeadTags = {"location", "layer", "label", "thread",
                                         "custom",   "hour",  "minute", "second",
                                         "centisecond"};
const std::set<std::string> kFormatTags = {"bold",      "italic",      "underline",
                                           "strikethrough", "subscript", "superscript"};
const std::set<std::string> kOtslTokens = {"fcel", "ecel", "lcel", "ucel", "xcel",
                                           "nl",   "ched", "rhed", "srow", "corn"};

bool blank(const std::string& text) {
    return text.find_first_not_of(" \t\r\n") == std::string::npos;
}

std::optional<int> parse_int(const std::string& text) {
    const std::string t = trim(text);
    int value = 0;
    const auto [ptr, ec] = std::from_chars(t.data(), t.data() + t.size(), value);
    if (t.empty() || ec != std::errc() || ptr != t.data() + t.size()) {
        return std::nullopt;
    }
    return value;
}

void merge_format(const std::string& tag, docv1::Formatting* f) {
    if (tag == "bold") {
        f->set_bold(true);
    } else if (tag == "italic") {
        f->set_italic(true);
    } else if (tag == "underline") {
        f->set_underline(true);
    } else if (tag == "strikethrough") {
        f->set_strikethrough(true);
    } else if (tag == "subscript") {
        f->set_script(docv1::SCRIPT_SUB);
    } else if (tag == "superscript") {
        f->set_script(docv1::SCRIPT_SUPER);
    }
}

std::optional<docv1::ContentLayer> layer_of(const std::string& value) {
    static const std::map<std::string, docv1::ContentLayer> kLayers = {
        {"body", docv1::CONTENT_LAYER_BODY},
        {"furniture", docv1::CONTENT_LAYER_FURNITURE},
        {"background", docv1::CONTENT_LAYER_BACKGROUND},
        {"invisible", docv1::CONTENT_LAYER_INVISIBLE},
        {"notes", docv1::CONTENT_LAYER_NOTES},
    };
    const auto it = kLayers.find(value);
    if (it == kLayers.end()) {
        return std::nullopt;
    }
    return it->second;
}

class DocLangBuilder {
  public:
    DocLangBuilder(LayoutWriter& w, const PageContext& page, docv1::Document* out)
        : w_(w), page_(page), out_(out) {}

    void set_resolution(const MarkupNode& root) {
        const MarkupNode* res = find_element(root, "default_resolution");
        if (res == nullptr) {
            return;
        }
        const auto width = parse_int(res->attr("width", "0"));
        const auto height = parse_int(res->attr("height", "0"));
        if (!width || !height) {
            return;
        }
        if (*width > 0) {
            res_w_ = *width;
            res_ = *width;
        }
        if (*height > 0) {
            res_h_ = *height;
        }
    }

    void walk_root(const MarkupNode& root) {
        for (const auto& child : root.children) {
            if (!child->is_text()) {
                dispatch(*child, LayoutWriter::kBody);
            }
        }
    }

  private:
    // ---- element heads ----------------------------------------------------

    static std::vector<const MarkupNode*> head_nodes(const MarkupNode& el) {
        std::vector<const MarkupNode*> head;
        for (const auto& child : el.children) {
            if (child->is_text()) {
                if (blank(child->text)) {
                    continue;
                }
                break;
            }
            if (!kHeadTags.contains(child->tag)) {
                break;
            }
            head.push_back(child.get());
        }
        return head;
    }

    static std::vector<const MarkupNode*> body_nodes(const MarkupNode& el) {
        std::vector<const MarkupNode*> body;
        bool in_body = false;
        for (const auto& child : el.children) {
            if (!in_body && ((child->is_text() && blank(child->text)) ||
                             (!child->is_text() && kHeadTags.contains(child->tag)))) {
                continue;
            }
            in_body = true;
            body.push_back(child.get());
        }
        return body;
    }

    // The first <location> quartet of a node run, on the page.
    Box box_of(const std::vector<const MarkupNode*>& nodes) const {
        std::vector<double> values;
        double res_x = res_w_;
        double res_y = res_h_;
        for (const MarkupNode* node : nodes) {
            if (node->is_text() || node->tag != "location") {
                continue;
            }
            values.push_back(parse_int(node->attr("value", "0")).value_or(0));
            if (node->has_attr("resolution")) {
                const int r = parse_int(node->attr("resolution")).value_or(res_);
                res_x = r;
                res_y = r;
            }
            if (values.size() == 4) {
                const double l = std::min(values[0], values[2]);
                const double t = std::min(values[1], values[3]);
                const double r = std::max(values[0], values[2]);
                const double b = std::max(values[1], values[3]);
                return w_.scaled_box(l, t, r, b, res_x > 0 ? page_.width / res_x : 1.0,
                                     res_y > 0 ? page_.height / res_y : 1.0);
            }
        }
        return std::nullopt;
    }

    Box head_box(const MarkupNode& el) const { return box_of(head_nodes(el)); }

    static std::optional<docv1::ContentLayer> head_layer(const MarkupNode& el) {
        for (const MarkupNode* node : head_nodes(el)) {
            if (node->tag == "layer") {
                return layer_of(node->attr("value"));
            }
        }
        return std::nullopt;
    }

    static std::string head_label(const MarkupNode& el) {
        for (const MarkupNode* node : head_nodes(el)) {
            if (node->tag == "label") {
                return node->attr("value");
            }
        }
        return "";
    }

    void apply_layer(Ref ref, const MarkupNode& el) {
        if (const auto layer = head_layer(el)) {
            w_.set_content_layer(ref, *layer);
        }
    }

    // ---- text ---------------------------------------------------------------

    static std::string get_text(const MarkupNode& el) {
        std::string out;
        for (const auto& child : el.children) {
            if (child->is_text()) {
                if (!blank(child->text)) {
                    out += el.tag == "content" ? child->text : trim(child->text);
                }
            } else if (child->tag == "location" || child->tag == "custom") {
                continue;
            } else if (child->tag == "br") {
                out += "\n";
            } else {
                out += get_text(*child);
            }
        }
        return out;
    }

    // _get_children_simple_text_block: the one text run of an element whose
    // children are only inline wrappers, or nullopt.
    static std::optional<std::string> simple_text(const MarkupNode& el) {
        static const std::set<std::string> kSimple = {
            "location", "layer",       "label",       "br",  "bold",       "italic",
            "underline", "strikethrough", "subscript", "superscript", "rtl", "handwriting",
            "checkbox", "content"};
        std::optional<std::string> result;
        for (const auto& child : el.children) {
            if (!child->is_text()) {
                if (kHeadTags.contains(child->tag)) {
                    continue;
                }
                if (!kSimple.contains(child->tag)) {
                    return std::nullopt;
                }
                if (const auto inner = simple_text(*child); inner && !inner->empty()) {
                    if (result) {
                        return std::nullopt;
                    }
                    result = inner;
                } else if (!blank(get_text(*child))) {
                    return std::nullopt;
                }
            } else if (!blank(child->text)) {
                if (result) {
                    return std::nullopt;
                }
                result = el.tag == "content" ? child->text : trim(child->text);
            }
        }
        return result;
    }

    static bool empty_block(const MarkupNode& el) {
        for (const auto& child : el.children) {
            if (child->is_text() ? !blank(child->text) : !kHeadTags.contains(child->tag)) {
                return false;
            }
        }
        return true;
    }

    // _extract_text_with_formatting: a lone formatting child (recursively)
    // becomes the run's formatting.
    static std::string text_with_formatting(const MarkupNode& el, docv1::Formatting* f,
                                            bool* formatted) {
        std::vector<const MarkupNode*> elements;
        for (const auto& child : el.children) {
            if (!child->is_text() && child->tag != "location") {
                elements.push_back(child.get());
            }
        }
        if (elements.size() == 1 &&
            (kFormatTags.contains(elements[0]->tag) || elements[0]->tag == "rtl")) {
            const std::string text = text_with_formatting(*elements[0], f, formatted);
            merge_format(elements[0]->tag, f);
            *formatted = true;
            return text;
        }
        return get_text(el);
    }

    void inline_runs(const std::vector<const MarkupNode*>& nodes, Ref parent,
                     const docv1::Formatting* formatting) {
        for (const MarkupNode* node : nodes) {
            if (node->is_text()) {
                const std::string text = trim(node->text);
                if (!text.empty()) {
                    w_.styled_text(docv1::DOC_ITEM_LABEL_TEXT, std::nullopt, text, formatting, "",
                                   parent);
                }
            } else if (kFormatTags.contains(node->tag)) {
                docv1::Formatting merged;
                if (formatting != nullptr) {
                    merged = *formatting;
                }
                merge_format(node->tag, &merged);
                inline_runs(children(*node), parent, &merged);
            } else {
                dispatch(*node, parent);
            }
        }
    }

    static std::vector<const MarkupNode*> children(const MarkupNode& el) {
        std::vector<const MarkupNode*> out;
        for (const auto& child : el.children) {
            out.push_back(child.get());
        }
        return out;
    }

    void layer_subtree(int texts_before, int groups_before, const MarkupNode& el) {
        const auto layer = head_layer(el);
        if (!layer) {
            return;
        }
        for (int i = texts_before; i < out_->texts_size(); i++) {
            w_.set_content_layer(Ref{BodyChild::TEXT, i}, *layer);
        }
        for (int i = groups_before; i < out_->groups_size(); i++) {
            w_.set_content_layer(Ref{BodyChild::GROUP, i}, *layer);
        }
    }

    Ref inline_host(const MarkupNode& el, Ref parent, docv1::DocItemLabel label) {
        const int texts_before = out_->texts_size();
        const int groups_before = out_->groups_size();
        const Ref host = w_.styled_text(label, head_box(el), "", nullptr, "", parent);
        const Ref group = w_.inline_group(host);
        inline_runs(body_nodes(el), group, nullptr);
        layer_subtree(texts_before, groups_before, el);
        return host;
    }

    static std::optional<docv1::DocItemLabel> text_label(const std::string& tag) {
        static const std::map<std::string, docv1::DocItemLabel> kLabels = {
            {"text", docv1::DOC_ITEM_LABEL_TEXT},
            {"caption", docv1::DOC_ITEM_LABEL_CAPTION},
            {"footnote", docv1::DOC_ITEM_LABEL_FOOTNOTE},
            {"page_header", docv1::DOC_ITEM_LABEL_PAGE_HEADER},
            {"page_footer", docv1::DOC_ITEM_LABEL_PAGE_FOOTER},
            {"bold", docv1::DOC_ITEM_LABEL_TEXT},
            {"italic", docv1::DOC_ITEM_LABEL_TEXT},
            {"underline", docv1::DOC_ITEM_LABEL_TEXT},
            {"strikethrough", docv1::DOC_ITEM_LABEL_TEXT},
            {"subscript", docv1::DOC_ITEM_LABEL_TEXT},
            {"superscript", docv1::DOC_ITEM_LABEL_TEXT},
            {"rtl", docv1::DOC_ITEM_LABEL_TEXT},
            {"content", docv1::DOC_ITEM_LABEL_TEXT},
        };
        const auto it = kLabels.find(tag);
        if (it == kLabels.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    std::vector<Ref> captions_of(const MarkupNode& el) {
        std::vector<Ref> captions;
        for (const auto& child : el.children) {
            if (child->is_text() || child->tag != "caption") {
                continue;
            }
            const std::string text = trim(get_text(*child));
            if (text.empty()) {
                continue;
            }
            if (!simple_text(*child)) {
                captions.push_back(inline_host(*child, LayoutWriter::kBody,
                                               docv1::DOC_ITEM_LABEL_CAPTION));
                continue;
            }
            const Ref caption =
                w_.styled_text(docv1::DOC_ITEM_LABEL_CAPTION, head_box(*child), text, nullptr, "");
            apply_layer(caption, *child);
            captions.push_back(caption);
        }
        return captions;
    }

    void text_like(const MarkupNode& el, Ref parent) {
        size_t element_children = 0;
        std::vector<const MarkupNode*> elements;
        for (const auto& child : el.children) {
            if (!child->is_text() && !kHeadTags.contains(child->tag)) {
                element_children++;
                elements.push_back(child.get());
            }
        }
        const std::optional<std::string> simple = simple_text(el);
        const bool is_empty = !simple && empty_block(el);
        if (!is_empty && (element_children > 1 || !simple)) {
            static const std::set<std::string> kHosts = {"text", "caption", "footnote",
                                                         "page_header", "page_footer"};
            if (kHosts.contains(el.tag) && (el.tag != "text" || head_box(el))) {
                inline_host(el, parent, *text_label(el.tag));
                return;
            }
            const int texts_before = out_->texts_size();
            const int groups_before = out_->groups_size();
            const Ref group = w_.inline_group(parent);
            docv1::Formatting f;
            if (kFormatTags.contains(el.tag)) {
                merge_format(el.tag, &f);
            }
            inline_runs(children(el), group, kFormatTags.contains(el.tag) ? &f : nullptr);
            layer_subtree(texts_before, groups_before, el);
            return;
        }
        const Box box = head_box(el);
        docv1::Formatting formatting;
        bool formatted = false;
        const std::string text = text_with_formatting(el, &formatting, &formatted);
        if (text.empty() && !is_empty) {
            return;
        }
        if (el.tag == "code") {
            std::string lang = head_label(el);
            std::string body;
            for (const auto& child : el.children) {
                if (child->is_text()) {
                    if (!blank(child->text)) {
                        body += child->text;
                    }
                } else if (child->tag == "location" || child->tag == "layer" ||
                           child->tag == "label" || child->tag == "caption") {
                    continue;
                } else if (child->tag == "br") {
                    body += "\n";
                } else {
                    body += get_text(*child);
                }
            }
            const Ref item = w_.code(box, body, lang, parent);
            apply_layer(item, el);
            for (const Ref caption : captions_of(el)) {
                w_.link_caption(item, caption);
            }
            last_code_ = item;
            return;
        }
        if (el.tag == "formula") {
            apply_layer(w_.formula(box, text, parent), el);
            return;
        }
        std::optional<docv1::DocItemLabel> label = text_label(el.tag);
        if (!label) {
            return;
        }
        if (kFormatTags.contains(el.tag)) {
            merge_format(el.tag, &formatting);
            formatted = true;
        }
        if (el.tag == "text") {
            for (const MarkupNode* child : elements) {
                if (child->tag == "handwriting") {
                    label = docv1::DOC_ITEM_LABEL_HANDWRITTEN_TEXT;
                    break;
                }
                if (child->tag == "checkbox") {
                    const std::string cls = child->attr("class");
                    if (cls == "selected") {
                        label = docv1::DOC_ITEM_LABEL_CHECKBOX_SELECTED;
                        break;
                    }
                    if (cls == "unselected") {
                        label = docv1::DOC_ITEM_LABEL_CHECKBOX_UNSELECTED;
                        break;
                    }
                }
            }
        }
        const Ref item = w_.styled_text(*label, box, text, formatted ? &formatting : nullptr, "",
                                        parent);
        apply_layer(item, el);
    }

    void heading(const MarkupNode& el, Ref parent) {
        const int level = parse_int(el.attr("level", "1")).value_or(1);
        const std::string text = trim(get_text(el));
        if (text.empty()) {
            return;
        }
        const Box box = head_box(el);
        const Ref item = level == 1 ? w_.title(box, text, parent)
                                    : w_.heading(level - 1, box, text, parent);
        apply_layer(item, el);
    }

    // ---- lists --------------------------------------------------------------

    void list(const MarkupNode& el, Ref parent) {
        const bool ordered = el.attr("class") == "ordered";
        const Ref group = w_.list_group(parent);
        // Items run from each <ldiv> to the next.
        std::vector<size_t> ldivs;
        for (size_t i = 0; i < el.children.size(); i++) {
            if (!el.children[i]->is_text() && el.children[i]->tag == "ldiv") {
                ldivs.push_back(i);
            }
        }
        for (size_t k = 0; k < ldivs.size(); k++) {
            const MarkupNode& ldiv = *el.children[ldivs[k]];
            std::string marker;
            for (const auto& child : ldiv.children) {
                if (!child->is_text() && child->tag == "marker") {
                    marker = trim(get_text(*child));
                    break;
                }
            }
            const size_t end = k + 1 < ldivs.size() ? ldivs[k + 1] : el.children.size();
            std::vector<const MarkupNode*> nodes;
            for (size_t i = ldivs[k] + 1; i < end; i++) {
                nodes.push_back(el.children[i].get());
            }
            list_item(nodes, group, ordered, marker);
        }
    }

    void list_item(const std::vector<const MarkupNode*>& nodes, Ref group, bool ordered,
                   const std::string& marker) {
        std::vector<const MarkupNode*> content;
        bool skipping = true;
        for (const MarkupNode* node : nodes) {
            if (skipping && ((node->is_text() && blank(node->text)) ||
                             (!node->is_text() && kHeadTags.contains(node->tag)))) {
                continue;
            }
            skipping = false;
            content.push_back(node);
        }
        const Box box = box_of(nodes);
        if (content.empty()) {
            w_.list_item(box, "", ordered, marker, group);
            return;
        }
        static const std::set<std::string> kBlocks = {"text",   "heading", "list",    "group",
                                                      "table",  "index",   "picture", "code",
                                                      "formula", "caption", "footnote",
                                                      "page_header", "page_footer"};
        const MarkupNode* first = content.front();
        const bool virtual_text = first->is_text() || !kBlocks.contains(first->tag);
        if (virtual_text) {
            // Leading text, then nested lists or pictures, or inline runs.
            std::string leading;
            size_t rest = 0;
            for (; rest < content.size(); rest++) {
                const MarkupNode* node = content[rest];
                if (node->is_text()) {
                    leading += node->text;
                } else if (node->tag == "content") {
                    leading += get_text(*node);
                } else {
                    break;
                }
            }
            leading = trim(leading);
            std::vector<const MarkupNode*> rest_elements;
            bool only_nested = true;
            for (size_t i = rest; i < content.size(); i++) {
                if (content[i]->is_text()) {
                    if (!blank(content[i]->text)) {
                        only_nested = false;
                    }
                    continue;
                }
                rest_elements.push_back(content[i]);
                if (content[i]->tag != "list" && content[i]->tag != "picture") {
                    only_nested = false;
                }
            }
            if (!leading.empty() && only_nested) {
                const Ref item = w_.list_item(box, leading, ordered, marker, group);
                for (const MarkupNode* child : rest_elements) {
                    dispatch(*child, item);
                }
                return;
            }
            const Ref item = w_.list_item(box, "", ordered, marker, group);
            const Ref inline_group = w_.inline_group(item);
            inline_runs(content, inline_group, nullptr);
            return;
        }
        std::vector<const MarkupNode*> elements;
        for (const MarkupNode* node : content) {
            if (!node->is_text()) {
                elements.push_back(node);
            }
        }
        if (elements.size() == 1 && elements[0]->tag == "text" && simple_text(*elements[0])) {
            w_.list_item(head_box(*elements[0]), trim(get_text(*elements[0])), ordered, marker,
                         group);
            return;
        }
        const Ref item = w_.list_item(box, "", ordered, marker, group);
        for (const MarkupNode* child : elements) {
            dispatch(*child, item);
        }
    }

    // ---- floats -------------------------------------------------------------

    static const MarkupNode* first_child(const MarkupNode& el, const std::string& tag) {
        for (const auto& child : el.children) {
            if (!child->is_text() && child->tag == tag) {
                return child.get();
            }
        }
        return nullptr;
    }

    static const MarkupNode* float_host(const MarkupNode& el) {
        static const std::set<std::string> kHosts = {"picture", "table", "index", "code"};
        static const std::set<std::string> kSatellites = {"text", "footnote", "caption"};
        const MarkupNode* host = nullptr;
        size_t hosts = 0;
        for (const auto& child : el.children) {
            if (child->is_text()) {
                continue;
            }
            if (kHosts.contains(child->tag)) {
                host = child.get();
                hosts++;
            } else if (!kSatellites.contains(child->tag)) {
                return nullptr;
            }
        }
        return hosts == 1 ? host : nullptr;
    }

    // <text> siblings are extra captions, <footnote> siblings footnotes.
    void satellites(const MarkupNode& el, std::vector<Ref>* captions, std::vector<Ref>* footnotes) {
        for (const auto& child : el.children) {
            if (child->is_text() || (child->tag != "text" && child->tag != "footnote")) {
                continue;
            }
            const docv1::DocItemLabel label = child->tag == "text"
                                                  ? docv1::DOC_ITEM_LABEL_CAPTION
                                                  : docv1::DOC_ITEM_LABEL_FOOTNOTE;
            const std::string text = trim(get_text(*child));
            if (text.empty()) {
                continue;
            }
            const Ref item = w_.styled_text(label, head_box(*child), text, nullptr, "");
            (child->tag == "text" ? captions : footnotes)->push_back(item);
        }
    }

    void attach(Ref owner, const std::vector<Ref>& captions, const std::vector<Ref>& footnotes) {
        for (const Ref caption : captions) {
            w_.link_caption(owner, caption);
        }
        for (const Ref footnote : footnotes) {
            w_.link_footnote(owner, footnote);
        }
    }

    // OTSL cells of a <table>/<tabular> body (_otsl_parse_texts): spans
    // from the lcel/xcel run to the right and the ucel/xcel run below.
    std::vector<SpanCell> otsl_cells(const std::vector<const MarkupNode*>& body) {
        std::vector<std::vector<std::string>> rows(1);
        for (const MarkupNode* node : body) {
            if (node->is_text() || !kOtslTokens.contains(node->tag)) {
                continue;
            }
            if (node->tag == "nl") {
                rows.emplace_back();
            } else {
                rows.back().push_back(node->tag);
            }
        }
        std::erase_if(rows, [](const std::vector<std::string>& row) { return row.empty(); });
        std::vector<SpanCell> cells;
        size_t r = 0;
        size_t c = 0;
        static const std::set<std::string> kOrigins = {"fcel", "ecel", "ched", "rhed", "srow",
                                                       "corn"};
        static const std::set<std::string> kContinuations = {"lcel", "ucel", "xcel"};
        auto token_at = [&](size_t i) -> std::string {
            const MarkupNode* node = body[i];
            return !node->is_text() && kOtslTokens.contains(node->tag) ? node->tag : "";
        };
        bool row_has_cells = false;
        for (size_t i = 0; i < body.size(); i++) {
            const std::string t = token_at(i);
            if (t.empty()) {
                continue;
            }
            if (t == "nl") {
                if (row_has_cells) {
                    r++;
                }
                c = 0;
                row_has_cells = false;
                continue;
            }
            row_has_cells = true;
            std::string text;
            bool parts = false;
            size_t next = i + 1;
            if (t != "ecel") {
                while (next < body.size() && token_at(next).empty()) {
                    const MarkupNode* part = body[next];
                    if (part->is_text()) {
                        text += trim(part->text);
                        parts = parts || !blank(part->text);
                    } else if (part->tag != "location") {
                        text += get_text(*part);
                        parts = true;
                    }
                    next++;
                }
            }
            const bool continuation = kContinuations.contains(t);
            if (!(continuation && trim(text).empty() && !parts)) {
                SpanCell cell;
                cell.row = r;
                cell.col = c;
                cell.text = trim(text);
                cell.column_header = t == "ched" || t == "corn";
                cell.row_header = t == "rhed" || t == "corn";
                const std::string right = next < body.size() ? token_at(next) : "";
                if (right == "lcel" || right == "xcel") {
                    for (size_t j = c + 1; r < rows.size() && j < rows[r].size() &&
                                           (rows[r][j] == "lcel" || rows[r][j] == "xcel");
                         j++) {
                        cell.col_span++;
                    }
                }
                if (r + 1 < rows.size() && c < rows[r + 1].size() &&
                    (rows[r + 1][c] == "ucel" || rows[r + 1][c] == "xcel")) {
                    for (size_t k = r + 1; k < rows.size() && c < rows[k].size() &&
                                           (rows[k][c] == "ucel" || rows[k][c] == "xcel");
                         k++) {
                        cell.row_span++;
                    }
                }
                cells.push_back(std::move(cell));
            }
            if (kOrigins.contains(t) || continuation) {
                c++;
            }
        }
        return cells;
    }

    void table(const MarkupNode& el, Ref parent) {
        const MarkupNode* otsl = &el;
        std::vector<Ref> captions;
        std::vector<Ref> footnotes;
        if (el.tag != "table" && el.tag != "index") {
            satellites(el, &captions, &footnotes);
            otsl = first_child(el, "table");
            if (otsl == nullptr) {
                otsl = first_child(el, "index");
            }
            std::vector<Ref> own = captions_of(el);
            if (own.empty() && otsl != nullptr) {
                own = captions_of(*otsl);
            }
            own.insert(own.end(), captions.begin(), captions.end());
            captions = std::move(own);
            if (otsl == nullptr) {
                const Ref t = w_.table(std::nullopt, {}, parent);
                attach(t, captions, footnotes);
                return;
            }
        } else {
            captions = captions_of(el);
        }
        const Ref t = w_.table(head_box(*otsl), otsl_cells(body_nodes(*otsl)), parent);
        if (otsl->tag == "index") {
            w_.table_item(t)->set_label(docv1::DOC_ITEM_LABEL_DOCUMENT_INDEX);
        }
        apply_layer(t, *otsl);
        attach(t, captions, footnotes);
    }

    void picture(const MarkupNode& el, Ref parent) {
        const MarkupNode* host = &el;
        std::vector<Ref> captions;
        std::vector<Ref> footnotes;
        if (el.tag != "picture") {
            satellites(el, &captions, &footnotes);
            host = first_child(el, "picture");
            std::vector<Ref> own = captions_of(el);
            if (own.empty() && host != nullptr) {
                own = captions_of(*host);
            }
            own.insert(own.end(), captions.begin(), captions.end());
            captions = std::move(own);
        } else {
            captions = captions_of(el);
        }
        const Ref pic = w_.picture(host != nullptr ? head_box(*host) : std::nullopt, parent);
        attach(pic, captions, footnotes);
        if (host == nullptr) {
            return;
        }
        apply_layer(pic, *host);
        const std::string label = head_label(*host);
        if (!label.empty()) {
            w_.classify_picture(pic, label, "doclang");
        }
        const std::vector<const MarkupNode*> body = body_nodes(*host);
        size_t i = 0;
        for (; i < body.size(); i++) {
            const MarkupNode* node = body[i];
            if (node->is_text()) {
                continue;
            }
            if (node->tag == "src") {
                continue;
            }
            if (node->tag == "tabular") {
                std::vector<std::vector<std::string>> rows;
                for (const SpanCell& cell : otsl_cells(body_nodes(*node))) {
                    if (rows.size() <= cell.row) {
                        rows.resize(cell.row + 1);
                    }
                    if (rows[cell.row].size() <= cell.col) {
                        rows[cell.row].resize(cell.col + 1);
                    }
                    rows[cell.row][cell.col] = cell.text;
                }
                w_.set_tabular_chart(pic, "", rows, "doclang");
                continue;
            }
            break;
        }
        for (; i < body.size(); i++) {
            if (!body[i]->is_text() && body[i]->tag != "caption") {
                dispatch(*body[i], pic);
            }
        }
    }

    void code_group(const MarkupNode& el, const MarkupNode& code, Ref parent) {
        std::vector<Ref> captions;
        std::vector<Ref> footnotes;
        satellites(el, &captions, &footnotes);
        last_code_.reset();
        text_like(code, parent);
        if (last_code_) {
            attach(*last_code_, captions, footnotes);
        }
    }

    // ---- dispatch -----------------------------------------------------------

    void walk_children(const MarkupNode& el, Ref parent) {
        static const std::set<std::string> kSkip = {"head",   "location", "layer", "label",
                                                    "custom", "caption",  "src"};
        for (const auto& child : el.children) {
            if (!child->is_text() && !kSkip.contains(child->tag)) {
                dispatch(*child, parent);
            }
        }
    }

    void dispatch(const MarkupNode& el, Ref parent) {
        static const std::set<std::string> kTextLike = {
            "text",   "caption", "footnote",      "page_header", "page_footer",
            "code",   "formula", "ldiv",          "bold",        "italic",
            "underline", "strikethrough", "subscript", "superscript", "content"};
        const std::string& name = el.tag;
        if (name == "track" || name == "page_break") {
            return;
        }
        if (kTextLike.contains(name)) {
            text_like(el, parent);
        } else if (name == "heading") {
            heading(el, parent);
        } else if (name == "list") {
            list(el, parent);
        } else if (name == "group") {
            if (const std::string group_name = el.attr("name"); !group_name.empty()) {
                const std::string label = head_label(el);
                docv1::GroupLabel group_label = docv1::GROUP_LABEL_UNSPECIFIED;
                static const std::map<std::string, docv1::GroupLabel> kLabels = {
                    {"list", docv1::GROUP_LABEL_LIST},
                    {"ordered_list", docv1::GROUP_LABEL_ORDERED_LIST},
                    {"chapter", docv1::GROUP_LABEL_CHAPTER},
                    {"section", docv1::GROUP_LABEL_SECTION},
                    {"sheet", docv1::GROUP_LABEL_SHEET},
                    {"slide", docv1::GROUP_LABEL_SLIDE},
                    {"form_area", docv1::GROUP_LABEL_FORM_AREA},
                    {"key_value_area", docv1::GROUP_LABEL_KEY_VALUE_AREA},
                    {"comment_section", docv1::GROUP_LABEL_COMMENT_SECTION},
                    {"inline", docv1::GROUP_LABEL_INLINE},
                    {"picture_area", docv1::GROUP_LABEL_PICTURE_AREA},
                };
                if (const auto it = kLabels.find(label); it != kLabels.end()) {
                    group_label = it->second;
                }
                walk_children(el, w_.group(group_name, group_label, parent));
            } else if (const MarkupNode* host = float_host(el)) {
                if (host->tag == "picture") {
                    picture(el, parent);
                } else if (host->tag == "code") {
                    code_group(el, *host, parent);
                } else {
                    table(el, parent);
                }
            } else {
                walk_children(el, parent);
            }
        } else if (name == "table" || name == "index") {
            table(el, parent);
        } else if (name == "picture") {
            picture(el, parent);
        } else {
            walk_children(el, parent);
        }
    }

    LayoutWriter& w_;
    const PageContext& page_;
    docv1::Document* out_;
    double res_w_ = 512;
    double res_h_ = 512;
    int res_ = 512;
    std::optional<Ref> last_code_;
};

// _DOCLANG_OPEN_RE then the first </doclang> after "<doclang".
std::optional<std::string> doclang_fragment(const std::string& text) {
    bool open = false;
    for (size_t at = text.find("<doclang"); at != std::string::npos;
         at = text.find("<doclang", at + 1)) {
        const size_t after = at + 8;
        if (after < text.size() &&
            (text[after] == '>' || std::isspace(static_cast<unsigned char>(text[after])) != 0)) {
            if (text[after] == '>' || text.find('>', after) != std::string::npos) {
                open = true;
                break;
            }
        }
    }
    if (!open) {
        return std::nullopt;
    }
    const size_t start = text.find("<doclang");
    const size_t end = text.find("</doclang>", start);
    if (end == std::string::npos) {
        return std::nullopt;
    }
    return text.substr(start, end + 10 - start);
}

}  // namespace

bool map_doclang(const std::string& text, const PageContext& page, docv1::Document* out,
                 std::string* error, std::vector<vlmv1::PageWarning>* warnings) {
    const std::optional<std::string> fragment = doclang_fragment(text);
    if (!fragment) {
        *error = "No <doclang> XML fragment found in VLM response.";
        return false;
    }
    const std::unique_ptr<MarkupNode> parsed = parse_markup(*fragment, MarkupMode::kXml);
    const MarkupNode* root = find_element(*parsed, "doclang");
    if (root == nullptr) {
        *error = "No <doclang> XML fragment found in VLM response.";
        return false;
    }
    LayoutWriter w(page, out, warnings);
    DocLangBuilder builder(w, page, out);
    builder.set_resolution(*root);
    builder.walk_root(*root);
    return w.finish(error, "DocLang", /*allow_empty=*/true);
}

}  // namespace vlm::mapping
