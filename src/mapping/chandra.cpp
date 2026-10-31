// Chandra layout HTML (docling chandra_utils.py, v2.142.1). Each layout
// block is an element carrying data-label and/or data-bbox ("x0 y0 x1
// y1" on a 0..1000 grid); its HTML content maps to document primitives:
// formatted inline runs, headings, lists, code, formulas, form regions,
// pictures (alt text as the description, <chem> as a molecule) and span
// tables with rich cells and nested captions. A block's box goes to the
// one item that represents it; derived children carry the page box (the
// service's convention for items without coordinates).

#include <algorithm>
#include <cctype>
#include <charconv>
#include <optional>
#include <functional>
#include <set>

#include "layout.h"

namespace vlm::mapping {

namespace {

using Ref = LayoutWriter::Ref;
using Box = std::optional<docv1::BoundingBox>;

const std::set<std::string> kBlockTags = {"div", "p",     "h1",  "h2",     "h3",      "h4",
                                          "h5",  "h6",    "table", "ul",   "ol",      "li",
                                          "pre", "figure", "form", "caption", "figcaption",
                                          "hr"};

bool is_block(const MarkupNode& node) {
    return !node.is_text() && kBlockTags.contains(node.tag);
}

bool is_heading_tag(const std::string& tag) {
    return tag.size() == 2 && tag[0] == 'h' && tag[1] >= '1' && tag[1] <= '6';
}

// Any descendant element matching `pred` (chandra _Element.elements()).
template <typename Pred>
bool any_descendant(const MarkupNode& node, Pred pred) {
    for (const auto& child : node.children) {
        if (child->is_text()) {
            continue;
        }
        if (pred(*child) || any_descendant(*child, pred)) {
            return true;
        }
    }
    return false;
}

template <typename Fn>
void for_each_descendant(const MarkupNode& node, Fn fn) {
    for (const auto& child : node.children) {
        if (child->is_text()) {
            continue;
        }
        fn(*child);
        for_each_descendant(*child, fn);
    }
}

std::string lower(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::optional<docv1::DocItemLabel> layout_label_of(const std::string& name) {
    static const std::map<std::string, docv1::DocItemLabel> kLabels = {
        {"Title", docv1::DOC_ITEM_LABEL_TITLE},
        {"Section-Header", docv1::DOC_ITEM_LABEL_SECTION_HEADER},
        {"Caption", docv1::DOC_ITEM_LABEL_CAPTION},
        {"Footnote", docv1::DOC_ITEM_LABEL_FOOTNOTE},
        {"Page-Header", docv1::DOC_ITEM_LABEL_PAGE_HEADER},
        {"Page-Footer", docv1::DOC_ITEM_LABEL_PAGE_FOOTER},
        {"Equation-Block", docv1::DOC_ITEM_LABEL_FORMULA},
        {"Code-Block", docv1::DOC_ITEM_LABEL_CODE},
        {"Bibliography", docv1::DOC_ITEM_LABEL_REFERENCE},
    };
    const auto it = kLabels.find(name);
    if (it == kLabels.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool python_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::string strip_chars(const std::string& text, std::string_view chars) {
    const size_t first = text.find_first_not_of(chars);
    if (first == std::string::npos) {
        return "";
    }
    return text.substr(first, text.find_last_not_of(chars) - first + 1);
}

std::string py_strip(const std::string& text) {
    return strip_chars(text, " \t\n\r\f\v");
}

bool has_alnum(const std::string& text) {
    for (unsigned char c : text) {
        if (std::isalnum(c) != 0 || c >= 0x80) {
            return true;
        }
    }
    return false;
}

// One inline run (chandra _Run).
struct Run {
    std::string text;
    docv1::DocItemLabel label = docv1::DOC_ITEM_LABEL_TEXT;
    docv1::Formatting formatting;
    std::string hyperlink;

    bool plain_formatting() const { return formatting.SerializeAsString().empty(); }
    bool plain() const {
        return label == docv1::DOC_ITEM_LABEL_TEXT && plain_formatting() && hyperlink.empty();
    }
    bool same_style(const Run& other) const {
        return formatting.SerializeAsString() == other.formatting.SerializeAsString() &&
               hyperlink == other.hyperlink;
    }
};

// A block's box, handed to the first item that claims it (chandra
// _Provenance).
struct Provenance {
    Box box;
    bool text_owner = true;
    bool claimed = false;

    Box take() {
        if (!box || claimed) {
            return std::nullopt;
        }
        claimed = true;
        return box;
    }
};

std::string collapse_runs(const std::string& text) {
    // re.sub(r"\s+", " ", text)
    std::string out;
    bool space = false;
    for (char c : text) {
        if (python_space(c)) {
            space = true;
            continue;
        }
        if (space) {
            out.push_back(' ');
            space = false;
        }
        out.push_back(c);
    }
    if (space) {
        out.push_back(' ');
    }
    return out;
}

// Set while a picture's content is walked and its one <chem> is already
// the molecule: the SMILES text is not repeated as a run.
thread_local bool t_skip_chem = false;

void inline_runs(const MarkupNode& node, const docv1::Formatting& inherited,
                 const std::string& inherited_link, std::vector<Run>* out) {
    if (node.is_text()) {
        Run run;
        run.text = collapse_runs(node.text);
        run.formatting = inherited;
        run.hyperlink = inherited_link;
        out->push_back(std::move(run));
        return;
    }
    docv1::Formatting formatting = inherited;
    std::string hyperlink = inherited_link;
    const std::string& tag = node.tag;
    if (tag == "b" || tag == "strong") {
        formatting.set_bold(true);
    } else if (tag == "i" || tag == "em") {
        formatting.set_italic(true);
    } else if (tag == "u") {
        formatting.set_underline(true);
    } else if (tag == "del" || tag == "s") {
        formatting.set_strikethrough(true);
    } else if (tag == "sup") {
        formatting.set_script(docv1::SCRIPT_SUPER);
    } else if (tag == "sub") {
        formatting.set_script(docv1::SCRIPT_SUB);
    } else if (tag == "a" && !node.attr("href").empty()) {
        hyperlink = node.attr("href");
    }
    if (tag == "input") {
        const std::string kind = lower(node.attr("type", "text"));
        if (kind == "hidden") {
            return;
        }
        Run run;
        const bool box = kind == "checkbox" || kind == "radio";
        run.label = box ? (node.has_attr("checked") ? docv1::DOC_ITEM_LABEL_CHECKBOX_SELECTED
                                                    : docv1::DOC_ITEM_LABEL_CHECKBOX_UNSELECTED)
                        : docv1::DOC_ITEM_LABEL_FIELD_VALUE;
        run.text = box ? "" : html_text(node);
        run.formatting = formatting;
        run.hyperlink = hyperlink;
        out->push_back(std::move(run));
        return;
    }
    if (tag == "math" || tag == "code") {
        Run run;
        run.label = tag == "math" ? docv1::DOC_ITEM_LABEL_FORMULA : docv1::DOC_ITEM_LABEL_CODE;
        run.text = py_strip(html_text(node));
        run.formatting = formatting;
        run.hyperlink = hyperlink;
        out->push_back(std::move(run));
        return;
    }
    if (tag == "br") {
        Run run;
        run.text = "\n";
        run.formatting = formatting;
        run.hyperlink = hyperlink;
        out->push_back(std::move(run));
        return;
    }
    if (tag == "script" || tag == "style" || tag == "head" || (t_skip_chem && tag == "chem")) {
        return;
    }
    for (const auto& child : node.children) {
        inline_runs(*child, formatting, hyperlink, out);
    }
}

std::optional<double> py_float(std::string_view piece) {
    if (!piece.empty() && piece.front() == '+') {
        piece.remove_prefix(1);
    }
    double value = 0;
    const auto [end, ec] = std::from_chars(piece.data(), piece.data() + piece.size(), value);
    if (ec != std::errc() || end != piece.data() + piece.size()) {
        return std::nullopt;
    }
    return value;
}

class ChandraBuilder {
  public:
    explicit ChandraBuilder(LayoutWriter& writer) : w_(writer) {}

    void walk(const std::vector<const MarkupNode*>& children, Ref parent, Provenance* prov,
              docv1::DocItemLabel label) {
        std::vector<Run> runs;
        for (const MarkupNode* node : children) {
            if (!node->is_text() &&
                ((skip_img_ && node->tag == "img") || (skip_chem_ && node->tag == "chem"))) {
                continue;  // inside a picture: described, not content
            }
            if (node->is_text() || inline_node(*node)) {
                inline_runs(*node, {}, "", &runs);
                continue;
            }
            emit_runs(std::move(runs), parent, prov, label);
            runs.clear();
            block(*node, parent, prov, label);
        }
        emit_runs(std::move(runs), parent, prov, label);
    }

    void walk_children(const MarkupNode& node, Ref parent, Provenance* prov,
                       docv1::DocItemLabel label) {
        walk(child_list(node), parent, prov, label);
    }

    void block(const MarkupNode& node, Ref parent, Provenance* prov, docv1::DocItemLabel label) {
        const bool has_label = node.has_attr("data-label");
        const std::string layout_label = node.attr("data-label");
        Provenance own;
        if (node.has_attr("data-bbox") || has_label) {
            static const std::set<std::string> kNotTextOwners = {
                "Table", "List-Group", "Figure", "Image", "Diagram", "Chemical-Block", "Form"};
            own.box = provenance(node);
            own.text_owner = !(has_label && kNotTextOwners.contains(layout_label));
            prov = &own;
            label = layout_label_of(has_label ? layout_label : "")
                        .value_or(docv1::DOC_ITEM_LABEL_TEXT);
        }
        const std::string& tag = node.tag;
        if ((has_label && (layout_label == "Figure" || layout_label == "Image" ||
                           layout_label == "Diagram" || layout_label == "Chemical-Block")) ||
            tag == "figure" || tag == "img" || tag == "chem") {
            picture(node, parent, prov);
        } else if (tag == "table") {
            table(node, parent, prov);
        } else if (tag == "ul" || tag == "ol" ||
                   (has_label && layout_label == "List-Group" &&
                    !any_descendant(node, [](const MarkupNode& n) {
                        return n.tag == "ul" || n.tag == "ol";
                    }))) {
            list(node, parent);
        } else if (tag == "pre" || tag == "code" ||
                   (has_label && layout_label == "Code-Block" &&
                    !any_descendant(node, [](const MarkupNode& n) { return n.tag == "pre"; }))) {
            const std::string body = strip_chars(html_text(node), "\n\r");
            w_.code(text_prov(prov), body, node.attr("class"), parent);
        } else if (tag == "math") {
            w_.formula(text_prov(prov), py_strip(html_text(node)), parent);
        } else if (!kBlockTags.contains(tag) && (node.has_attr("data-bbox") || has_label)) {
            std::vector<Run> runs;
            inline_runs(node, {}, "", &runs);
            if (!has_label && runs.size() == 1) {
                label = runs[0].label;
            }
            emit_runs(std::move(runs), parent, prov, label);
        } else if ((has_label && layout_label == "Form") || tag == "form") {
            const Ref region = w_.field_region(prov != nullptr ? prov->take() : std::nullopt, parent);
            walk_children(node, region, nullptr, docv1::DOC_ITEM_LABEL_TEXT);
        } else if (tag == "caption" || tag == "figcaption" ||
                   (has_label && (layout_label == "Caption" || layout_label == "Footnote"))) {
            const docv1::DocItemLabel caption_label = has_label && layout_label == "Footnote"
                                                          ? docv1::DOC_ITEM_LABEL_FOOTNOTE
                                                          : docv1::DOC_ITEM_LABEL_CAPTION;
            const int before = w_.text_count();
            walk_children(node, parent, prov, caption_label);
            if (parent.kind == BodyChild::PICTURE || parent.kind == BodyChild::TABLE) {
                for (int i = before; i < w_.text_count(); i++) {
                    const Ref item{BodyChild::TEXT, i};
                    if (w_.label_of(item) != caption_label) {
                        continue;
                    }
                    if (caption_label == docv1::DOC_ITEM_LABEL_FOOTNOTE) {
                        w_.link_footnote(parent, item);
                    } else {
                        w_.link_caption(parent, item);
                    }
                }
            }
        } else if (tag != "script" && tag != "style" && tag != "head" && tag != "hr") {
            if (is_heading_tag(tag) && label == docv1::DOC_ITEM_LABEL_TEXT) {
                label = docv1::DOC_ITEM_LABEL_SECTION_HEADER;
            }
            const int before = w_.text_count();
            walk_children(node, parent, prov, label);
            if (is_heading_tag(tag)) {
                for (int i = before; i < w_.text_count(); i++) {
                    w_.set_heading_level(i, tag[1] - '0');
                }
            }
        }
    }

    bool produced_content() const { return w_.item_count() > 0; }

    void table_block(const MarkupNode& node, Ref parent, Provenance* prov) {
        table(node, parent, prov);
    }

  private:
    static std::vector<const MarkupNode*> child_list(const MarkupNode& node) {
        std::vector<const MarkupNode*> out;
        out.reserve(node.children.size());
        for (const auto& child : node.children) {
            out.push_back(child.get());
        }
        return out;
    }

    static bool inline_node(const MarkupNode& node) {
        return !kBlockTags.contains(node.tag) && node.tag != "img" && node.tag != "chem" &&
               !node.has_attr("data-label") && !node.has_attr("data-bbox") &&
               !(node.tag == "math" && node.attr("display") == "block") &&
               !any_descendant(node, [](const MarkupNode& n) { return is_block(n); });
    }

    Box provenance(const MarkupNode& node) const {
        const std::string raw = node.attr("data-bbox");
        std::vector<double> coords;
        size_t i = 0;
        while (i < raw.size()) {
            while (i < raw.size() && python_space(raw[i])) {
                i++;
            }
            const size_t start = i;
            while (i < raw.size() && !python_space(raw[i])) {
                i++;
            }
            if (i > start) {
                const auto value = py_float(std::string_view(raw).substr(start, i - start));
                if (!value) {
                    return std::nullopt;
                }
                coords.push_back(*value);
            }
        }
        if (coords.size() != 4) {
            return std::nullopt;
        }
        const double x0 = coords[0], y0 = coords[1], x1 = coords[2], y1 = coords[3];
        if (!(0 <= x0 && x0 <= x1 && x1 <= 1000 && 0 <= y0 && y0 <= y1 && y1 <= 1000)) {
            return std::nullopt;
        }
        return w_.grid_box(coords, 1000.0);
    }

    static Box text_prov(Provenance* prov) {
        return prov != nullptr && prov->text_owner ? prov->take() : std::nullopt;
    }

    void emit_runs(std::vector<Run> runs, Ref parent, Provenance* prov,
                   docv1::DocItemLabel label) {
        std::vector<Run> merged;
        for (Run& run : runs) {
            if (!merged.empty() && run.label == docv1::DOC_ITEM_LABEL_TEXT &&
                merged.back().label == docv1::DOC_ITEM_LABEL_TEXT && run.same_style(merged.back())) {
                merged.back().text += run.text;
            } else {
                merged.push_back(std::move(run));
            }
        }
        const bool heading = parent_is_heading(parent) ||
                             label == docv1::DOC_ITEM_LABEL_SECTION_HEADER ||
                             label == docv1::DOC_ITEM_LABEL_TITLE;
        std::vector<Run> kept;
        bool line_break = false;
        for (Run& run : merged) {
            const std::string edges = strip_chars(run.text, " \t");
            run.text = py_strip(run.text);
            const bool is_text = run.label == docv1::DOC_ITEM_LABEL_TEXT;
            line_break = line_break || (is_text && edges.starts_with('\n'));
            if (run.text.empty() && run.label != docv1::DOC_ITEM_LABEL_FIELD_VALUE &&
                run.label != docv1::DOC_ITEM_LABEL_CHECKBOX_SELECTED &&
                run.label != docv1::DOC_ITEM_LABEL_CHECKBOX_UNSELECTED) {
                continue;
            }
            if (line_break && !kept.empty() && !heading) {
                if (run.plain()) {
                    run.text = "\n" + run.text;
                } else if (kept.back().plain()) {
                    kept.back().text += "\n";
                }
            }
            line_break = is_text && edges.ends_with('\n');
            kept.push_back(std::move(run));
        }
        if (heading && kept.size() > 1) {
            for (Run& run : kept) {
                run.text = collapse_whitespace(run.text);
            }
        }
        if (kept.empty()) {
            return;
        }
        const int before = w_.text_count();
        const docv1::ContentLayer layer =
            label == docv1::DOC_ITEM_LABEL_PAGE_HEADER || label == docv1::DOC_ITEM_LABEL_PAGE_FOOTER
                ? docv1::CONTENT_LAYER_FURNITURE
                : docv1::CONTENT_LAYER_BODY;
        if (kept.size() > 1 && label == docv1::DOC_ITEM_LABEL_FORMULA) {
            const bool all_math = std::ranges::all_of(kept, [](const Run& run) {
                return run.label == docv1::DOC_ITEM_LABEL_FORMULA || !has_alnum(run.text);
            });
            if (all_math) {
                std::string joined;
                for (const Run& run : kept) {
                    if (!joined.empty()) {
                        joined.push_back(' ');
                    }
                    joined += py_strip(run.text);
                }
                Run formula;
                formula.text = joined;
                formula.label = docv1::DOC_ITEM_LABEL_FORMULA;
                kept = {formula};
            } else {
                label = docv1::DOC_ITEM_LABEL_TEXT;
            }
        }
        if (kept.size() > 1) {
            if (label != docv1::DOC_ITEM_LABEL_TEXT) {
                parent = w_.styled_text(label, text_prov(prov), "", nullptr, "", parent);
                w_.set_content_layer(parent, layer);
            }
            parent = w_.inline_group(parent);
            w_.set_content_layer(parent, layer);
        }
        for (const Run& run : kept) {
            const docv1::DocItemLabel run_label =
                run.label == docv1::DOC_ITEM_LABEL_TEXT && kept.size() == 1 ? label : run.label;
            const docv1::Formatting* formatting =
                run.plain_formatting() ? nullptr : &run.formatting;
            if (parent.kind == BodyChild::TEXT && !parent.is_body() &&
                w_.children_of(parent) == 0 && w_.text_of(parent).empty() && kept.size() == 1 &&
                run_label == docv1::DOC_ITEM_LABEL_TEXT) {
                w_.set_text(parent, run.text, formatting, run.hyperlink);
                continue;
            }
            const Ref item =
                w_.styled_text(run_label, run_label == label ? text_prov(prov) : std::nullopt,
                               run.text, formatting, run.hyperlink, parent);
            w_.set_content_layer(item, layer);
        }
        for (int i = before; i < w_.text_count(); i++) {
            // Without <hN>, a section header has raw depth 2, like "##".
            w_.set_heading_level(i, 2);
        }
    }

    bool parent_is_heading(Ref parent) const {
        if (parent.kind != BodyChild::TEXT || parent.is_body()) {
            return false;
        }
        const docv1::DocItemLabel label = w_.label_of(parent);
        return label == docv1::DOC_ITEM_LABEL_SECTION_HEADER || label == docv1::DOC_ITEM_LABEL_TITLE;
    }

    void list(const MarkupNode& node, Ref parent) {
        const Ref group = w_.list_group(parent);
        const bool ordered = node.tag == "ol";
        long long number = 1;
        {
            const std::string start = py_strip(node.attr("start", "1"));
            long long value = 0;
            const char* begin = start.data() + (start.starts_with('+') ? 1 : 0);
            const auto [end, ec] = std::from_chars(begin, start.data() + start.size(), value);
            if (ec == std::errc() && end == start.data() + start.size()) {
                number = std::clamp<long long>(value, 1, 2147483647LL);
            }
        }
        // A List-Group block without <ul>/<ol>: items are its <li>/<p>
        // children, with loose content between them gathered into items.
        std::vector<std::unique_ptr<MarkupNode>> synthesized;
        std::vector<const MarkupNode*> children;
        if (node.tag != "ul" && node.tag != "ol") {
            std::vector<const MarkupNode*> pending_refs;
            auto flush = [&] {
                if (!pending_refs.empty()) {
                    synthesized.push_back(make_wrapper(pending_refs));
                    children.push_back(synthesized.back().get());
                    pending_refs.clear();
                }
            };
            for (const auto& child : node.children) {
                if (!child->is_text() &&
                    (child->tag == "li" || child->tag == "p" || child->tag == "br")) {
                    flush();
                    if (child->tag != "br") {
                        children.push_back(child.get());
                    }
                } else {
                    pending_refs.push_back(child.get());
                }
            }
            // The trailing loose content only counts with text or elements.
            bool keep = false;
            for (const MarkupNode* ref : pending_refs) {
                keep = keep || !ref->is_text() || !py_strip(ref->text).empty();
            }
            if (keep) {
                flush();
            }
        } else {
            children = child_list(node);
        }
        for (const MarkupNode* child : children) {
            if (child->is_text() && py_strip(child->text).empty()) {
                continue;
            }
            if (!child->is_text() && (child->tag == "ul" || child->tag == "ol")) {
                list(*child, group);
                continue;
            }
            const Ref item = w_.list_item(std::nullopt, "", ordered,
                                          ordered ? std::to_string(number) + "." : "", group);
            if (!child->is_text() && (child->tag == "li" || child->tag == "p" ||
                                      child->tag == kWrapperTag)) {
                walk(wrapper_children(*child), item, nullptr, docv1::DOC_ITEM_LABEL_TEXT);
            } else {
                walk({child}, item, nullptr, docv1::DOC_ITEM_LABEL_TEXT);
            }
            number++;
        }
    }

    // Loose list content is wrapped without copying: the wrapper's
    // children are borrowed pointers kept beside it.
    static constexpr const char* kWrapperTag = "#li";
    std::unique_ptr<MarkupNode> make_wrapper(const std::vector<const MarkupNode*>& refs) {
        auto wrapper = std::make_unique<MarkupNode>();
        wrapper->tag = kWrapperTag;
        borrowed_[wrapper.get()] = refs;
        return wrapper;
    }
    std::vector<const MarkupNode*> wrapper_children(const MarkupNode& node) {
        const auto it = borrowed_.find(&node);
        if (it != borrowed_.end()) {
            return it->second;
        }
        return child_list(node);
    }

    void table(const MarkupNode& node, Ref parent, Provenance* prov) {
        std::vector<SpanCell> cells = chandra_table_cells(node);
        if (cells.empty()) {
            walk_children(node, parent, prov, docv1::DOC_ITEM_LABEL_TEXT);
            return;
        }
        const Ref table = w_.table(prov != nullptr ? prov->take() : std::nullopt, {}, parent);
        std::vector<std::string> rich_refs(cells.size());
        for (size_t i = 0; i < cells.size(); i++) {
            const MarkupNode& cell_node = *cells[i].node;
            if (!any_descendant(cell_node, [](const MarkupNode& n) { return n.tag != "span"; })) {
                continue;
            }
            const Ref group = w_.group("group", docv1::GROUP_LABEL_UNSPECIFIED, table);
            walk_children(cell_node, group, nullptr, docv1::DOC_ITEM_LABEL_TEXT);
            if (w_.children_of(group) > 0) {
                rich_refs[i] = body_child_ref(BodyChild::GROUP, group.index);
            }
        }
        docv1::TableItem* item = w_.table_item(table);
        const TableCut cut = fill_span_table(item->mutable_data(), cells);
        note_table_cut(cut, item->data(), "table", body_child_ref(BodyChild::TABLE, table.index),
                       w_.warnings());
        attach_rich_refs(item->mutable_data(), cells, rich_refs);
        for (const auto& child : node.children) {
            if (!child->is_text() &&
                (child->tag == "caption" || child->tag == "figcaption" ||
                 child->attr("data-label") == "Caption" || child->attr("data-label") == "Footnote")) {
                block(*child, table, nullptr, docv1::DOC_ITEM_LABEL_CAPTION);
            }
        }
    }

    static void attach_rich_refs(docv1::TableData* data, const std::vector<SpanCell>& cells,
                                 const std::vector<std::string>& refs) {
        // table_cells holds the kept anchors in source order; match them
        // back by their start position.
        std::map<std::pair<int32_t, int32_t>, std::string> by_anchor;
        for (size_t i = 0; i < cells.size(); i++) {
            if (!refs[i].empty()) {
                by_anchor[{static_cast<int32_t>(cells[i].row), static_cast<int32_t>(cells[i].col)}] =
                    refs[i];
            }
        }
        if (by_anchor.empty()) {
            return;
        }
        auto stamp = [&](docv1::TableCell* cell) {
            const auto it =
                by_anchor.find({cell->start_row_offset_idx(), cell->start_col_offset_idx()});
            if (it != by_anchor.end()) {
                cell->mutable_ref()->set_ref(it->second);
            }
        };
        for (docv1::TableCell& cell : *data->mutable_table_cells()) {
            stamp(&cell);
        }
        for (docv1::TableRow& row : *data->mutable_grid()) {
            for (docv1::TableCell& cell : *row.mutable_cells()) {
                stamp(&cell);
            }
        }
    }

    void picture(const MarkupNode& node, Ref parent, Provenance* prov) {
        const Ref picture = w_.picture(prov != nullptr ? prov->take() : std::nullopt, parent);
        std::vector<const MarkupNode*> images;
        std::vector<const MarkupNode*> chemicals;
        if (node.tag == "img") {
            images.push_back(&node);
        } else {
            for_each_descendant(node, [&](const MarkupNode& n) {
                if (n.tag == "img") {
                    images.push_back(&n);
                }
            });
        }
        if (node.tag == "chem") {
            chemicals.push_back(&node);
        } else {
            for_each_descendant(node, [&](const MarkupNode& n) {
                if (n.tag == "chem") {
                    chemicals.push_back(&n);
                }
            });
        }
        std::vector<std::string> descriptions;
        for (const MarkupNode* image : images) {
            const std::string alt = image->attr("alt");
            if (!alt.empty() && std::ranges::find(descriptions, alt) == descriptions.end()) {
                descriptions.push_back(alt);
            }
        }
        std::string description;
        for (const std::string& alt : descriptions) {
            if (!description.empty()) {
                description.push_back('\n');
            }
            description += alt;
        }
        w_.describe_picture(picture, description, "chandra");
        if (chemicals.size() == 1) {
            w_.set_molecule(picture, py_strip(html_text(*chemicals[0])), "chandra");
        }
        if (node.tag == "chem") {
            return;
        }
        // The box covers the whole picture; its <img> tags describe it
        // rather than locate images, and a lone <chem> is the molecule.
        // Structured children are kept as picture content; bare text
        // beside an <img> restates the alt text and is dropped.
        const bool drop_chem = chemicals.size() == 1;
        std::vector<const MarkupNode*> kept;
        for (const auto& child : node.children) {
            if (child->is_text()) {
                if (images.empty()) {
                    kept.push_back(child.get());
                }
                continue;
            }
            if (child->tag == "img" || (child->tag == "chem" && drop_chem)) {
                continue;
            }
            kept.push_back(child.get());
        }
        const bool saved_img = skip_img_;
        const bool saved_chem = skip_chem_;
        skip_img_ = true;
        skip_chem_ = drop_chem;
        t_skip_chem = drop_chem;
        walk(kept, picture, nullptr, docv1::DOC_ITEM_LABEL_TEXT);
        skip_img_ = saved_img;
        skip_chem_ = saved_chem;
        t_skip_chem = saved_chem;
    }

    LayoutWriter& w_;
    std::map<const MarkupNode*, std::vector<const MarkupNode*>> borrowed_;
    bool skip_img_ = false;
    bool skip_chem_ = false;
};

}  // namespace

bool add_html_fragment(LayoutWriter& writer, const std::string& html, docv1::DocItemLabel label,
                       const std::optional<docv1::BoundingBox>& box, LayoutRef parent) {
    const std::unique_ptr<MarkupNode> root = parse_markup(html, MarkupMode::kHtml);
    if (!any_descendant(*root, [](const MarkupNode&) { return true; })) {
        return false;
    }
    ChandraBuilder builder(writer);
    Provenance prov{box, true, false};
    builder.walk_children(*root, parent, box ? &prov : nullptr, label);
    return true;
}

std::optional<LayoutRef> add_html_table(LayoutWriter& writer, const std::string& html,
                                        const std::optional<docv1::BoundingBox>& box,
                                        LayoutRef parent) {
    const std::unique_ptr<MarkupNode> root = parse_markup(html, MarkupMode::kHtml);
    MarkupNode* table = const_cast<MarkupNode*>(find_element(*root, "table"));
    if (table == nullptr || chandra_table_cells(*table).empty()) {
        return std::nullopt;
    }
    for (auto& child : table->children) {
        if (!child->is_text() && (child->tag == "caption" || child->tag == "figcaption")) {
            auto text = std::make_unique<MarkupNode>();
            text->text = html_text(*child);
            child->children.clear();
            child->children.push_back(std::move(text));
        }
    }
    ChandraBuilder builder(writer);
    Provenance prov{box, false, false};
    const size_t before = writer.item_count();
    builder.table_block(*table, parent, &prov);
    if (writer.item_count() == before) {
        return std::nullopt;
    }
    return writer.last_table();
}

bool add_inline_text(LayoutWriter& writer, const std::string& text, docv1::DocItemLabel label,
                     const std::optional<docv1::BoundingBox>& box,
                     std::optional<LayoutRef> parent) {
    const std::string html = markdown_inline_html(text);
    if (!has_inline_html_tag(html)) {
        return false;
    }
    std::optional<docv1::BoundingBox> fragment_box = box;
    if (!parent) {
        parent = writer.styled_text(label, box, "", nullptr, "");
        writer.set_orig(*parent, text);
        label = docv1::DOC_ITEM_LABEL_TEXT;
        fragment_box.reset();
    }
    return add_html_fragment(writer, html, label, fragment_box, *parent);
}

bool map_chandra_html(const std::string& text, const PageContext& page, docv1::Document* out,
                      std::string* error, std::vector<vlmv1::PageWarning>* warnings) {
    LayoutWriter writer(page, out, warnings);
    if (py_strip(text).empty()) {
        return writer.finish(error, "Chandra", true);
    }
    const std::unique_ptr<MarkupNode> root = parse_markup(text, MarkupMode::kHtml);
    ChandraBuilder builder(writer);
    std::vector<const MarkupNode*> blocks;
    bool blank_page = false;
    // Outer layout blocks only, in document order: a block nested in
    // another is walked as part of it.
    std::function<void(const MarkupNode&)> find_blocks = [&](const MarkupNode& node) {
        for (const auto& child : node.children) {
            if (child->is_text()) {
                continue;
            }
            if (child->has_attr("data-label") || child->has_attr("data-bbox")) {
                blocks.push_back(child.get());
                blank_page = blank_page || child->attr("data-label") == "Blank-Page";
                for_each_descendant(*child, [&](const MarkupNode& nested) {
                    blank_page = blank_page || nested.attr("data-label") == "Blank-Page";
                });
                continue;
            }
            find_blocks(*child);
        }
    };
    find_blocks(*root);
    if (!blocks.empty()) {
        for (const MarkupNode* block : blocks) {
            builder.block(*block, LayoutWriter::kBody, nullptr, docv1::DOC_ITEM_LABEL_TEXT);
        }
    } else {
        std::vector<const MarkupNode*> html_nodes;
        for (const auto& child : root->children) {
            if (!child->is_text() && child->tag != "script" && child->tag != "style" &&
                child->tag != "head") {
                html_nodes.push_back(child.get());
            }
        }
        if (html_nodes.empty()) {
            *error = "Chandra response contains no HTML transcription or layout blocks";
            return false;
        }
        // No layout blocks: the HTML is recovered without coordinates.
        builder.walk(html_nodes, LayoutWriter::kBody, nullptr, docv1::DOC_ITEM_LABEL_TEXT);
    }
    if (!builder.produced_content() && !blank_page) {
        *error = "Chandra HTML contains no document content";
        return false;
    }
    return writer.finish(error, "Chandra", true);
}

}  // namespace vlm::mapping
