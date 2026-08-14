#include "builder.h"
#include "mapper.h"
#include "otsl_grid.h"

#include <vector>

namespace vlm::mapping {

namespace {

// DocTags locations are integers on a 0..500 grid over the page raster.
constexpr double kLocGrid = 500.0;

struct Element {
    std::string name;
    std::vector<long> locs;  // up to 4: x1, y1, x2, y2 on the grid
    std::string text;
    std::string otsl;  // raw OTSL body inside a table element
};

docv1::BoundingBox locs_box(const Element& element, const PageContext& page) {
    if (element.locs.size() < 4 || page.width == 0 || page.height == 0) {
        return full_page_box(page);
    }
    docv1::BoundingBox box;
    box.set_l(element.locs[0] / kLocGrid * page.width);
    box.set_t(element.locs[1] / kLocGrid * page.height);
    box.set_r(element.locs[2] / kLocGrid * page.width);
    box.set_b(element.locs[3] / kLocGrid * page.height);
    box.set_coord_origin(docv1::COORD_ORIGIN_TOPLEFT);
    return box;
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

// Emits one parsed DocTags element as the matching Document item.
// Returns false only for elements that carried nothing at all.
bool emit_element(const Element& element, const PageContext& page, docv1::Document* doc) {
    const std::string text = trim(element.text);
    const docv1::ProvenanceItem prov = make_prov(page, locs_box(element, page));
    const std::string& tag = element.name;

    if (tag == "title") {
        add_title(doc, page, prov, text);
    } else if (starts_with(tag, "section_header_level_")) {
        int level = 1;
        try {
            level = std::max(1, std::stoi(tag.substr(21)));
        } catch (...) {
        }
        add_section_header(doc, page, level, prov, text);
    } else if (starts_with(tag, "subtitle-level_")) {
        add_section_header(doc, page, 1, prov, text);
    } else if (tag == "list_item") {
        add_list_item(doc, page, prov, text, /*enumerated=*/false, "");
    } else if (tag == "code") {
        add_code(doc, page, prov, text, "");
    } else if (tag == "formula") {
        add_formula(doc, page, prov, text);
    } else if (tag == "picture") {
        add_picture(doc, page, prov);
        return true;
    } else if (tag == "table" || tag == "otsl") {
        docv1::TableItem* table = add_table(doc, page, prov);
        const std::string& body = tag == "otsl" && element.otsl.empty() ? text : element.otsl;
        if (!body.empty()) {
            parse_otsl_grid(body, table->mutable_data());  // spans kept verbatim in v1
        }
        return true;
    } else if (tag == "page_header" || tag == "page_footer") {
        docv1::BaseTextItem* item =
            add_text(doc, page,
                     tag == "page_header" ? docv1::DOC_ITEM_LABEL_PAGE_HEADER
                                          : docv1::DOC_ITEM_LABEL_PAGE_FOOTER,
                     prov, text);
        item->mutable_text()->mutable_base()->set_content_layer(docv1::CONTENT_LAYER_FURNITURE);
    } else if (tag == "footnote") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_FOOTNOTE, prov, text);
    } else if (tag == "caption") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_CAPTION, prov, text);
    } else if (tag == "checkbox_selected") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_CHECKBOX_SELECTED, prov, text);
    } else if (tag == "checkbox_unselected") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_CHECKBOX_UNSELECTED, prov, text);
    } else if (tag == "document_index") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_DOCUMENT_INDEX, prov, text);
    } else if (tag == "text" || tag == "paragraph") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_TEXT, prov, text);
    } else {
        // Open-vocabulary tags (Docling adds labels over time): keep the
        // text as a plain item rather than dropping the page's content.
        if (text.empty() && element.otsl.empty()) {
            return false;
        }
        add_text(doc, page, docv1::DOC_ITEM_LABEL_TEXT, prov, text);
    }
    return true;
}

}  // namespace

bool map_doctags(const std::string& text, const PageContext& page, docv1::Document* out,
                 std::string* error) {
    if (text.find('<') == std::string::npos) {
        *error = "response holds no DocTags markup";
        return false;
    }

    size_t items = 0;
    Element current;
    bool element_open = false;
    bool in_otsl = false;
    std::vector<long> pending_locs;
    size_t pos = 0;

    auto flush = [&] {
        if (element_open && emit_element(current, page, out)) {
            items++;
        }
        current = Element{};
        element_open = false;
        pending_locs.clear();
    };

    while (pos < text.size()) {
        if (in_otsl) {
            // OTSL cell markup is the table's payload, not DocTags
            // structure: consume it raw up to the closing tag.
            size_t end = text.find("</otsl>", pos);
            current.otsl += text.substr(pos, end == std::string::npos ? end : end - pos);
            pos = end == std::string::npos ? text.size() : end + 7;
            in_otsl = false;
            continue;
        }
        size_t open = text.find('<', pos);
        if (open == std::string::npos) {
            if (element_open) {
                current.text += text.substr(pos);
            }
            break;
        }
        size_t close = text.find('>', open);
        if (close == std::string::npos) {
            break;  // truncated tag: keep what parsed
        }
        if (element_open && open > pos) {
            current.text += text.substr(pos, open - pos);
        }
        std::string token = text.substr(open + 1, close - open - 1);
        pos = close + 1;

        if (starts_with(token, "loc_")) {
            try {
                (element_open ? current.locs : pending_locs)
                    .push_back(std::stol(token.substr(4)));
            } catch (...) {
            }
            continue;
        }
        if (token == "doctag" || token == "/doctag") {
            continue;  // root wrapper
        }
        if (token == "otsl") {
            in_otsl = true;
            if (!element_open) {
                current.name = "otsl";
                current.locs = pending_locs;
                pending_locs.clear();
                element_open = true;
            }
            continue;
        }
        if (!token.empty() && token[0] == '/') {
            if (element_open && token == "/" + current.name) {
                flush();
            }
            continue;  // stray close: ignore
        }
        // A new element auto-closes any unclosed predecessor — SmolDocling
        // output is not always well nested.
        flush();
        current.name = token;
        current.locs = pending_locs;
        pending_locs.clear();
        element_open = true;
    }
    flush();

    if (items == 0) {
        *error = "DocTags response produced no document items";
        return false;
    }
    finalize_document(out, page);
    return true;
}

}  // namespace vlm::mapping
