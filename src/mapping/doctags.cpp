#include <algorithm>
#include <cctype>
#include <utility>
#include <vector>

#include "builder.h"
#include "image_crop.h"
#include "mapper.h"
#include "otsl_grid.h"

namespace vlm::mapping {

namespace {

// DocTags locations are integers on a 0..500 grid over the page raster.
constexpr double kLocGrid = 500.0;

// Attribution docling stamps on doctags-derived picture predictions.
constexpr const char* kCreatedBy = "load_from_doctags";

struct Element {
    std::string name;
    std::vector<long> locs;  // up to 4: x1, y1, x2, y2 on the grid
    std::string text;
    std::string otsl;  // raw OTSL body inside a table/chart element
    // Container extras (picture/chart/table): classification candidate
    // tags in encounter order, and an embedded <caption>.
    std::vector<std::string> class_tags;
    bool has_caption = false;
    std::vector<long> caption_locs;
    std::string caption_text;
};

docv1::BoundingBox locs_box(const std::vector<long>& locs, const PageContext& page) {
    if (locs.size() < 4 || page.width == 0 || page.height == 0) {
        return full_page_box(page);
    }
    docv1::BoundingBox box;
    box.set_l(locs[0] / kLocGrid * page.width);
    box.set_t(locs[1] / kLocGrid * page.height);
    box.set_r(locs[2] / kLocGrid * page.width);
    box.set_b(locs[3] / kLocGrid * page.height);
    box.set_coord_origin(docv1::COORD_ORIGIN_TOPLEFT);
    return box;
}

// Docling stamps charspan on every doctags-derived provenance: [0, len)
// for text items (len in Unicode characters), [0, 0] for floating items.
docv1::ProvenanceItem prov_with_charspan(const PageContext& page,
                                         const docv1::BoundingBox& box, int32_t start,
                                         int32_t end) {
    docv1::ProvenanceItem prov = make_prov(page, box);
    prov.mutable_charspan()->set_start(start);
    prov.mutable_charspan()->set_end(end);
    return prov;
}

// UTF-8 character count (docling's len() counts code points, not bytes).
int32_t char_length(const std::string& text) {
    int32_t count = 0;
    for (unsigned char c : text) {
        if ((c & 0xC0) != 0x80) {
            count++;
        }
    }
    return count;
}

bool is_container(const std::string& tag) {
    return tag == "picture" || tag == "chart" || tag == "table" || tag == "otsl";
}

// Group/graph container chunks are captured raw (like OTSL) so their
// nested list items / key-value cells are not parsed as top-level elements.
bool is_group_container(const std::string& tag) {
    return tag == "ordered_list" || tag == "unordered_list" || tag == "inline" ||
           tag == "key_value_region";
}

// The first up-to-4 <loc_*> values in the chunk, in encounter order —
// docling's extract_bounding_box truncates to the first four and only
// produces a box when there are exactly four.
std::vector<long> first_locs(const std::string& chunk) {
    std::vector<long> locs;
    size_t pos = 0;
    while (locs.size() < 4) {
        const size_t open = chunk.find("<loc_", pos);
        if (open == std::string::npos) {
            break;
        }
        const size_t gt = chunk.find('>', open);
        if (gt == std::string::npos) {
            break;
        }
        try {
            locs.push_back(std::stol(chunk.substr(open + 5, gt - open - 5)));
        } catch (...) {
        }
        pos = gt + 1;
    }
    return locs;
}

// Docling's extract_inner_text: strip every <...> tag except the <_..._>
// sequences (code language tokens), then strip whitespace. A "<" not
// followed by a letter or "/" is plain text, not a tag.
std::string inner_text(const std::string& chunk) {
    std::string out;
    size_t pos = 0;
    while (pos < chunk.size()) {
        const size_t open = chunk.find('<', pos);
        if (open == std::string::npos) {
            out += chunk.substr(pos);
            break;
        }
        const size_t gt = chunk.find('>', open);
        if (gt == std::string::npos) {
            out += chunk.substr(pos);
            break;
        }
        const std::string token = chunk.substr(open + 1, gt - open - 1);
        const bool keep = token.empty() ||
                          (token[0] == '_' && token.back() == '_') ||
                          (!std::isalpha(static_cast<unsigned char>(token[0])) &&
                           token[0] != '/');
        out += chunk.substr(pos, open - pos);
        if (keep) {
            out += chunk.substr(open, gt - open + 1);
        }
        pos = gt + 1;
    }
    return trim(out);
}

// Docling's key/value cell cleanup: only <loc_*> and <link_*> tokens come
// out of the raw cell content before stripping.
std::string clean_cell_text(const std::string& content) {
    std::string out;
    size_t pos = 0;
    while (pos < content.size()) {
        const size_t open = content.find('<', pos);
        if (open == std::string::npos) {
            out += content.substr(pos);
            break;
        }
        const size_t gt = content.find('>', open);
        if (gt == std::string::npos) {
            out += content.substr(pos);
            break;
        }
        const std::string token = content.substr(open + 1, gt - open - 1);
        if (token.starts_with("loc_") || token.starts_with("link_")) {
            out += content.substr(pos, open - pos);
        } else {
            out += content.substr(pos, gt - pos + 1);
        }
        pos = gt + 1;
    }
    return trim(out);
}

// <link_N> targets in encounter order (docling's link-token regex).
std::vector<int> link_targets(const std::string& content) {
    std::vector<int> targets;
    size_t pos = 0;
    while (pos < content.size()) {
        const size_t open = content.find("<link_", pos);
        if (open == std::string::npos) {
            break;
        }
        const size_t gt = content.find('>', open);
        if (gt == std::string::npos) {
            break;
        }
        try {
            targets.push_back(std::stoi(content.substr(open + 6, gt - open - 6)));
        } catch (...) {
        }
        pos = gt + 1;
    }
    return targets;
}

// Picture classification labels, in docling's precedence order (first
// label whose tag appears in the chunk wins): v2 model labels, then the
// legacy v1 labels, then the legacy SmolDocling aliases. The second of
// each pair is the reported class name after legacy mapping.
const std::vector<std::pair<const char*, const char*>> kClassificationLabels = {
    {"logo", "logo"},
    {"photograph", "photograph"},
    {"icon", "icon"},
    {"engineering_drawing", "engineering_drawing"},
    {"line_chart", "line_chart"},
    {"bar_chart", "bar_chart"},
    {"other", "other"},
    {"table", "table"},
    {"flow_chart", "flow_chart"},
    {"screenshot_from_computer", "screenshot_from_computer"},
    {"signature", "signature"},
    {"screenshot_from_manual", "screenshot_from_manual"},
    {"geographical_map", "geographical_map"},
    {"pie_chart", "pie_chart"},
    {"page_thumbnail", "page_thumbnail"},
    {"stamp", "stamp"},
    {"music", "music"},
    {"calendar", "calendar"},
    {"qr_code", "qr_code"},
    {"bar_code", "bar_code"},
    {"full_page_image", "full_page_image"},
    {"scatter_plot", "scatter_plot"},
    {"chemistry_structure", "chemistry_structure"},
    {"topographical_map", "topographical_map"},
    {"crossword_puzzle", "crossword_puzzle"},
    {"box_plot", "box_plot"},
    // Legacy v1 model labels.
    {"stacked_bar_chart", "stacked_bar_chart"},
    {"scatter_chart", "scatter_chart"},
    {"heatmap", "heatmap"},
    {"natural_image", "natural_image"},
    {"remote_sensing", "remote_sensing"},
    {"screenshot", "screenshot"},
    {"chemistry_molecular_structure", "chemistry_molecular_structure"},
    {"chemistry_markush_structure", "chemistry_markush_structure"},
    {"picture_group", "picture_group"},
    // Legacy SmolDocling aliases.
    {"line", "line_chart"},
    {"dot_line", "line_chart"},
    {"vbar_categorical", "bar_chart"},
    {"hbar_categorical", "bar_chart"},
};

bool is_classification_tag(const std::string& token) {
    return std::ranges::any_of(kClassificationLabels,
                               [&](const auto& label) { return token == label.first; });
}

// Docling scans the chunk for each label in list order and keeps the
// first hit; encounter order in the token stream does not matter.
std::string pick_classification(const std::vector<std::string>& seen) {
    for (const auto& [tag, mapped] : kClassificationLabels) {
        if (std::ranges::contains(seen, tag)) {
            return mapped;
        }
    }
    return "";
}

// Docling matches the <_language_> token against CodeLanguageLabel values
// exactly (case-sensitive); anything else falls back to UNKNOWN.
docv1::CodeLanguageLabel code_language_of(const std::string& raw) {
    static const std::vector<std::pair<const char*, docv1::CodeLanguageLabel>> kLanguages = {
        {"Ada", docv1::CODE_LANGUAGE_LABEL_ADA},
        {"Awk", docv1::CODE_LANGUAGE_LABEL_AWK},
        {"Bash", docv1::CODE_LANGUAGE_LABEL_BASH},
        {"bc", docv1::CODE_LANGUAGE_LABEL_BC},
        {"C", docv1::CODE_LANGUAGE_LABEL_C},
        {"C#", docv1::CODE_LANGUAGE_LABEL_C_SHARP},
        {"C++", docv1::CODE_LANGUAGE_LABEL_C_PLUS_PLUS},
        {"CMake", docv1::CODE_LANGUAGE_LABEL_CMAKE},
        {"COBOL", docv1::CODE_LANGUAGE_LABEL_COBOL},
        {"CSS", docv1::CODE_LANGUAGE_LABEL_CSS},
        {"Ceylon", docv1::CODE_LANGUAGE_LABEL_CEYLON},
        {"Clojure", docv1::CODE_LANGUAGE_LABEL_CLOJURE},
        {"Crystal", docv1::CODE_LANGUAGE_LABEL_CRYSTAL},
        {"Cuda", docv1::CODE_LANGUAGE_LABEL_CUDA},
        {"Cython", docv1::CODE_LANGUAGE_LABEL_CYTHON},
        {"D", docv1::CODE_LANGUAGE_LABEL_D},
        {"Dart", docv1::CODE_LANGUAGE_LABEL_DART},
        {"dc", docv1::CODE_LANGUAGE_LABEL_DC},
        {"Dockerfile", docv1::CODE_LANGUAGE_LABEL_DOCKERFILE},
        {"DocLang", docv1::CODE_LANGUAGE_LABEL_DOCLANG},
        {"Elixir", docv1::CODE_LANGUAGE_LABEL_ELIXIR},
        {"Erlang", docv1::CODE_LANGUAGE_LABEL_ERLANG},
        {"FORTRAN", docv1::CODE_LANGUAGE_LABEL_FORTRAN},
        {"Forth", docv1::CODE_LANGUAGE_LABEL_FORTH},
        {"Go", docv1::CODE_LANGUAGE_LABEL_GO},
        {"HTML", docv1::CODE_LANGUAGE_LABEL_HTML},
        {"Haskell", docv1::CODE_LANGUAGE_LABEL_HASKELL},
        {"Haxe", docv1::CODE_LANGUAGE_LABEL_HAXE},
        {"Java", docv1::CODE_LANGUAGE_LABEL_JAVA},
        {"JavaScript", docv1::CODE_LANGUAGE_LABEL_JAVASCRIPT},
        {"JSON", docv1::CODE_LANGUAGE_LABEL_JSON},
        {"Julia", docv1::CODE_LANGUAGE_LABEL_JULIA},
        {"Kotlin", docv1::CODE_LANGUAGE_LABEL_KOTLIN},
        {"Latex", docv1::CODE_LANGUAGE_LABEL_LATEX},
        {"Lisp", docv1::CODE_LANGUAGE_LABEL_LISP},
        {"Lua", docv1::CODE_LANGUAGE_LABEL_LUA},
        {"Matlab", docv1::CODE_LANGUAGE_LABEL_MATLAB},
        {"MoonScript", docv1::CODE_LANGUAGE_LABEL_MOONSCRIPT},
        {"Nim", docv1::CODE_LANGUAGE_LABEL_NIM},
        {"OCaml", docv1::CODE_LANGUAGE_LABEL_OCAML},
        {"ObjectiveC", docv1::CODE_LANGUAGE_LABEL_OBJECTIVEC},
        {"Octave", docv1::CODE_LANGUAGE_LABEL_OCTAVE},
        {"PHP", docv1::CODE_LANGUAGE_LABEL_PHP},
        {"Pascal", docv1::CODE_LANGUAGE_LABEL_PASCAL},
        {"Perl", docv1::CODE_LANGUAGE_LABEL_PERL},
        {"Prolog", docv1::CODE_LANGUAGE_LABEL_PROLOG},
        {"Python", docv1::CODE_LANGUAGE_LABEL_PYTHON},
        {"Racket", docv1::CODE_LANGUAGE_LABEL_RACKET},
        {"Ruby", docv1::CODE_LANGUAGE_LABEL_RUBY},
        {"Rust", docv1::CODE_LANGUAGE_LABEL_RUST},
        {"SML", docv1::CODE_LANGUAGE_LABEL_SML},
        {"SQL", docv1::CODE_LANGUAGE_LABEL_SQL},
        {"Scala", docv1::CODE_LANGUAGE_LABEL_SCALA},
        {"Scheme", docv1::CODE_LANGUAGE_LABEL_SCHEME},
        {"Swift", docv1::CODE_LANGUAGE_LABEL_SWIFT},
        {"Tikz", docv1::CODE_LANGUAGE_LABEL_TIKZ},
        {"TypeScript", docv1::CODE_LANGUAGE_LABEL_TYPESCRIPT},
        {"VisualBasic", docv1::CODE_LANGUAGE_LABEL_VISUALBASIC},
        {"XML", docv1::CODE_LANGUAGE_LABEL_XML},
        {"YAML", docv1::CODE_LANGUAGE_LABEL_YAML},
    };
    const auto match = std::ranges::find_if(
        kLanguages, [&](const auto& language) { return raw == language.first; });
    return match != kLanguages.end() ? match->second : docv1::CODE_LANGUAGE_LABEL_UNKNOWN;
}

// Adds the element's embedded caption as a CAPTION text item and returns
// its index in doc.texts (-1 when there is none). Docling adds the caption
// before the floating item it belongs to, so callers invoke this first.
int emit_caption(const Element& element, const PageContext& page, docv1::Document* doc,
                 std::vector<BodyChild>* order) {
    if (!element.has_caption) {
        return -1;
    }
    const std::string text = trim(element.caption_text);
    const docv1::ProvenanceItem prov =
        prov_with_charspan(page, locs_box(element.caption_locs, page), 0, char_length(text));
    add_text(doc, page, docv1::DOC_ITEM_LABEL_CAPTION, prov, text);
    const int index = doc->texts_size() - 1;
    order->push_back({BodyChild::TEXT, index});
    return index;
}

// <picture> / <chart>: PictureItem with the region crop, the classification
// prediction (confidence 1.0, created_by load_from_doctags), and — for
// charts — the embedded OTSL as tabular chart data. Both land in meta and
// in the annotations union, as docling-core does.
bool emit_picture(const Element& element, const PageContext& page, docv1::Document* doc,
                  std::vector<BodyChild>* order) {
    const docv1::BoundingBox box = locs_box(element.locs, page);
    const int caption_index = emit_caption(element, page, doc, order);
    docv1::PictureItem* picture = add_picture(doc, page, prov_with_charspan(page, box, 0, 0));
    const int index = doc->pictures_size() - 1;
    order->push_back({BodyChild::PICTURE, index});
    if (caption_index >= 0) {
        picture->add_captions()->set_ref(body_child_ref(BodyChild::TEXT, caption_index));
    }

    const std::string classification = pick_classification(element.class_tags);
    if (!classification.empty()) {
        auto* prediction = picture->mutable_meta()
                               ->mutable_classification()
                               ->add_predictions();
        prediction->set_class_name(classification);
        prediction->set_confidence(1.0);
        prediction->set_created_by(kCreatedBy);
        auto* data = picture->add_annotations()->mutable_classification();
        data->set_kind("classification");
        data->set_provenance(kCreatedBy);
        auto* predicted = data->add_predicted_classes();
        predicted->set_class_name(classification);
        predicted->set_confidence(1.0);
    }

    if (element.name == "chart" && !element.otsl.empty()) {
        docv1::TableData chart_data;
        if (parse_otsl_grid(element.otsl, &chart_data)) {
            const std::string title = classification.empty() ? "other" : classification;
            auto* meta_chart = picture->mutable_meta()->mutable_tabular_chart();
            meta_chart->set_title(title);
            *meta_chart->mutable_chart_data() = chart_data;
            auto* annotation = picture->add_annotations()->mutable_tabular_chart();
            annotation->set_kind("tabular_chart_data");
            annotation->set_title(title);
            *annotation->mutable_chart_data() = chart_data;
        }
    }

    // The region crop: best-effort. A page that carries no raster (or one
    // stb cannot decode) still gets the PictureItem, just without image.
    if (!page.png.empty() && element.locs.size() >= 4) {
        docv1::ImageRef crop;
        if (crop_png_image(page.png, box.l(), box.t(), box.r(), box.b(), page.width,
                           page.height, &crop)) {
            *picture->mutable_image() = std::move(crop);
        }
    }
    return true;
}

// Emits one parsed DocTags element as the matching Document item,
// appending to `order` in source (token) order. Returns false only for
// elements that carried nothing at all. `forced_box` overrides the
// element's own locations — docling's <inline> groups give every child
// the chunk's shared box.
bool emit_element(const Element& element, const PageContext& page, docv1::Document* doc,
                  std::vector<BodyChild>* order,
                  const docv1::BoundingBox* forced_box = nullptr) {
    const std::string text = trim(element.text);
    const docv1::ProvenanceItem prov =
        prov_with_charspan(page,
                           forced_box != nullptr ? *forced_box : locs_box(element.locs, page),
                           0, char_length(text));
    const std::string& tag = element.name;

    int index = -1;
    if (tag == "title") {
        add_title(doc, page, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag.starts_with("section_header_level_")) {
        int level = 1;
        try {
            level = std::max(1, std::stoi(tag.substr(21)));
        } catch (...) {
        }
        add_section_header(doc, page, level, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag.starts_with("subtitle-level_")) {
        add_section_header(doc, page, 1, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag == "list_item") {
        add_list_item(doc, page, prov, text, /*enumerated=*/false, "");
        index = doc->texts_size() - 1;
    } else if (tag == "code") {
        // A leading <_language_> token selects the code language; the
        // charspan above already covers the unstripped text, as docling's
        // does (it measures before stripping the token).
        std::string body = text;
        docv1::CodeLanguageLabel language = docv1::CODE_LANGUAGE_LABEL_UNKNOWN;
        std::string language_raw;
        if (body.starts_with("<_")) {
            if (const size_t end = body.find("_>", 2); end != std::string::npos) {
                language_raw = body.substr(2, end - 2);
                language = code_language_of(language_raw);
                body = body.substr(end + 2);
            }
        }
        docv1::BaseTextItem* item = add_code(doc, page, prov, body, language_raw);
        item->mutable_code()->set_code_language(language);
        index = doc->texts_size() - 1;
    } else if (tag == "formula") {
        add_formula(doc, page, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag == "picture" || tag == "chart") {
        return emit_picture(element, page, doc, order);
    } else if (tag == "table" || tag == "otsl") {
        const int caption_index = emit_caption(element, page, doc, order);
        const docv1::ProvenanceItem table_prov =
            prov_with_charspan(page, locs_box(element.locs, page), 0, 0);
        docv1::TableItem* table = add_table(doc, page, table_prov);
        order->push_back({BodyChild::TABLE, doc->tables_size() - 1});
        if (caption_index >= 0) {
            table->add_captions()->set_ref(body_child_ref(BodyChild::TEXT, caption_index));
        }
        const std::string& body = tag == "otsl" && element.otsl.empty() ? text : element.otsl;
        if (!body.empty()) {
            parse_otsl_grid(body, table->mutable_data());
        }
        return true;
    } else if (tag == "page_header" || tag == "page_footer") {
        docv1::BaseTextItem* item =
            add_text(doc, page,
                     tag == "page_header" ? docv1::DOC_ITEM_LABEL_PAGE_HEADER
                                          : docv1::DOC_ITEM_LABEL_PAGE_FOOTER,
                     prov, text);
        item->mutable_text()->mutable_base()->set_content_layer(docv1::CONTENT_LAYER_FURNITURE);
        index = doc->texts_size() - 1;
    } else if (tag == "footnote") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_FOOTNOTE, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag == "caption") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_CAPTION, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag == "checkbox_selected") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_CHECKBOX_SELECTED, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag == "checkbox_unselected") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_CHECKBOX_UNSELECTED, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag == "document_index") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_DOCUMENT_INDEX, prov, text);
        index = doc->texts_size() - 1;
    } else if (tag == "text" || tag == "paragraph") {
        add_text(doc, page, docv1::DOC_ITEM_LABEL_TEXT, prov, text);
        index = doc->texts_size() - 1;
    } else {
        // Open-vocabulary tags (Docling adds labels over time): keep the
        // text as a plain item rather than dropping the page's content.
        if (text.empty() && element.otsl.empty()) {
            return false;
        }
        add_text(doc, page, docv1::DOC_ITEM_LABEL_TEXT, prov, text);
        index = doc->texts_size() - 1;
    }
    if (index >= 0) {
        order->push_back({BodyChild::TEXT, index});
    }
    return true;
}

// <key_value_region>: a KeyValueItem whose GraphData holds the <key_N>/
// <value_N> cells (each with its own box, loc/link tokens stripped from
// the text) plus TO_VALUE links from the cells' <link_N> tokens,
// validated against the emitted cell ids — docling's parse_key_value_item.
// The proto's GraphData covers docling's cells/links shape exactly.
bool emit_key_value_region(const std::string& chunk, const PageContext& page,
                           docv1::Document* doc, std::vector<BodyChild>* order) {
    docv1::KeyValueItem* kv = doc->add_key_value_items();
    kv->set_label(docv1::DOC_ITEM_LABEL_KEY_VALUE_REGION);
    kv->set_content_layer(docv1::CONTENT_LAYER_BODY);
    add_collector_source(kv->mutable_source(), page);

    // The region's own box is the loc run between the opening tag and the
    // first <key_N> cell; without it docling passes prov=None, so no
    // provenance is stamped here either (never an invented box).
    const size_t head_start = chunk.find('>') + 1;
    const size_t first_key = chunk.find("<key_", head_start);
    if (first_key != std::string::npos) {
        const std::vector<long> locs =
            first_locs(chunk.substr(head_start, first_key - head_start));
        if (locs.size() == 4) {
            *kv->add_prov() = prov_with_charspan(page, locs_box(locs, page), 0, 0);
        }
    }

    docv1::GraphData* graph = kv->mutable_graph();
    std::vector<std::pair<int, int>> raw_links;
    std::vector<int> cell_ids;
    size_t pos = head_start;
    while (pos < chunk.size()) {
        const size_t key_pos = chunk.find("<key_", pos);
        const size_t value_pos = chunk.find("<value_", pos);
        size_t open = std::string::npos;
        bool is_key = false;
        if (key_pos != std::string::npos &&
            (value_pos == std::string::npos || key_pos < value_pos)) {
            open = key_pos;
            is_key = true;
        } else if (value_pos != std::string::npos) {
            open = value_pos;
        }
        if (open == std::string::npos) {
            break;
        }
        const size_t id_start = open + (is_key ? 5 : 7);
        const size_t gt = chunk.find('>', id_start);
        if (gt == std::string::npos) {
            break;
        }
        int cell_id = -1;
        try {
            cell_id = std::stoi(chunk.substr(id_start, gt - id_start));
        } catch (...) {
            pos = gt + 1;  // not a cell tag: skip it
            continue;
        }
        // Docling's cell regex requires the matching close tag.
        const std::string close_tag =
            std::string(is_key ? "</key_" : "</value_") + std::to_string(cell_id) + ">";
        const size_t end = chunk.find(close_tag, gt + 1);
        if (end == std::string::npos) {
            pos = gt + 1;
            continue;
        }
        const std::string content = chunk.substr(gt + 1, end - gt - 1);
        pos = end + close_tag.size();

        docv1::GraphCell* cell = graph->add_cells();
        cell->set_label(is_key ? docv1::GRAPH_CELL_LABEL_KEY : docv1::GRAPH_CELL_LABEL_VALUE);
        cell->set_cell_id(cell_id);
        const std::string text = clean_cell_text(content);
        cell->set_text(text);
        cell->set_orig(text);
        const std::vector<long> cell_locs = first_locs(content);
        if (cell_locs.size() == 4) {
            *cell->mutable_prov() = prov_with_charspan(page, locs_box(cell_locs, page), 0, 0);
        }
        cell_ids.push_back(cell_id);
        for (const int target : link_targets(content)) {
            raw_links.push_back({cell_id, target});
        }
    }
    for (const auto& [source, target] : raw_links) {
        // Docling validates the prediction: links to missing cells drop.
        if (!std::ranges::contains(cell_ids, target)) {
            continue;
        }
        docv1::GraphLink* link = graph->add_links();
        link->set_label(docv1::GRAPH_LINK_LABEL_TO_VALUE);
        link->set_source_cell_id(source);
        link->set_target_cell_id(target);
    }

    order->push_back({BodyChild::KEY_VALUE, doc->key_value_items_size() - 1});
    return true;
}

// <ordered_list>/<unordered_list>: a list group in document.groups (label
// LIST, name "list" — docling folds ORDERED_LIST onto LIST) with the
// chunk's <list_item> children parented to it. Ordered items carry
// enumeration markers ("1.", "2.", ...). Source-order emission: the group
// ref sits in the body children where the list chunk appeared.
bool emit_list_group(const std::string& chunk, bool ordered, const PageContext& page,
                     docv1::Document* doc, std::vector<BodyChild>* order) {
    docv1::GroupItem* group = doc->add_groups();
    const std::string group_ref = body_child_ref(BodyChild::GROUP, doc->groups_size() - 1);
    group->set_name("list");
    group->set_label(docv1::GROUP_LABEL_LIST);
    group->set_content_layer(docv1::CONTENT_LAYER_BODY);
    order->push_back({BodyChild::GROUP, doc->groups_size() - 1});

    int enum_value = 0;
    size_t pos = 0;
    while (pos < chunk.size()) {
        const size_t open = chunk.find("<list_item>", pos);
        if (open == std::string::npos) {
            break;
        }
        const size_t content_start = open + 11;
        const size_t end = chunk.find("</list_item>", content_start);
        if (end == std::string::npos) {
            break;  // docling's pattern requires the closing tag
        }
        const std::string content = chunk.substr(content_start, end - content_start);
        pos = end + 12;

        enum_value++;
        const std::string text = inner_text(content);
        const std::string marker = ordered ? std::to_string(enum_value) + "." : "";
        // No locs on the item falls back to the full-page box here (our
        // standing convention); docling would stamp no provenance at all.
        const docv1::ProvenanceItem prov = prov_with_charspan(
            page, locs_box(first_locs(content), page), 0, char_length(text));
        add_list_item(doc, page, prov, text, ordered, marker);
        const int index = doc->texts_size() - 1;
        const std::string self_ref = body_child_ref(BodyChild::TEXT, index);
        internal::stamp_text_item(*doc->mutable_texts(index), self_ref, group_ref);
        group->add_children()->set_ref(self_ref);
    }
    return true;
}

// <inline>: an inline group whose children are the chunk's items, each
// carrying the chunk's first (shared) box — docling's add_inline_group.
bool emit_inline_group(const std::string& chunk, const PageContext& page,
                       docv1::Document* doc, std::vector<BodyChild>* order) {
    docv1::GroupItem* group = doc->add_groups();
    const std::string group_ref = body_child_ref(BodyChild::GROUP, doc->groups_size() - 1);
    group->set_name("group");  // docling's InlineGroup default name
    group->set_label(docv1::GROUP_LABEL_INLINE);
    group->set_content_layer(docv1::CONTENT_LAYER_BODY);
    order->push_back({BodyChild::GROUP, doc->groups_size() - 1});

    const size_t content_start = chunk.find('>') + 1;
    const size_t close = chunk.rfind("</inline>");
    const std::string content = close == std::string::npos || close < content_start
                                    ? chunk.substr(content_start)
                                    : chunk.substr(content_start, close - content_start);
    // The shared box: the chunk's first four locs. Without them the
    // full-page fallback applies (docling would stamp no provenance).
    const std::vector<long> locs = first_locs(content);
    const docv1::BoundingBox box =
        locs.size() == 4 ? locs_box(locs, page) : full_page_box(page);

    size_t pos = 0;
    while (pos < content.size()) {
        const size_t open = content.find('<', pos);
        if (open == std::string::npos) {
            break;
        }
        // A '<' that cannot open a tag is literal text; skip it (same
        // rule as the main tokenizer).
        if (open + 1 == content.size() ||
            (!std::isalpha(static_cast<unsigned char>(content[open + 1])) &&
             content[open + 1] != '/' && content[open + 1] != '_')) {
            pos = open + 1;
            continue;
        }
        const size_t gt = content.find('>', open);
        if (gt == std::string::npos) {
            break;
        }
        const std::string token = content.substr(open + 1, gt - open - 1);
        pos = gt + 1;
        if (token.empty() || token[0] == '/' || token[0] == '_' ||
            token.starts_with("loc_")) {
            continue;
        }
        const std::string close_tag = "</" + token + ">";
        const size_t end = content.find(close_tag, pos);
        if (end == std::string::npos) {
            continue;  // docling's item pattern requires the closing tag
        }
        const std::string child_chunk = content.substr(open, end + close_tag.size() - open);
        pos = end + close_tag.size();

        Element child;
        child.name = token;
        child.text = inner_text(child_chunk);
        std::vector<BodyChild> child_order;
        if (!emit_element(child, page, doc, &child_order, &box)) {
            continue;
        }
        // The child is not a body child: parent it to the group instead.
        for (const BodyChild& emitted : child_order) {
            const std::string self_ref = body_child_ref(emitted.kind, emitted.index);
            if (emitted.kind == BodyChild::TEXT) {
                internal::stamp_text_item(*doc->mutable_texts(emitted.index), self_ref,
                                          group_ref);
            } else if (emitted.kind == BodyChild::PICTURE) {
                docv1::PictureItem* picture = doc->mutable_pictures(emitted.index);
                picture->set_self_ref(self_ref);
                picture->mutable_parent()->set_ref(group_ref);
            } else if (emitted.kind == BodyChild::TABLE) {
                docv1::TableItem* table = doc->mutable_tables(emitted.index);
                table->set_self_ref(self_ref);
                table->mutable_parent()->set_ref(group_ref);
            }
            group->add_children()->set_ref(self_ref);
        }
    }
    return true;
}

}  // namespace

bool map_doctags(const std::string& text, const PageContext& page, docv1::Document* out,
                 std::string* error) {
    if (!text.contains('<')) {
        *error = "response holds no DocTags markup";
        return false;
    }

    size_t items = 0;
    Element current;
    bool element_open = false;
    bool in_otsl = false;
    bool in_caption = false;
    std::vector<long> pending_locs;
    std::vector<BodyChild> order;
    size_t pos = 0;

    auto flush = [&] {
        if (element_open && emit_element(current, page, out, &order)) {
            items++;
        }
        current = Element{};
        element_open = false;
        in_caption = false;
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
                (in_caption ? current.caption_text : current.text) += text.substr(pos);
            }
            break;
        }
        // A '<' that cannot open a tag is literal text, not markup —
        // docling's tag regex requires a letter, '/', or '_' next.
        if (open + 1 == text.size() ||
            (!std::isalpha(static_cast<unsigned char>(text[open + 1])) &&
             text[open + 1] != '/' && text[open + 1] != '_')) {
            if (element_open) {
                (in_caption ? current.caption_text : current.text) +=
                    text.substr(pos, open - pos + 1);
            }
            pos = open + 1;
            continue;
        }
        if (element_open && open > pos) {
            (in_caption ? current.caption_text : current.text) += text.substr(pos, open - pos);
        }
        size_t close = text.find('>', open);
        if (close == std::string::npos) {
            break;  // truncated tag: keep what parsed
        }
        std::string token = text.substr(open + 1, close - open - 1);
        pos = close + 1;

        if (token.starts_with("loc_")) {
            try {
                (in_caption ? current.caption_locs
                            : element_open ? current.locs : pending_locs)
                    .push_back(std::stol(token.substr(4)));
            } catch (...) {
            }
            continue;
        }
        if (token == "doctag" || token == "/doctag") {
            continue;  // root wrapper
        }
        if (!token.empty() && token[0] == '_') {
            // <_..._> sequences (the code language token) are text, not
            // DocTags structure — docling's tag regex keeps them too.
            if (element_open) {
                (in_caption ? current.caption_text : current.text) += "<" + token + ">";
            }
            continue;
        }
        if (in_caption) {
            if (token == "/caption") {
                in_caption = false;
            }
            continue;  // markup inside a caption carries no structure
        }
        if (token == "caption" && element_open && is_container(current.name)) {
            in_caption = true;
            current.has_caption = true;
            continue;
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
        if (is_group_container(token)) {
            // Group/graph containers: capture the raw chunk up to the
            // matching close so nested list items / inline items / cells
            // are not parsed as top-level elements.
            flush();
            const std::string close_tag = "</" + token + ">";
            const size_t end = text.find(close_tag, pos);
            const std::string chunk = end == std::string::npos
                                          ? text.substr(open)
                                          : text.substr(open, end + close_tag.size() - open);
            pos = end == std::string::npos ? text.size() : end + close_tag.size();
            bool emitted = false;
            if (token == "key_value_region") {
                emitted = emit_key_value_region(chunk, page, out, &order);
            } else if (token == "inline") {
                emitted = emit_inline_group(chunk, page, out, &order);
            } else {
                emitted = emit_list_group(chunk, token == "ordered_list", page, out, &order);
            }
            if (emitted) {
                items++;
            }
            continue;
        }
        if (element_open && is_container(current.name) && token == "/" + current.name) {
            flush();
            continue;
        }
        // Classification tags are picture/chart content (docling extracts
        // them from those chunks only); anywhere else a new tag starts a
        // new element as usual.
        if (element_open && (current.name == "picture" || current.name == "chart") &&
            is_classification_tag(token)) {
            current.class_tags.push_back(token);
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
    finalize_document(out, page, order);
    return true;
}

}  // namespace vlm::mapping
