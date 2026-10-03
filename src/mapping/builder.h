#pragma once

// Shared helpers for the format mappers: provenance construction, item
// stamping, and the fragment skeleton (body group, self refs, page entry)
// every mapper's Document ends up with.

#include "mapper.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace vlm::mapping {

// The full-page box — the only provenance formats without real locations
// (markdown, HTML, plaintext) are allowed to carry. Never invent tighter
// boxes for them.
inline docv1::BoundingBox full_page_box(const PageContext& page) {
    docv1::BoundingBox box;
    box.set_l(0);
    box.set_t(0);
    box.set_r(page.width);
    box.set_b(page.height);
    box.set_coord_origin(docv1::COORD_ORIGIN_TOPLEFT);
    return box;
}

inline docv1::ProvenanceItem make_prov(const PageContext& page,
                                       const docv1::BoundingBox& box) {
    docv1::ProvenanceItem prov;
    prov.set_page_no(static_cast<int32_t>(page.page_no));
    *prov.mutable_bbox() = box;
    return prov;
}

inline docv1::ProvenanceItem page_prov(const PageContext& page) {
    return make_prov(page, full_page_box(page));
}

// Attributes an item to this collector and, when the page carries one, to
// the model invocation that generated it. Sources never overwrite each
// other; the coordinator merges additively.
inline void add_sources(google::protobuf::RepeatedPtrField<docv1::SourceType>* sources,
                        const PageContext& page) {
    *sources->Add()->mutable_collector() = page.source;
    if (page.has_generation) {
        *sources->Add()->mutable_generation() = page.generation;
    }
}

// Common stamping for a text item: provenance, source, orig/text.
inline void fill_text_base(docv1::TextItemBase* base, const PageContext& page,
                           docv1::DocItemLabel label, const docv1::ProvenanceItem& prov,
                           const std::string& text) {
    base->set_label(label);
    base->set_content_layer(docv1::CONTENT_LAYER_BODY);
    *base->add_prov() = prov;
    base->set_orig(text);
    base->set_text(text);
    add_sources(base->mutable_source(), page);
}

inline docv1::BaseTextItem* add_text(docv1::Document* doc, const PageContext& page,
                                     docv1::DocItemLabel label,
                                     const docv1::ProvenanceItem& prov,
                                     const std::string& text) {
    docv1::BaseTextItem* item = doc->add_texts();
    fill_text_base(item->mutable_text()->mutable_base(), page, label, prov, text);
    return item;
}

inline docv1::BaseTextItem* add_section_header(docv1::Document* doc, const PageContext& page,
                                               int level, const docv1::ProvenanceItem& prov,
                                               const std::string& text) {
    docv1::BaseTextItem* item = doc->add_texts();
    fill_text_base(item->mutable_section_header()->mutable_base(), page,
                   docv1::DOC_ITEM_LABEL_SECTION_HEADER, prov, text);
    item->mutable_section_header()->set_level(level);
    return item;
}

inline docv1::BaseTextItem* add_title(docv1::Document* doc, const PageContext& page,
                                      const docv1::ProvenanceItem& prov,
                                      const std::string& text) {
    docv1::BaseTextItem* item = doc->add_texts();
    fill_text_base(item->mutable_title()->mutable_base(), page, docv1::DOC_ITEM_LABEL_TITLE,
                   prov, text);
    return item;
}

inline docv1::BaseTextItem* add_list_item(docv1::Document* doc, const PageContext& page,
                                          const docv1::ProvenanceItem& prov,
                                          const std::string& text, bool enumerated,
                                          const std::string& marker) {
    docv1::BaseTextItem* item = doc->add_texts();
    fill_text_base(item->mutable_list_item()->mutable_base(), page,
                   docv1::DOC_ITEM_LABEL_LIST_ITEM, prov, text);
    item->mutable_list_item()->set_enumerated(enumerated);
    if (!marker.empty()) {
        item->mutable_list_item()->set_marker(marker);
    }
    return item;
}

inline docv1::BaseTextItem* add_code(docv1::Document* doc, const PageContext& page,
                                     const docv1::ProvenanceItem& prov, const std::string& text,
                                     const std::string& language_raw) {
    docv1::BaseTextItem* item = doc->add_texts();
    // CodeItem inlines the base fields (see document.proto) — no base wrapper.
    docv1::CodeItem* code = item->mutable_code();
    code->set_label(docv1::DOC_ITEM_LABEL_CODE);
    code->set_content_layer(docv1::CONTENT_LAYER_BODY);
    *code->add_prov() = prov;
    code->set_orig(text);
    code->set_text(text);
    if (!language_raw.empty()) {
        code->set_code_language_raw(language_raw);
    }
    add_sources(code->mutable_source(), page);
    return item;
}

inline docv1::BaseTextItem* add_formula(docv1::Document* doc, const PageContext& page,
                                        const docv1::ProvenanceItem& prov,
                                        const std::string& text) {
    docv1::BaseTextItem* item = doc->add_texts();
    fill_text_base(item->mutable_formula()->mutable_base(), page, docv1::DOC_ITEM_LABEL_FORMULA,
                   prov, text);
    return item;
}

inline docv1::PictureItem* add_picture(docv1::Document* doc, const PageContext& page,
                                       const docv1::ProvenanceItem& prov) {
    docv1::PictureItem* picture = doc->add_pictures();
    picture->set_label(docv1::DOC_ITEM_LABEL_PICTURE);
    picture->set_content_layer(docv1::CONTENT_LAYER_BODY);
    *picture->add_prov() = prov;
    add_sources(picture->mutable_source(), page);
    return picture;
}

inline docv1::TableItem* add_table(docv1::Document* doc, const PageContext& page,
                                   const docv1::ProvenanceItem& prov) {
    docv1::TableItem* table = doc->add_tables();
    table->set_label(docv1::DOC_ITEM_LABEL_TABLE);
    table->set_content_layer(docv1::CONTENT_LAYER_BODY);
    *table->add_prov() = prov;
    add_sources(table->mutable_source(), page);
    return table;
}

// A model stuck repeating itself (SmolDocling and Granite-Docling do) can
// emit one enormous row and then thousands of short ones, and every table
// here is a full rows × columns grid of TableCells: 4000 × 4000 is sixteen
// million of them, gigabytes for one page. Tables keep their leading rows
// and columns up to these caps (a real page's tables stay far inside
// them), and the page says what was cut in a PageWarning.
constexpr size_t kMaxTableRows = 2000;
constexpr size_t kMaxTableCols = 250;
// rows × columns of the grid.
constexpr size_t kMaxTableCells = 50000;
// What one table's grid may cost in cell copies: every covered position
// holds a full copy of its anchor cell, text included, so overlapping
// spans or long spanned text multiply. Past it the grid is left empty and
// the cells stay.
constexpr size_t kMaxGridBytes = 32ULL * 1024 * 1024;

// What building one table cut, and the shape the source had.
struct TableCut {
    size_t source_rows = 0;
    size_t source_cols = 0;
    // Rows or columns past the caps were dropped.
    bool truncated = false;
    // The grid was left empty (kMaxGridBytes).
    bool grid_omitted = false;
};

// The leading part of a rows × cols source that fits the caps.
inline std::pair<size_t, size_t> kept_table_shape(size_t rows, size_t cols) {
    const size_t kept_cols = std::min(cols, kMaxTableCols);
    size_t kept_rows = std::min(rows, kMaxTableRows);
    if (kept_cols > 0) {
        kept_rows = std::min(kept_rows, kMaxTableCells / kept_cols);
    }
    return {kept_rows, kept_cols};
}

inline void add_warning(std::vector<vlmv1::PageWarning>* warnings, vlmv1::PageWarningCode code,
                        std::string message, std::string ref) {
    if (warnings == nullptr) {
        return;
    }
    vlmv1::PageWarning& warning = warnings->emplace_back();
    warning.set_code(code);
    warning.set_message(std::move(message));
    warning.set_ref(std::move(ref));
}

// Reports what building the table at `ref` cut, if anything. `what` names
// it in the message ("table", "chart data").
inline void note_table_cut(const TableCut& cut, const docv1::TableData& data,
                           const std::string& what, const std::string& ref,
                           std::vector<vlmv1::PageWarning>* warnings) {
    if (cut.truncated) {
        add_warning(warnings, vlmv1::PAGE_WARNING_CODE_TABLE_TRUNCATED,
                    what + " held " + std::to_string(cut.source_rows) + " rows by " +
                        std::to_string(cut.source_cols) + " columns; kept the first " +
                        std::to_string(data.num_rows()) + " by " +
                        std::to_string(data.num_cols()) + " (caps: " +
                        std::to_string(kMaxTableRows) + " rows, " +
                        std::to_string(kMaxTableCols) + " columns, " +
                        std::to_string(kMaxTableCells) + " cells)",
                    ref);
    }
    if (cut.grid_omitted) {
        add_warning(warnings, vlmv1::PAGE_WARNING_CODE_TABLE_GRID_OMITTED,
                    what + " grid would cost more than " + std::to_string(kMaxGridBytes) +
                        " bytes of cell copies (overlapping or long spanned cells); kept its " +
                        std::to_string(data.table_cells_size()) + " cells with an empty grid",
                    ref);
    }
}

// Fills a TableData from rows of already-split cell text: a rectangular
// grid of 1x1 cells, `table_cells` mirroring the grid, and the first
// `header_rows` rows flagged as column headers. Ragged source rows still
// produce a rectangular grid (the grid invariant): short rows pad with
// empty 1x1 cells up to the widest row. Only the leading rows and columns
// within the table caps are kept. Shared by every mapper whose source
// gives it rows and cells but no spans.
inline TableCut fill_table_data(docv1::TableData* data,
                                const std::vector<std::vector<std::string>>& rows,
                                size_t header_rows) {
    TableCut cut;
    cut.source_rows = rows.size();
    for (const std::vector<std::string>& row : rows) {
        cut.source_cols = std::max(cut.source_cols, row.size());
    }
    const auto [num_rows, kept_cols] = kept_table_shape(cut.source_rows, cut.source_cols);
    cut.truncated = num_rows < cut.source_rows || kept_cols < cut.source_cols;
    // The widest kept row sets the grid width (the widest source row may be
    // one of the dropped ones).
    size_t num_cols = 0;
    for (size_t r = 0; r < num_rows; r++) {
        num_cols = std::max(num_cols, std::min(rows[r].size(), kept_cols));
    }
    data->set_num_rows(static_cast<int32_t>(num_rows));
    data->set_num_cols(static_cast<int32_t>(num_cols));
    for (size_t r = 0; r < num_rows; r++) {
        docv1::TableRow* grid_row = data->add_grid();
        for (size_t c = 0; c < num_cols; c++) {
            docv1::TableCell cell;
            if (c < rows[r].size()) {
                cell.set_text(rows[r][c]);
                cell.set_column_header(r < header_rows);
            }
            cell.set_row_span(1);
            cell.set_col_span(1);
            cell.set_start_row_offset_idx(static_cast<int32_t>(r));
            cell.set_end_row_offset_idx(static_cast<int32_t>(r + 1));
            cell.set_start_col_offset_idx(static_cast<int32_t>(c));
            cell.set_end_col_offset_idx(static_cast<int32_t>(c + 1));
            *grid_row->add_cells() = cell;
            // Padding cells fill the grid only; they are not source cells.
            if (c < rows[r].size()) {
                *data->add_table_cells() = cell;
            }
        }
    }
    return cut;
}

namespace internal {

inline void stamp_text_ref(docv1::TextItemBase* base, const std::string& self_ref,
                           const std::string& parent_ref) {
    base->set_self_ref(self_ref);
    base->mutable_parent()->set_ref(parent_ref);
}

inline void stamp_text_item(docv1::BaseTextItem& item, const std::string& self_ref,
                            const std::string& parent_ref) {
    switch (item.item_case()) {
        case docv1::BaseTextItem::kTitle:
            stamp_text_ref(item.mutable_title()->mutable_base(), self_ref, parent_ref);
            break;
        case docv1::BaseTextItem::kSectionHeader:
            stamp_text_ref(item.mutable_section_header()->mutable_base(), self_ref, parent_ref);
            break;
        case docv1::BaseTextItem::kListItem:
            stamp_text_ref(item.mutable_list_item()->mutable_base(), self_ref, parent_ref);
            break;
        case docv1::BaseTextItem::kFormula:
            stamp_text_ref(item.mutable_formula()->mutable_base(), self_ref, parent_ref);
            break;
        case docv1::BaseTextItem::kText:
            stamp_text_ref(item.mutable_text()->mutable_base(), self_ref, parent_ref);
            break;
        case docv1::BaseTextItem::kFieldHeading:
            stamp_text_ref(item.mutable_field_heading()->mutable_base(), self_ref, parent_ref);
            break;
        case docv1::BaseTextItem::kFieldValue:
            stamp_text_ref(item.mutable_field_value()->mutable_base(), self_ref, parent_ref);
            break;
        case docv1::BaseTextItem::kCode:
            item.mutable_code()->set_self_ref(self_ref);
            item.mutable_code()->mutable_parent()->set_ref(parent_ref);
            break;
        default:
            break;
    }
}

// The common case: items whose parent is the body group.
inline void stamp_text_item(docv1::BaseTextItem& item, const std::string& self_ref) {
    stamp_text_item(item, self_ref, "#/body");
}

}  // namespace internal

// One body child in emission order: which item list and the index in it.
// Mappers that know their source order (DocTags) pass it through; the
// others default to texts → pictures → tables. GROUP entries are the
// DocTags list/inline groups; their children are parented to the group,
// not the body, at emission time.
struct BodyChild {
    enum Kind { TEXT = 0, PICTURE = 1, TABLE = 2, KEY_VALUE = 3, GROUP = 4 };
    Kind kind;
    int index;
};

inline std::string body_child_ref(BodyChild::Kind kind, int index) {
    const char* list = nullptr;
    switch (kind) {
        case BodyChild::TEXT:
            list = "texts";
            break;
        case BodyChild::PICTURE:
            list = "pictures";
            break;
        case BodyChild::TABLE:
            list = "tables";
            break;
        case BodyChild::KEY_VALUE:
            list = "key_value_items";
            break;
        case BodyChild::GROUP:
            list = "groups";
            break;
    }
    return "#/" + std::string(list) + "/" + std::to_string(index);
}

// Sets the fragment skeleton after items are added: "#/body" group with a
// child ref per item in the given order, self refs and parents, and the
// pages map entry with the raster size.
inline void finalize_document(docv1::Document* doc, const PageContext& page,
                              const std::vector<BodyChild>& order) {
    // Root schema identity: the wire schema name and minor every producer
    // in the fleet stamps on a finished Document.
    doc->set_schema_name("docling_document_v2");
    doc->set_version("1.10.0");
    doc->set_name("page-" + std::to_string(page.page_no));
    docv1::GroupItem* body = doc->mutable_body();
    body->set_self_ref("#/body");
    body->set_content_layer(docv1::CONTENT_LAYER_BODY);
    // The page's alternate readings belong to the page, so they hang on
    // its container rather than on any one item.
    if (page.has_alternatives) {
        *body->mutable_meta()->mutable_alternatives() = page.alternatives;
    }

    for (const BodyChild& child : order) {
        const std::string self_ref = body_child_ref(child.kind, child.index);
        if (child.kind == BodyChild::TEXT) {
            internal::stamp_text_item(*doc->mutable_texts(child.index), self_ref);
        } else if (child.kind == BodyChild::PICTURE) {
            docv1::PictureItem* picture = doc->mutable_pictures(child.index);
            picture->set_self_ref(self_ref);
            picture->mutable_parent()->set_ref("#/body");
        } else if (child.kind == BodyChild::TABLE) {
            docv1::TableItem* table = doc->mutable_tables(child.index);
            table->set_self_ref(self_ref);
            table->mutable_parent()->set_ref("#/body");
        } else if (child.kind == BodyChild::KEY_VALUE) {
            docv1::KeyValueItem* kv = doc->mutable_key_value_items(child.index);
            kv->set_self_ref(self_ref);
            kv->mutable_parent()->set_ref("#/body");
        } else {
            // Groups: self ref and body parent here; their children were
            // parented to the group when it was emitted.
            docv1::GroupItem* group = doc->mutable_groups(child.index);
            group->set_self_ref(self_ref);
            group->mutable_parent()->set_ref("#/body");
        }
        body->add_children()->set_ref(self_ref);
    }

    docv1::PageItem& page_item = (*doc->mutable_pages())[static_cast<int32_t>(page.page_no)];
    page_item.set_page_no(static_cast<int32_t>(page.page_no));
    page_item.mutable_size()->set_width(page.width);
    page_item.mutable_size()->set_height(page.height);
}

// The legacy fragment order — texts, then pictures, then tables — for
// mappers whose source has no interleaved reading order to preserve.
inline void finalize_document(docv1::Document* doc, const PageContext& page) {
    std::vector<BodyChild> order;
    for (int i = 0; i < doc->texts_size(); i++) {
        order.push_back({BodyChild::TEXT, i});
    }
    for (int i = 0; i < doc->pictures_size(); i++) {
        order.push_back({BodyChild::PICTURE, i});
    }
    for (int i = 0; i < doc->tables_size(); i++) {
        order.push_back({BodyChild::TABLE, i});
    }
    finalize_document(doc, page, order);
}

inline std::string trim(const std::string& text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

}  // namespace vlm::mapping
