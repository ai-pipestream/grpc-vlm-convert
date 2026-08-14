#pragma once

// Shared helpers for the format mappers: provenance construction, item
// stamping, and the fragment skeleton (body group, self refs, page entry)
// every mapper's Document ends up with.

#include "mapper.h"

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

// Attributes an item to this collector. Sources never overwrite each
// other; the coordinator merges additively.
inline void add_collector_source(
    google::protobuf::RepeatedPtrField<docv1::SourceType>* sources, const PageContext& page) {
    *sources->Add()->mutable_collector() = page.source;
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
    add_collector_source(base->mutable_source(), page);
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
    add_collector_source(code->mutable_source(), page);
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
    add_collector_source(picture->mutable_source(), page);
    return picture;
}

inline docv1::TableItem* add_table(docv1::Document* doc, const PageContext& page,
                                   const docv1::ProvenanceItem& prov) {
    docv1::TableItem* table = doc->add_tables();
    table->set_label(docv1::DOC_ITEM_LABEL_TABLE);
    table->set_content_layer(docv1::CONTENT_LAYER_BODY);
    *table->add_prov() = prov;
    add_collector_source(table->mutable_source(), page);
    return table;
}

namespace internal {

inline void stamp_text_ref(docv1::TextItemBase* base, const std::string& self_ref) {
    base->set_self_ref(self_ref);
    base->mutable_parent()->set_ref("#/body");
}

}  // namespace internal

// Sets the fragment skeleton after items are added: "#/body" group with a
// child ref per item in document order (texts, then pictures, then
// tables — the docling fragment order), self refs and parents, and the
// pages map entry with the raster size.
inline void finalize_document(docv1::Document* doc, const PageContext& page) {
    doc->set_name("page-" + std::to_string(page.page_no));
    docv1::GroupItem* body = doc->mutable_body();
    body->set_self_ref("#/body");
    body->set_content_layer(docv1::CONTENT_LAYER_BODY);

    int index = 0;
    for (docv1::BaseTextItem& item : *doc->mutable_texts()) {
        const std::string self_ref = "#/texts/" + std::to_string(index++);
        switch (item.item_case()) {
            case docv1::BaseTextItem::kTitle:
                internal::stamp_text_ref(item.mutable_title()->mutable_base(), self_ref);
                break;
            case docv1::BaseTextItem::kSectionHeader:
                internal::stamp_text_ref(item.mutable_section_header()->mutable_base(), self_ref);
                break;
            case docv1::BaseTextItem::kListItem:
                internal::stamp_text_ref(item.mutable_list_item()->mutable_base(), self_ref);
                break;
            case docv1::BaseTextItem::kFormula:
                internal::stamp_text_ref(item.mutable_formula()->mutable_base(), self_ref);
                break;
            case docv1::BaseTextItem::kText:
                internal::stamp_text_ref(item.mutable_text()->mutable_base(), self_ref);
                break;
            case docv1::BaseTextItem::kFieldHeading:
                internal::stamp_text_ref(item.mutable_field_heading()->mutable_base(), self_ref);
                break;
            case docv1::BaseTextItem::kFieldValue:
                internal::stamp_text_ref(item.mutable_field_value()->mutable_base(), self_ref);
                break;
            case docv1::BaseTextItem::kCode:
                item.mutable_code()->set_self_ref(self_ref);
                item.mutable_code()->mutable_parent()->set_ref("#/body");
                break;
            default:
                break;
        }
        body->add_children()->set_ref(self_ref);
    }
    index = 0;
    for (docv1::PictureItem& picture : *doc->mutable_pictures()) {
        const std::string self_ref = "#/pictures/" + std::to_string(index++);
        picture.set_self_ref(self_ref);
        picture.mutable_parent()->set_ref("#/body");
        body->add_children()->set_ref(self_ref);
    }
    index = 0;
    for (docv1::TableItem& table : *doc->mutable_tables()) {
        const std::string self_ref = "#/tables/" + std::to_string(index++);
        table.set_self_ref(self_ref);
        table.mutable_parent()->set_ref("#/body");
        body->add_children()->set_ref(self_ref);
    }

    docv1::PageItem& page_item = (*doc->mutable_pages())[static_cast<int32_t>(page.page_no)];
    page_item.set_page_no(static_cast<int32_t>(page.page_no));
    page_item.mutable_size()->set_width(page.width);
    page_item.mutable_size()->set_height(page.height);
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
