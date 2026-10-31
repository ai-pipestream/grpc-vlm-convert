#pragma once

// Shared machinery for the layout-grounded formats docling added after
// DocTags (DocLang, DeepSeek-OCR, Unlimited-OCR, Chandra, dots.ocr,
// Nemotron Parse): a writer that builds one page's Document tree (items
// nested under groups, pictures and tables, captions linked, furniture
// layered, picture crops attached), span-aware table grids, and the text
// clean-ups docling applies to model bodies.

#include <map>
#include <tuple>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "builder.h"
#include "image_crop.h"
#include "mapper.h"
#include "markup_tree.h"

namespace vlm::mapping {

// One table cell as a model wrote it, before the grid is laid out.
struct SpanCell {
    size_t row = 0;
    size_t col = 0;
    size_t row_span = 1;
    size_t col_span = 1;
    std::string text;
    bool column_header = false;
    bool row_header = false;
    // The element the cell came from, when the source was markup.
    const MarkupNode* node = nullptr;
};

// Lays `cells` out as TableData: table_cells holds one entry per anchor,
// the grid stamps each anchor over every position it spans and fills the
// rest with empty cells. num_rows and num_cols are the furthest span ends.
// The builder.h caps apply (leading rows and columns kept, spans clipped,
// the grid left empty past kMaxGridBytes); the returned cut says what was
// dropped.
TableCut fill_span_table(docv1::TableData* data, const std::vector<SpanCell>& cells);

// Docling's text of an HTML element (chandra_utils _Element.text): <br>
// is a line break, a checkbox or radio input is a ballot box (checked or
// not), a block element ends with a line break.
std::string html_text(const MarkupNode& node);

// Collapses every whitespace run to one space and trims.
std::string collapse_whitespace(std::string_view text);

// Cells of an HTML table the way docling's Chandra mapper reads them
// (_table_cells): direct rows and thead/tbody/tfoot rows, colspan and
// rowspan (rowspan="0" runs to the end of the row group), occupied
// positions skipped; a header cell is scope="col" or a <th> in <thead> or
// row 0, a row header is scope="row" or a <th> in column 0 below row 0
// outside <thead>.
std::vector<SpanCell> chandra_table_cells(const MarkupNode& table);

// The first MarkupNode named `tag` under `node` (depth-first), or null.
const MarkupNode* find_element(const MarkupNode& node, std::string_view tag);

// Docling's _markdown_inline_html: text holding * or _ is rendered as
// paragraph-only Markdown; with emphasis the result is inline HTML (line
// breaks as <br>), without it the text with \* and \_ unescaped. Text
// without either character comes back unchanged.
std::string markdown_inline_html(const std::string& text);

// Whether `html` holds one of the inline tags docling's dots mapper hands
// to the HTML builder (_HTML_TAG_PATTERN).
bool has_inline_html_tag(const std::string& html);

// Docling's parse_markdown_heading: "## Text" -> ("Text", 2). No marker
// gives the text unchanged and level 0.
std::string strip_markdown_heading(const std::string& text, int* level);

// Docling's strip_math_delimiters: drops \( and \) (keeping \\), then
// unwraps an outer $$...$$, \[...\] or $...$ while one wraps the whole
// formula.
std::string strip_math_delimiters(const std::string& text);

// Markdown escapes \* and \_ become literal characters (dots_utils
// _unescape_markdown_literals, outside code spans and math).
std::string unescape_markdown_literals(const std::string& text);

// Plain text of a short Markdown body (Nemotron Parse 2.0's
// _markdown_to_text): heading markers, emphasis markers, link targets and
// code-span backticks dropped, backslash escapes resolved, blank lines
// removed.
std::string markdown_to_text(const std::string& markdown);

// The Qwen2-VL smart_resize input size docling assumes for dots.ocr boxes
// (vlm_utils compute_qwen2vl_image_size with scale 1 and no max size).
std::pair<double, double> qwen2vl_image_size(double width, double height);

// A node handle for LayoutWriter: an item list and an index into it. The
// default value is the page body.
struct LayoutRef {
    BodyChild::Kind kind = BodyChild::GROUP;
    int index = -1;
    bool is_body() const { return index < 0; }
    bool operator==(const LayoutRef&) const = default;
};

// Builds one page fragment. Items are created in output order; each
// lives under the body or under a parent node, and finish() stamps every
// self_ref, parent and children list.
class LayoutWriter {
  public:
    using Ref = LayoutRef;
    static constexpr Ref kBody{};

    LayoutWriter(const PageContext& page, docv1::Document* out,
                 std::vector<vlmv1::PageWarning>* warnings);

    // A page box from coordinates on a `grid`-unit square (0..grid on
    // both axes), or the full page when `coords` is empty.
    docv1::BoundingBox grid_box(const std::vector<double>& coords, double grid) const;
    // A page box from coordinates scaled by (sx, sy), clamped to the page.
    docv1::BoundingBox scaled_box(double l, double t, double r, double b, double sx,
                                  double sy) const;

    // Text items. A missing box falls back to the full page (the service's
    // standing convention; docling would leave the item without
    // provenance). PAGE_HEADER and PAGE_FOOTER land in the furniture
    // layer.
    Ref text(docv1::DocItemLabel label, const std::optional<docv1::BoundingBox>& box,
             const std::string& text, Ref parent = kBody);
    Ref title(const std::optional<docv1::BoundingBox>& box, const std::string& text,
              Ref parent = kBody);
    // `level` is the raw heading depth (docling normalizes it once per
    // document, which a single page cannot).
    Ref heading(int level, const std::optional<docv1::BoundingBox>& box,
                const std::string& text, Ref parent = kBody);
    Ref formula(const std::optional<docv1::BoundingBox>& box, const std::string& text,
                Ref parent = kBody);
    Ref code(const std::optional<docv1::BoundingBox>& box, const std::string& text,
             const std::string& language_raw, Ref parent = kBody);
    Ref list_item(const std::optional<docv1::BoundingBox>& box, const std::string& text,
                  bool enumerated, const std::string& marker, Ref parent);
    // Groups: a list group ("list", GROUP_LABEL_LIST) or a named one.
    Ref list_group(Ref parent = kBody);
    Ref group(const std::string& name, docv1::GroupLabel label, Ref parent = kBody);
    // A form field region (docling add_field_region).
    Ref field_region(const std::optional<docv1::BoundingBox>& box, Ref parent = kBody);
    // An inline group (docling add_inline_group): runs of one paragraph.
    Ref inline_group(Ref parent = kBody);
    // A text item of any label from a formatted run: TITLE, SECTION_HEADER
    // (level 1 until set), FORMULA, CODE and FIELD_VALUE (kind "fillable")
    // get their own item kinds, every other label a TextItem.
    Ref styled_text(docv1::DocItemLabel label, const std::optional<docv1::BoundingBox>& box,
                    const std::string& text, const docv1::Formatting* formatting,
                    const std::string& hyperlink, Ref parent = kBody);
    // Fills an item's text after the fact (docling sets an empty list
    // item's text from its one plain run).
    void set_text(Ref ref, const std::string& text, const docv1::Formatting* formatting,
                  const std::string& hyperlink);
    // Sets an item's orig (the model's raw text) apart from its text.
    void set_orig(Ref ref, const std::string& orig);
    // The item's text, and how many children it holds so far.
    std::string text_of(Ref ref) const;
    size_t children_of(Ref ref) const;
    int text_count() const { return out_->texts_size(); }
    void set_heading_level(int text_index, int level);
    void set_content_layer(Ref ref, docv1::ContentLayer layer);

    // A picture; a real box crops the page raster into its image.
    Ref picture(const std::optional<docv1::BoundingBox>& box, Ref parent = kBody);
    // A table holding `cells` (empty for a table the model left unreadable).
    Ref table(const std::optional<docv1::BoundingBox>& box, const std::vector<SpanCell>& cells,
              Ref parent = kBody);

    // Picture metadata, each in meta and in the annotations union the
    // way the DocTags mapper writes it; `created_by` names the producer.
    void describe_picture(Ref picture, const std::string& description,
                          const std::string& created_by);
    void classify_picture(Ref picture, const std::string& class_name,
                          const std::string& created_by);
    void set_molecule(Ref picture, const std::string& smiles, const std::string& created_by);
    // Chart data from rows of cell text (row 0 the header row).
    void set_tabular_chart(Ref picture, const std::string& title,
                           const std::vector<std::vector<std::string>>& rows,
                           const std::string& created_by);

    // Links `caption` (a text item) as a caption or footnote of `owner`.
    void link_caption(Ref owner, Ref caption);
    void link_footnote(Ref owner, Ref footnote);
    bool has_captions(Ref owner) const;

    docv1::PictureItem* picture_item(Ref ref) { return out_->mutable_pictures(ref.index); }
    docv1::TableItem* table_item(Ref ref) { return out_->mutable_tables(ref.index); }
    docv1::BaseTextItem* text_item(Ref ref) { return out_->mutable_texts(ref.index); }
    // The label of a text item, or PICTURE / TABLE for those kinds.
    docv1::DocItemLabel label_of(Ref ref) const;

    // Body children so far, for docling's adjacent-caption pass.
    size_t body_size() const { return body_.size(); }
    const std::vector<Ref>& body() const { return body_; }
    size_t item_count() const { return entries_.size(); }
    // The most recently added table.
    Ref last_table() const { return Ref{BodyChild::TABLE, out_->tables_size() - 1}; }
    std::vector<vlmv1::PageWarning>* warnings() const { return warnings_; }

    // Docling's link_adjacent_captions: block i spans body children
    // [starts[i], starts[i+1]); a caption block holding exactly one
    // non-empty caption is linked to its one immediate neighbor block that
    // is a single item of an owner kind (PICTURE or TABLE) with no caption
    // yet. Ambiguous captions stay unlinked.
    // With `code_owners` a single code item also counts as a neighbor
    // owner (dots.ocr), which leaves the caption unlinked.
    void link_adjacent_captions(const std::vector<size_t>& starts,
                                const std::vector<size_t>& caption_blocks,
                                bool code_owners = false);

    // Stamps refs, the page entry and the crop warnings. Returns false
    // with `error` set when the page produced no item at all.
    bool finish(std::string* error, const std::string& format_name, bool allow_empty = false);

  private:
    struct Entry {
        Ref ref;
        Ref parent;
    };
    Ref add_entry(BodyChild::Kind kind, int index, Ref parent);
    docv1::ProvenanceItem prov(const std::optional<docv1::BoundingBox>& box, int32_t chars) const;
    void stamp_layer(docv1::BaseTextItem* item, docv1::DocItemLabel label);

    const PageContext& page_;
    docv1::Document* out_;
    std::vector<vlmv1::PageWarning>* warnings_;
    PageRaster raster_;
    size_t crops_over_budget_ = 0;
    size_t crops_over_bytes_ = 0;
    size_t crops_too_large_ = 0;
    std::vector<Entry> entries_;
    std::map<std::pair<int, int>, size_t> child_counts_;
    std::vector<Ref> body_;
};

// DeepSeek-OCR helpers, exposed for tests (deepseek.cpp): the cells of a
// grounded block's <table>, and Unlimited-OCR's inline annotations
// rewritten to DeepSeek-OCR annotation lines.
std::vector<SpanCell> deepseek_table_cells(const std::string& content);
std::string normalize_unlimited_ocr(const std::string& content);

// HTML through docling's Chandra builder, for the mappers that embed
// HTML (dots.ocr, Nemotron Parse 3.5); defined in chandra.cpp.
//
// _add_html_fragment: walks `html` under `parent` with `label`, the box
// going to the first text item. False when the HTML holds no element.
bool add_html_fragment(LayoutWriter& writer, const std::string& html, docv1::DocItemLabel label,
                       const std::optional<docv1::BoundingBox>& box, LayoutRef parent);
// _add_html_table: the first <table> in `html` (captions flattened to
// text) as one rich table under `parent`. Nullopt without a table or
// without cells; nothing is added then.
std::optional<LayoutRef> add_html_table(LayoutWriter& writer, const std::string& html,
                                        const std::optional<docv1::BoundingBox>& box,
                                        LayoutRef parent);
// dots_utils _add_inline_text: Markdown emphasis and inline HTML through
// the builder. Without a parent, a text item of `label` holding the box
// becomes the parent. False when the text holds no inline markup (the
// caller then adds plain text).
bool add_inline_text(LayoutWriter& writer, const std::string& text, docv1::DocItemLabel label,
                     const std::optional<docv1::BoundingBox>& box,
                     std::optional<LayoutRef> parent);

}  // namespace vlm::mapping
