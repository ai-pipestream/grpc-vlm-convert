// Unit tests for the docling 2.142.1 response formats 6-13 through
// map_response: each case is a model answer shaped like docling's own
// tests and fixtures (tests/test_*_vlm.py, tests/data/{md_deepseek,
// html_chandra,json_dots}), checked for the items docling builds from it.

#include <cmath>

#include "fixture.h"
#include "mapping/layout.h"
#include "mapping/mapper.h"

namespace docv1 = ai::pipestream::document::v1;
namespace vlmv1 = ai::pipestream::vlm::v1;

namespace {

vlm::mapping::PageContext page_context(uint32_t width = 1000, uint32_t height = 1000) {
    vlm::mapping::PageContext page;
    page.page_no = 1;
    page.width = width;
    page.height = height;
    page.source.set_collector("vlm-convert");
    return page;
}

docv1::Document map_or_fail(vlmv1::ResponseFormat format, const std::string& text,
                            const vlm::mapping::PageContext& page = page_context()) {
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_response(format, text, page, &doc, &error),
            vlmv1::ResponseFormat_Name(format) + " maps: " + error);
    return doc;
}

// Code items are flat (no TextItemBase); callers read item.code() directly.
const docv1::TextItemBase& base(const docv1::BaseTextItem& item) {
    static const docv1::TextItemBase kNone;
    switch (item.item_case()) {
        case docv1::BaseTextItem::kTitle:
            return item.title().base();
        case docv1::BaseTextItem::kSectionHeader:
            return item.section_header().base();
        case docv1::BaseTextItem::kListItem:
            return item.list_item().base();
        case docv1::BaseTextItem::kCode:
            return kNone;
        case docv1::BaseTextItem::kFormula:
            return item.formula().base();
        case docv1::BaseTextItem::kFieldHeading:
            return item.field_heading().base();
        case docv1::BaseTextItem::kFieldValue:
            return item.field_value().base();
        default:
            return item.text().base();
    }
}

const docv1::TextItemBase* find_text(const docv1::Document& doc, const std::string& needle) {
    for (const auto& item : doc.texts()) {
        if (base(item).text().find(needle) != std::string::npos) {
            return &base(item);
        }
    }
    return nullptr;
}

bool near(double a, double b) { return std::abs(a - b) < 0.01; }

const docv1::TableCell* cell_at(const docv1::TableData& data, int row, int col) {
    for (const auto& cell : data.table_cells()) {
        if (static_cast<int>(cell.start_row_offset_idx()) == row &&
            static_cast<int>(cell.start_col_offset_idx()) == col) {
            return &cell;
        }
    }
    return nullptr;
}

// tests/data/md_deepseek/sources/deepseek_simple.md, shortened: the
// unlabeled lead paragraph is dropped, sub_title blocks are headings.
void verify_deepseekocr_markdown() {
    const docv1::Document doc = map_or_fail(
        vlmv1::RESPONSE_FORMAT_DEEPSEEKOCR_MARKDOWN,
        "order to compute the TED score.  \n\n"
        "<|ref|>sub_title<|/ref|><|det|>[[217, 209, 520, 225]]<|/det|>\n"
        "### 5.1 Hyper Parameter Optimization  \n\n"
        "<|ref|>text<|/ref|><|det|>[[217, 230, 785, 321]]<|/det|>\n"
        "We have chosen the PubTabNet data set to perform HPO.  \n\n"
        "<|ref|>sub_title<|/ref|><|det|>[[217, 636, 432, 652]]<|/det|>\n"
        "### 5.2 Quantitative Results  \n");
    require(doc.texts_size() == 3, "deepseek: two headings and one text (lead text dropped)");
    require(doc.texts(0).has_section_header() &&
                base(doc.texts(0)).text() == "5.1 Hyper Parameter Optimization",
            "deepseek: sub_title is a section header");
    require(find_text(doc, "order to compute") == nullptr, "deepseek: unlabeled text dropped");
    const docv1::BoundingBox& box = base(doc.texts(1)).prov(0).bbox();
    require(near(box.l(), 217.0 / 999 * 1000) || near(box.l(), 217),
            "deepseek: box scaled to the page");
}

// test_unlimited_ocr_markdown.py::test_parse_builds_document_with_heading_and_table
void verify_unlimited_ocr_markdown() {
    require(vlm::mapping::normalize_unlimited_ocr(
                "<|det|>title [52, 40, 816, 63]<|/det|>Reconciliation report No. 1481") ==
                "<|ref|>title<|/ref|><|det|>[[52, 40, 816, 63]]<|/det|>\n"
                "Reconciliation report No. 1481",
            "unlimited: annotation moves to ref onto its own line");
    const std::string shaped =
        "<|ref|>title<|/ref|><|det|>[[52, 40, 816, 63]]<|/det|>\nAlready normalized";
    require(vlm::mapping::normalize_unlimited_ocr(shaped) == shaped,
            "unlimited: deepseek-shaped content untouched");
    const docv1::Document doc = map_or_fail(
        vlmv1::RESPONSE_FORMAT_UNLIMITED_OCR_MARKDOWN,
        "<|det|>title [52, 40, 816, 63]<|/det|>Reconciliation report No. 1481\n"
        "<|det|>text [52, 71, 411, 87]<|/det|>Period: 01.04.2026 to 30.06.2026\n"
        "<|det|>table [51, 216, 601, 420]<|/det|>"
        "<table><tr><td>Date</td><td>Debit</td></tr>"
        "<tr><td>05.04.2026</td><td>184 320.00</td></tr></table>",
        page_context(880, 1200));
    require(doc.tables_size() == 1, "unlimited: one table");
    require(find_text(doc, "Reconciliation report No. 1481") != nullptr, "unlimited: title text");
    const docv1::TableCell* debit = cell_at(doc.tables(0).data(), 1, 1);
    require(debit != nullptr && debit->text() == "184 320.00", "unlimited: table cell text");
}

// tests/data/html_chandra/sources/chandra_simple.html, shortened.
void verify_chandra_html() {
    const docv1::Document doc = map_or_fail(
        vlmv1::RESPONSE_FORMAT_CHANDRA_HTML,
        "<div data-bbox=\"312 115 786 128\" data-label=\"Page-Header\">Optimized Table "
        "Tokenization 9</div>\n"
        "<div data-bbox=\"215 210 521 225\" data-label=\"Section-Header\">\n"
        "<h3>5.1 Hyper Parameter Optimization</h3>\n</div>\n"
        "<div data-bbox=\"215 230 788 320\" data-label=\"Text\">\n"
        "<p>OTSL yields a <i>2x speed up</i> over HTML.</p>\n</div>\n"
        "<div data-bbox=\"225 422 770 590\" data-label=\"Table\">\n"
        "<table border=\"1\"><thead><tr><th colspan=\"2\">TEDs</th></tr></thead>"
        "<tbody><tr><td>0.965</td><td>0.934</td></tr></tbody></table>\n</div>");
    const docv1::TextItemBase* header = find_text(doc, "Optimized Table");
    require(header != nullptr && header->label() == docv1::DOC_ITEM_LABEL_PAGE_HEADER &&
                header->content_layer() == docv1::CONTENT_LAYER_FURNITURE,
            "chandra: page header is furniture");
    const docv1::TextItemBase* heading = find_text(doc, "5.1 Hyper");
    require(heading != nullptr && heading->label() == docv1::DOC_ITEM_LABEL_SECTION_HEADER,
            "chandra: section header");
    bool italic = false;
    for (const auto& item : doc.texts()) {
        italic = italic || (base(item).text() == "2x speed up" && base(item).formatting().italic());
    }
    require(italic, "chandra: <i> run keeps italic formatting");
    require(doc.tables_size() == 1, "chandra: one table");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 2 && data.num_cols() == 2, "chandra: table shape");
    const docv1::TableCell* teds = cell_at(data, 0, 0);
    require(teds != nullptr && teds->col_span() == 2 && teds->column_header(),
            "chandra: colspan header cell");
}

// tests/data/json_dots/sources/dots_simple.json, shortened, and
// test_dots_parser.py's caption association.
void verify_dots_json() {
    const docv1::Document doc = map_or_fail(
        vlmv1::RESPONSE_FORMAT_DOTS_JSON,
        "[{\"bbox\": [211, 161, 1021, 243], \"category\": \"Title\", \"text\": \"# DocLayNet: "
        "A Dataset\"}, {\"bbox\": [104, 495, 228, 520], \"category\": \"Section-header\", "
        "\"text\": \"## ABSTRACT\"}, {\"bbox\": [636, 495, 1127, 1065], \"category\": "
        "\"Picture\"}, {\"bbox\": [635, 1083, 1130, 1131], \"category\": \"Caption\", \"text\": "
        "\"Figure 1: Four examples\"}, {\"bbox\": [103, 1274, 596, 1426], \"category\": "
        "\"List-item\", \"text\": \"- first point\"}]",
        page_context(1224, 1584));
    require(doc.texts_size() >= 4, "dots: title, heading, caption, list item");
    require(doc.texts(0).has_title() && base(doc.texts(0)).text() == "DocLayNet: A Dataset",
            "dots: title strips the markdown marker");
    require(doc.texts(0).title().base().orig() == "# DocLayNet: A Dataset",
            "dots: title keeps orig");
    const docv1::TextItemBase* abstract = find_text(doc, "ABSTRACT");
    require(abstract != nullptr && abstract->label() == docv1::DOC_ITEM_LABEL_SECTION_HEADER,
            "dots: section header");
    require(doc.pictures_size() == 1 && doc.pictures(0).captions_size() == 1,
            "dots: caption below the picture links to it");
    const docv1::TextItemBase* item = find_text(doc, "first point");
    require(item != nullptr && item->label() == docv1::DOC_ITEM_LABEL_LIST_ITEM,
            "dots: list item");
    docv1::Document empty;
    std::string error;
    require(vlm::mapping::map_response(vlmv1::RESPONSE_FORMAT_DOTS_JSON, "[]", page_context(),
                                       &empty, &error),
            "dots: an empty layout is an empty page: " + error);
}

// test_nemotron_parse_v2.py::test_extract_nemotron_parse_v2_multiline_regions
void verify_nemotron_parse_v2() {
    const docv1::Document doc = map_or_fail(
        vlmv1::RESPONSE_FORMAT_NEMOTRON_PARSE_V2,
        "<x_0.1><y_0.2># A title\non two lines<x_0.8><y_0.3><class_Title>\n\n"
        "<x_0.2><y_0.4><x_0.5><y_0.7><class_Picture>\n\n"
        "<x_0.1><y_0.8>Body text<x_0.9><y_0.9><class_Text>");
    require(doc.pictures_size() == 1, "nemotron v2: picture region");
    const docv1::TextItemBase* title = find_text(doc, "A title");
    require(title != nullptr && title->label() == docv1::DOC_ITEM_LABEL_TITLE,
            "nemotron v2: multi-line title region");
    require(find_text(doc, "Body text") != nullptr, "nemotron v2: text region");
}

std::string v35_block(const std::string& label, const std::string& body,
                      const std::string& x1 = "0.1", const std::string& y1 = "0.1",
                      const std::string& x2 = "0.9", const std::string& y2 = "0.2") {
    return "<x_" + x1 + "><y_" + y1 + ">" + body + "<x_" + x2 + "><y_" + y2 + "><class_" + label +
           ">\n\n";
}

// test_nemotron_parse_v3_5.py: artifact/duplicate drop, bbox scaling,
// and the LaTeX tabular table.
void verify_nemotron_parse_v3_5() {
    const vlm::mapping::PageContext page = page_context(600, 800);
    const docv1::Document blocks = map_or_fail(
        vlmv1::RESPONSE_FORMAT_NEMOTRON_PARSE_V3_5,
        "</s>" + v35_block("Text", "Hello<tbc> world") + v35_block("Text", "Hello<tbc> world") +
            "<x_0.5><y_0.5><class_Text>" + v35_block("Text", "next"),
        page);
    require(blocks.texts_size() == 2 && base(blocks.texts(0)).text() == "Hello world" &&
                base(blocks.texts(1)).text() == "next",
            "nemotron v3.5: artifacts, duplicates and malformed blocks dropped");

    const docv1::Document boxes = map_or_fail(
        vlmv1::RESPONSE_FORMAT_NEMOTRON_PARSE_V3_5,
        v35_block("Text", "inside", "0.1", "0.25", "0.5", "0.5") +
            v35_block("Mystery", "outside", "0.0", "0.9", "1.2", "1.1"),
        page);
    require(boxes.texts_size() == 2, "nemotron v3.5: two blocks");
    const docv1::BoundingBox& inside = base(boxes.texts(0)).prov(0).bbox();
    require(near(inside.l(), 60) && near(inside.t(), 200) && near(inside.r(), 300) &&
                near(inside.b(), 400),
            "nemotron v3.5: normalized box scales to the page");
    const docv1::BoundingBox& outside = base(boxes.texts(1)).prov(0).bbox();
    require(near(outside.l(), 0) && near(outside.t(), 720) && near(outside.r(), 600) &&
                near(outside.b(), 800),
            "nemotron v3.5: box clamps to the page");
    require(base(boxes.texts(1)).label() == docv1::DOC_ITEM_LABEL_TEXT,
            "nemotron v3.5: unknown class is text");

    const std::string table =
        "\\begin{tabular}{cccc}\n"
        "\\multirow{2}{*}{\\thead{Solution}} & "
        "\\multicolumn{3}{c}{\\thead{Intracellular solutions}}\\\\\n"
        "\\mbox{} & \\thead{K<sup>+</sup>} & \\thead{**Na**} & \\thead{Cl}\\\\\n"
        "Kin \\$ & 145 & \\mbox{} & 5 \\% \\textless{} 6\\\\\n"
        "\\end{tabular}";
    const docv1::Document tables =
        map_or_fail(vlmv1::RESPONSE_FORMAT_NEMOTRON_PARSE_V3_5, v35_block("Table", table), page);
    require(tables.tables_size() == 1, "nemotron v3.5: one table");
    const docv1::TableData& data = tables.tables(0).data();
    require(data.num_rows() == 3 && data.num_cols() == 4, "nemotron v3.5: table shape 3x4");
    const docv1::TableCell* solution = cell_at(data, 0, 0);
    require(solution != nullptr && solution->text() == "Solution" && solution->row_span() == 2,
            "nemotron v3.5: multirow cell");
    const docv1::TableCell* intra = cell_at(data, 0, 1);
    require(intra != nullptr && intra->text() == "Intracellular solutions" &&
                intra->col_span() == 3,
            "nemotron v3.5: multicolumn cell");
    const docv1::TableCell* kin = cell_at(data, 2, 0);
    const docv1::TableCell* pct = cell_at(data, 2, 3);
    require(kin != nullptr && kin->text() == "Kin $" && !kin->column_header(),
            "nemotron v3.5: escaped dollar, body row");
    require(pct != nullptr && pct->text() == "5 % < 6", "nemotron v3.5: escapes in cells");
}

// DocLang (docling-core's DocLangDocDeserializer vocabulary).
void verify_doclang() {
    const std::string answer =
        "Sure, here it is:\n<doclang><default_resolution width=\"512\" height=\"512\"/>"
        "<heading level=\"1\"><location value=\"0\"/><location value=\"0\"/>"
        "<location value=\"256\"/><location value=\"51\"/>The Title</heading>"
        "<heading level=\"2\">Intro</heading>"
        "<text><bold>All bold</bold></text>"
        "<text>Hello <italic>world</italic></text>"
        "<list class=\"ordered\"><ldiv><marker>1.</marker></ldiv>first"
        "<ldiv><marker>2.</marker></ldiv>second</list>"
        "<table><ched/>A<ched/>B<nl/><fcel/>1<lcel/><nl/></table>"
        "<group><picture><label value=\"bar_chart\"/></picture><text>Figure 1</text>"
        "<footnote>Source: survey</footnote></group>"
        "<code><label value=\"python\"/>print(1)</code>"
        "<page_footer><layer value=\"furniture\"/>Page 3</page_footer>"
        "</doclang> trailing";
    const docv1::Document doc = map_or_fail(vlmv1::RESPONSE_FORMAT_DOCLANG, answer);
    require(doc.texts(0).has_title() && base(doc.texts(0)).text() == "The Title",
            "doclang: heading level 1 is the title");
    const docv1::BoundingBox& box = base(doc.texts(0)).prov(0).bbox();
    require(near(box.r(), 500) && near(box.b(), 51.0 * 1000 / 512),
            "doclang: locations scale from the 512 grid");
    require(doc.texts(1).has_section_header() && doc.texts(1).section_header().level() == 1 &&
                base(doc.texts(1)).text() == "Intro",
            "doclang: heading level 2 is a level-1 section header");
    const docv1::TextItemBase* bold = find_text(doc, "All bold");
    require(bold != nullptr && bold->formatting().bold(), "doclang: lone bold run formats text");
    const docv1::TextItemBase* world = find_text(doc, "world");
    require(world != nullptr && world->formatting().italic(), "doclang: mixed text inline run");
    const docv1::TextItemBase* first = find_text(doc, "first");
    require(first != nullptr && first->label() == docv1::DOC_ITEM_LABEL_LIST_ITEM,
            "doclang: list item");
    bool ordered = false;
    for (const auto& item : doc.texts()) {
        if (item.has_list_item() && item.list_item().base().text() == "second") {
            ordered = item.list_item().enumerated() && item.list_item().marker() == "2.";
        }
    }
    require(ordered, "doclang: ordered list keeps enumerated and marker");
    require(doc.tables_size() == 1, "doclang: one table");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 2 && data.num_cols() == 2, "doclang: OTSL table shape");
    const docv1::TableCell* one = cell_at(data, 1, 0);
    require(one != nullptr && one->text() == "1" && one->col_span() == 2,
            "doclang: lcel extends the span");
    require(cell_at(data, 0, 0)->column_header(), "doclang: ched is a column header");
    require(doc.pictures_size() == 1, "doclang: float group picture");
    const docv1::PictureItem& pic = doc.pictures(0);
    require(pic.captions_size() == 1 && pic.footnotes_size() == 1,
            "doclang: float group satellites become caption and footnote");
    require(pic.meta().classification().predictions_size() == 1 &&
                pic.meta().classification().predictions(0).class_name() == "bar_chart",
            "doclang: picture label is its classification");
    bool code = false;
    for (const auto& item : doc.texts()) {
        code = code || (item.has_code() && item.code().text() == "print(1)" &&
                        item.code().code_language_raw() == "python");
    }
    require(code, "doclang: code with language label");
    const docv1::TextItemBase* footer = find_text(doc, "Page 3");
    require(footer != nullptr && footer->content_layer() == docv1::CONTENT_LAYER_FURNITURE,
            "doclang: layer head sets the content layer");

    docv1::Document none;
    std::string error;
    require(!vlm::mapping::map_response(vlmv1::RESPONSE_FORMAT_DOCLANG, "<doctag>x</doctag>",
                                        page_context(), &none, &error) &&
                error == "No <doclang> XML fragment found in VLM response.",
            "doclang: missing fragment fails with docling's message, got: " + error);
}

void verify_mineru2_unsupported() {
    docv1::Document doc;
    std::string error;
    require(!vlm::mapping::map_response(vlmv1::RESPONSE_FORMAT_MINERU2, "anything", page_context(),
                                        &doc, &error),
            "mineru2: not mapped");
    require(error.find("MinerU2") != std::string::npos, "mineru2: typed message, got: " + error);
}

}  // namespace

int main() {
    try {
        verify_deepseekocr_markdown();
        verify_unlimited_ocr_markdown();
        verify_chandra_html();
        verify_dots_json();
        verify_nemotron_parse_v2();
        verify_nemotron_parse_v3_5();
        verify_doclang();
        verify_mineru2_unsupported();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("docling-formats-test passed");
    return 0;
}
