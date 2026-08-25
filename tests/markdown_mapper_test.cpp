// Unit tests for the markdown mapper: heading labels and levels, lists,
// fenced code, pipe tables — and the hard rule that markdown responses
// carry page-level provenance only, never invented word boxes.

#include "fixture.h"
#include "mapping/mapper.h"

namespace docv1 = ai::pipestream::document::v1;
namespace vlmv1 = ai::pipestream::vlm::v1;

namespace {

vlm::mapping::PageContext page_context() {
    vlm::mapping::PageContext page;
    page.page_no = 2;
    page.width = 612;
    page.height = 792;
    page.source.set_collector("vlm-convert");
    page.source.set_model("test-model");
    return page;
}

const docv1::TextItemBase* base_of(const docv1::BaseTextItem& item) {
    switch (item.item_case()) {
        case docv1::BaseTextItem::kSectionHeader:
            return &item.section_header().base();
        case docv1::BaseTextItem::kListItem:
            return &item.list_item().base();
        case docv1::BaseTextItem::kText:
            return &item.text().base();
        default:
            return nullptr;
    }
}

void verify_headings_lists_code() {
    const std::string text =
        "# Results\n"
        "\n"
        "The page summarizes the quarter.\n"
        "\n"
        "## Details\n"
        "\n"
        "- first point\n"
        "- second point\n"
        "1. step one\n"
        "\n"
        "```python\n"
        "print('hi')\n"
        "```\n";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_markdown(text, page_context(), &doc, &error),
            "markdown page maps: " + error);
    require(doc.texts_size() == 7, "two headings + paragraph + three list items + code");

    require(doc.texts(0).has_section_header() && doc.texts(0).section_header().level() == 1,
            "h1 is a level-1 section header");
    require(doc.texts(0).section_header().base().text() == "Results", "h1 text");
    require(doc.texts(2).has_section_header() && doc.texts(2).section_header().level() == 2,
            "h2 is a level-2 section header");
    require(doc.texts(1).has_text() &&
                doc.texts(1).text().base().label() == docv1::DOC_ITEM_LABEL_PARAGRAPH,
            "paragraph label");
    require(doc.texts(3).has_list_item() && !doc.texts(3).list_item().enumerated(),
            "bulleted list item");
    require(doc.texts(5).has_list_item() && doc.texts(5).list_item().enumerated() &&
                doc.texts(5).list_item().marker() == "1.",
            "enumerated list item with marker");
    require(doc.texts(6).has_code(), "fenced code item");
    require(doc.texts(6).code().text() == "print('hi')", "code body");
    require(doc.texts(6).code().code_language_raw() == "python", "code language raw");

    // No fake word boxes: every text item carries exactly the full page.
    for (const docv1::BaseTextItem& item : doc.texts()) {
        const docv1::TextItemBase* base = base_of(item);
        if (base == nullptr) {
            continue;  // code inlines the base fields
        }
        require(base->prov_size() == 1, "one provenance entry per item");
        const docv1::BoundingBox& box = base->prov(0).bbox();
        require(box.l() == 0 && box.t() == 0 && box.r() == 612 && box.b() == 792,
                "markdown items get the full-page box, never invented boxes");
        require(box.coord_origin() == docv1::COORD_ORIGIN_TOPLEFT, "TOPLEFT origin");
        require(base->prov(0).page_no() == 2, "page provenance");
    }
}

void verify_pipe_table() {
    const std::string text =
        "| Name | Qty |\n"
        "| --- | ---: |\n"
        "| bolts | 12 |\n"
        "| nuts | 30 |\n";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_markdown(text, page_context(), &doc, &error),
            "pipe table maps: " + error);
    require(doc.tables_size() == 1, "one table item");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 3 && data.num_cols() == 2, "separator row is not a data row");
    require(data.grid(0).cells(0).text() == "Name", "header row text");
    require(data.grid(0).cells(0).column_header(), "first row is column headers");
    require(data.grid(2).cells(1).text() == "30", "last cell text");
    require(!data.grid(2).cells(1).column_header(), "data rows are not headers");
}

void verify_mapping_failure() {
    docv1::Document doc;
    std::string error;
    require(!vlm::mapping::map_markdown("   \n  \n", page_context(), &doc, &error),
            "whitespace-only markdown is a mapping failure");
    require(!error.empty(), "failure carries a reason");
}

void verify_plaintext_and_html() {
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_plaintext("one page of words", page_context(), &doc, &error),
            "plaintext maps: " + error);
    require(doc.texts_size() == 1 && doc.texts(0).text().base().text() == "one page of words",
            "plaintext is exactly one TextItem per page");

    docv1::Document html_doc;
    require(vlm::mapping::map_html("<h1>Title</h1><p>Body.</p><ul><li>point</li></ul>",
                                   page_context(), &html_doc, &error),
            "html maps: " + error);
    require(html_doc.texts_size() == 3, "h1 + p + li");
    require(html_doc.texts(0).has_section_header(), "h1 is a section header");
    require(html_doc.texts(2).has_list_item(), "li is a list item");
}

// An HTML table used to ship as a TableItem with no TableData at all: an
// empty box where a table was. The rows and cells the model wrote are the
// table.
void verify_html_table() {
    const std::string text =
        "<p>Inventory:</p>\n"
        "<table>\n"
        "  <thead><tr><th>Name</th><th>Qty</th></tr></thead>\n"
        "  <tbody>\n"
        "    <tr><td>bolts</td><td>12</td></tr>\n"
        "    <tr><td>nuts</td><td>30</td></tr>\n"
        "  </tbody>\n"
        "</table>\n";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_html(text, page_context(), &doc, &error),
            "html table maps: " + error);
    require(doc.tables_size() == 1, "one table item");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 3 && data.num_cols() == 2, "header row plus two data rows");
    require(data.grid(0).cells(0).text() == "Name" && data.grid(0).cells(1).text() == "Qty",
            "header cell text");
    require(data.grid(0).cells(0).column_header(), "an all-th row is a header row");
    require(!data.grid(1).cells(0).column_header(), "td rows are not headers");
    require(data.grid(2).cells(1).text() == "30", "last cell text");
    require(data.table_cells_size() == 6, "every source cell is in table_cells");
    require(doc.tables(0).prov(0).bbox().r() == 612,
            "html tables keep the full-page box, never invented boxes");
}

}  // namespace

int main() {
    try {
        verify_headings_lists_code();
        verify_pipe_table();
        verify_mapping_failure();
        verify_plaintext_and_html();
        verify_html_table();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("markdown-mapper-test passed");
    return 0;
}
