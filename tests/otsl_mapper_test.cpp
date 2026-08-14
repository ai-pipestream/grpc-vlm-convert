// Unit tests for the standalone OTSL mapper: the grid shape, the header
// flags docling drops (we keep them), and docling's span resolution —
// <lcel>/<ucel>/<xcel> extend their anchor cell and vanish, <srow> starts
// a section row, and the grid stamps anchors over their whole span.

#include "fixture.h"
#include "mapping/mapper.h"

namespace docv1 = ai::pipestream::document::v1;

namespace {

vlm::mapping::PageContext page_context() {
    vlm::mapping::PageContext page;
    page.page_no = 1;
    page.width = 1000;
    page.height = 1000;
    page.source.set_collector("vlm-convert");
    return page;
}

docv1::Document map_or_fail(const std::string& text) {
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_otsl(text, page_context(), &doc, &error),
            "otsl maps: " + error);
    require(doc.tables_size() == 1, "one table item");
    return doc;
}

void verify_plain_grid_and_headers() {
    docv1::Document doc =
        map_or_fail("<otsl><ched>Name<ched>Qty<nl><rhed>bolts<fcel>12<nl></otsl>");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 2 && data.num_cols() == 2, "grid shape");
    require(data.table_cells_size() == 4, "four cells, no fillers");
    require(data.table_cells(0).column_header() && !data.table_cells(0).row_header(),
            "ched marks a column header");
    require(data.table_cells(2).row_header() && !data.table_cells(2).column_header(),
            "rhed marks a row header");
    require(data.table_cells(0).row_span() == 1 && data.table_cells(0).col_span() == 1,
            "plain cells are 1x1");
    require(data.grid(0).cells(0).text() == "Name" && data.grid(1).cells(1).text() == "12",
            "grid mirrors the cells");
}

void verify_horizontal_span() {
    // <fcel>A<lcel> — A spans two columns; the filler is not emitted.
    docv1::Document doc = map_or_fail("<otsl><fcel>A<lcel><nl><fcel>B<fcel>C<nl></otsl>");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 2 && data.num_cols() == 2, "span grid shape");
    require(data.table_cells_size() == 3, "lcel filler suppressed");
    const docv1::TableCell& a = data.table_cells(0);
    require(a.text() == "A" && a.col_span() == 2 && a.row_span() == 1, "col span grows left-to-right");
    require(a.start_col_offset_idx() == 0 && a.end_col_offset_idx() == 2, "col offsets");
    require(a.start_row_offset_idx() == 0 && a.end_row_offset_idx() == 1, "row offsets");
    require(data.grid(0).cells(0).text() == "A" && data.grid(0).cells(1).text() == "A",
            "anchor stamped over both covered grid positions");
}

void verify_vertical_and_2d_span() {
    // A spans 2x2: <lcel> right of A, <ucel> below A, <xcel> below the filler.
    docv1::Document doc = map_or_fail(
        "<otsl><fcel>A<lcel><nl><ucel><xcel><nl><fcel>B<fcel>C<nl></otsl>");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 3 && data.num_cols() == 2, "2d grid shape");
    require(data.table_cells_size() == 3, "ucel/xcel fillers suppressed");
    const docv1::TableCell& a = data.table_cells(0);
    require(a.row_span() == 2 && a.col_span() == 2, "xcel grows the span in both dimensions");
    require(a.end_row_offset_idx() == 2 && a.end_col_offset_idx() == 2, "2d end offsets");
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            require(data.grid(r).cells(c).text() == "A", "anchor covers the whole 2x2 region");
        }
    }
    require(data.grid(2).cells(0).text() == "B" && data.grid(2).cells(1).text() == "C",
            "cells below the span are untouched");
}

void verify_srow_section_row() {
    // <srow> starts a new row even without a preceding <nl> and marks a
    // section row; the flag survives (docling drops it, we keep it).
    docv1::Document doc =
        map_or_fail("<otsl><fcel>top<srow>Section<nl><fcel>x<fcel>y<nl></otsl>");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 3, "srow breaks the row");
    require(data.table_cells_size() == 4, "srow is a cell, not a filler");
    const docv1::TableCell& section = data.table_cells(1);
    require(section.text() == "Section" && section.row_section(), "srow marks a section row");
    require(section.start_row_offset_idx() == 1 && section.end_row_offset_idx() == 2,
            "section row offsets");
    require(!data.table_cells(0).row_section() && !data.table_cells(2).row_section(),
            "plain cells are not section rows");
}

void verify_failures() {
    docv1::Document doc;
    std::string error;
    require(!vlm::mapping::map_otsl("no table here", page_context(), &doc, &error),
            "no rows is a mapping failure");
    require(!error.empty(), "failure carries a reason");
}

}  // namespace

int main() {
    try {
        verify_plain_grid_and_headers();
        verify_horizontal_span();
        verify_vertical_and_2d_span();
        verify_srow_section_row();
        verify_failures();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "otsl-mapper-test passed\n";
    return 0;
}
