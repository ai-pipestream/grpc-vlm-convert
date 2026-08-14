#include "otsl_grid.h"

#include <vector>

#include "builder.h"

namespace vlm::mapping {

namespace {

struct OtslCell {
    std::string text;
    bool column_header = false;
    bool row_header = false;
};

// One cell token: the tag name, then the text up to the next '<'.
// Returns the tag (empty at end of input) and advances `pos` past the text.
std::string next_cell_token(const std::string& body, size_t* pos, std::string* text) {
    size_t open = body.find('<', *pos);
    if (open == std::string::npos) {
        *pos = body.size();
        return "";
    }
    size_t close = body.find('>', open);
    if (close == std::string::npos) {
        *pos = body.size();
        return "";
    }
    std::string tag = body.substr(open + 1, close - open - 1);
    size_t next = body.find('<', close);
    *text = trim(body.substr(close + 1, next == std::string::npos ? next : next - close - 1));
    *pos = next == std::string::npos ? body.size() : next;
    return tag;
}

}  // namespace

bool parse_otsl_grid(const std::string& body, docv1::TableData* data) {
    std::vector<std::vector<OtslCell>> rows;
    std::vector<OtslCell> current;
    size_t pos = 0;
    while (pos < body.size()) {
        std::string text;
        std::string tag = next_cell_token(body, &pos, &text);
        if (tag.empty()) {
            break;
        }
        if (tag == "nl") {
            if (!current.empty()) {
                rows.push_back(std::move(current));
                current.clear();
            }
            continue;
        }
        if (tag == "fcel" || tag == "ecel" || tag == "ched" || tag == "rhed") {
            OtslCell cell;
            cell.text = text;
            cell.column_header = tag == "ched";
            cell.row_header = tag == "rhed";
            current.push_back(std::move(cell));
            continue;
        }
        // srow/lcel/ucel/xcel spans and the enclosing otsl tags themselves:
        // stored verbatim in v1 — no second OCR pass to snap them.
        if (tag == "srow" || tag == "lcel" || tag == "ucel" || tag == "xcel") {
            OtslCell cell;
            cell.text = text;
            current.push_back(std::move(cell));
        }
    }
    if (!current.empty()) {
        rows.push_back(std::move(current));
    }
    if (rows.empty()) {
        return false;
    }

    int32_t num_cols = 0;
    for (const auto& row : rows) {
        num_cols = std::max<int32_t>(num_cols, static_cast<int32_t>(row.size()));
    }
    data->set_num_rows(static_cast<int32_t>(rows.size()));
    data->set_num_cols(num_cols);
    for (size_t r = 0; r < rows.size(); r++) {
        docv1::TableRow* grid_row = data->add_grid();
        for (size_t c = 0; c < rows[r].size(); c++) {
            docv1::TableCell cell;
            cell.set_text(rows[r][c].text);
            cell.set_column_header(rows[r][c].column_header);
            cell.set_row_header(rows[r][c].row_header);
            cell.set_row_span(1);
            cell.set_col_span(1);
            cell.set_start_row_offset_idx(static_cast<int32_t>(r));
            cell.set_end_row_offset_idx(static_cast<int32_t>(r + 1));
            cell.set_start_col_offset_idx(static_cast<int32_t>(c));
            cell.set_end_col_offset_idx(static_cast<int32_t>(c + 1));
            *grid_row->add_cells() = cell;
            *data->add_table_cells() = cell;
        }
    }
    return true;
}

bool map_otsl(const std::string& text, const PageContext& page, docv1::Document* out,
              std::string* error) {
    // The OTSL payload may arrive wrapped in <otsl>...</otsl> or bare.
    std::string body = text;
    size_t open = text.find("<otsl>");
    if (open != std::string::npos) {
        size_t close = text.find("</otsl>", open);
        body = close == std::string::npos
                   ? text.substr(open + 6)
                   : text.substr(open + 6, close - open - 6);
    }
    docv1::TableItem* table = add_table(out, page, page_prov(page));
    if (!parse_otsl_grid(body, table->mutable_data())) {
        *error = "OTSL response held no complete table row";
        return false;
    }
    finalize_document(out, page);
    return true;
}

}  // namespace vlm::mapping
