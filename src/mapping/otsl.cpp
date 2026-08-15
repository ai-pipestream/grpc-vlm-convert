#include "otsl_grid.h"

#include <algorithm>
#include <cctype>
#include <vector>

#include "builder.h"

namespace vlm::mapping {

namespace {

// One OTSL token on the table grid: the tag name, then the text up to the
// next '<'. Filler tags (lcel/ucel/xcel) occupy grid columns but are never
// emitted as cells — they extend their anchor cell's span instead.
struct OtslToken {
    std::string tag;
    std::string text;
};

bool is_anchor(const std::string& tag) {
    return tag == "fcel" || tag == "ecel" || tag == "ched" || tag == "rhed" || tag == "srow";
}

bool is_horizontal_filler(const std::string& tag) {
    return tag == "lcel" || tag == "xcel";
}

bool is_vertical_filler(const std::string& tag) {
    return tag == "ucel" || tag == "xcel";
}

// The next token opener: a '<' followed by a letter. A '<' followed by
// anything else is literal cell text, not markup.
size_t find_tag(const std::string& body, size_t from) {
    size_t open = from;
    while ((open = body.find('<', open)) != std::string::npos) {
        if (open + 1 < body.size() &&
            std::isalpha(static_cast<unsigned char>(body[open + 1])) != 0) {
            return open;
        }
        open++;
    }
    return std::string::npos;
}

// Reads one token: the tag name, then the text up to the next tag.
// Returns the tag (empty at end of input) and advances `pos` past the text.
std::string next_token(const std::string& body, size_t* pos, std::string* text) {
    const size_t open = find_tag(body, *pos);
    if (open == std::string::npos) {
        *pos = body.size();
        return "";
    }
    const size_t close = body.find('>', open);
    if (close == std::string::npos) {
        *pos = body.size();
        return "";
    }
    std::string tag = body.substr(open + 1, close - open - 1);
    const size_t next = find_tag(body, close + 1);
    *text = trim(body.substr(close + 1, next == std::string::npos ? next : next - close - 1));
    *pos = next == std::string::npos ? body.size() : next;
    return tag;
}

// Splits the token stream into grid rows. <nl> ends a row; <srow> starts a
// new (section) row, so a row break also happens when one arrives mid-row.
std::vector<std::vector<OtslToken>> split_rows(const std::string& body) {
    std::vector<std::vector<OtslToken>> rows;
    std::vector<OtslToken> current;
    size_t pos = 0;
    while (pos < body.size()) {
        OtslToken token;
        token.tag = next_token(body, &pos, &token.text);
        if (token.tag.empty()) {
            break;
        }
        if (token.tag == "nl") {
            if (!current.empty()) {
                rows.push_back(std::move(current));
                current.clear();
            }
            continue;
        }
        if (token.tag == "srow" && !current.empty()) {
            rows.push_back(std::move(current));
            current.clear();
        }
        if (is_anchor(token.tag) || is_horizontal_filler(token.tag) ||
            is_vertical_filler(token.tag)) {
            current.push_back(std::move(token));
        }
        // Anything else (the enclosing otsl tags, stray markup) is ignored.
    }
    if (!current.empty()) {
        rows.push_back(std::move(current));
    }
    return rows;
}

}  // namespace

bool parse_otsl_grid(const std::string& body, docv1::TableData* data) {
    const std::vector<std::vector<OtslToken>> rows = split_rows(body);
    if (rows.empty()) {
        return false;
    }

    const int32_t num_rows = static_cast<int32_t>(rows.size());
    int32_t num_cols = 0;
    for (const auto& row : rows) {
        num_cols = std::max<int32_t>(num_cols, static_cast<int32_t>(row.size()));
    }
    data->set_num_rows(num_rows);
    data->set_num_cols(num_cols);

    // Anchor cells carry the span; filler cells extend it and vanish —
    // docling_core/types/doc/utils.py otsl_parse_texts, with the header
    // flags docling drops kept on the cell (our deliberate advantage).
    for (size_t r = 0; r < rows.size(); r++) {
        for (size_t c = 0; c < rows[r].size(); c++) {
            const OtslToken& token = rows[r][c];
            if (!is_anchor(token.tag)) {
                continue;
            }
            int32_t col_span = 1;
            for (size_t cc = c + 1; cc < rows[r].size() && is_horizontal_filler(rows[r][cc].tag);
                 cc++) {
                col_span++;
            }
            int32_t row_span = 1;
            for (size_t rr = r + 1;
                 rr < rows.size() && c < rows[rr].size() && is_vertical_filler(rows[rr][c].tag);
                 rr++) {
                row_span++;
            }
            docv1::TableCell cell;
            // ecel is an empty cell: no text part follows the tag.
            cell.set_text(token.tag == "ecel" ? "" : token.text);
            cell.set_column_header(token.tag == "ched");
            cell.set_row_header(token.tag == "rhed");
            cell.set_row_section(token.tag == "srow");
            cell.set_row_span(row_span);
            cell.set_col_span(col_span);
            cell.set_start_row_offset_idx(static_cast<int32_t>(r));
            cell.set_end_row_offset_idx(static_cast<int32_t>(r) + row_span);
            cell.set_start_col_offset_idx(static_cast<int32_t>(c));
            cell.set_end_col_offset_idx(static_cast<int32_t>(c) + col_span);
            *data->add_table_cells() = cell;
        }
    }

    // The grid is the full num_rows × num_cols matrix with each anchor cell
    // stamped over every position its span covers (docling's TableData.grid
    // computed field); uncovered positions stay empty 1x1 cells.
    for (int32_t r = 0; r < num_rows; r++) {
        docv1::TableRow* grid_row = data->add_grid();
        for (int32_t c = 0; c < num_cols; c++) {
            docv1::TableCell* empty = grid_row->add_cells();
            empty->set_row_span(1);
            empty->set_col_span(1);
            empty->set_start_row_offset_idx(r);
            empty->set_end_row_offset_idx(r + 1);
            empty->set_start_col_offset_idx(c);
            empty->set_end_col_offset_idx(c + 1);
        }
    }
    for (const docv1::TableCell& cell : data->table_cells()) {
        for (int32_t r = cell.start_row_offset_idx();
             r < std::min(cell.end_row_offset_idx(), num_rows); r++) {
            for (int32_t c = cell.start_col_offset_idx();
                 c < std::min(cell.end_col_offset_idx(), num_cols); c++) {
                *data->mutable_grid(r)->mutable_cells(c) = cell;
            }
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
