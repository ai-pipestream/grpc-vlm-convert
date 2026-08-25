#include "builder.h"
#include "mapper.h"

#include <cctype>
#include <vector>

namespace vlm::mapping {

namespace {

// A pipe-table separator row: | --- | :---: | --- |
bool is_separator_row(const std::string& line) {
    bool saw_dash = false;
    for (char c : line) {
        if (c == '-') {
            saw_dash = true;
        } else if (c != '|' && c != ':' && c != ' ' && c != '\t') {
            return false;
        }
    }
    return saw_dash;
}

std::vector<std::string> split_pipe_row(const std::string& line) {
    std::string row = trim(line);
    if (row.starts_with('|')) {
        row.erase(0, 1);
    }
    // Split on unescaped pipes only: "\|" is cell content (a literal
    // pipe), never a column boundary.
    std::vector<std::string> cells;
    std::string cell;
    size_t last_boundary = std::string::npos;
    for (size_t i = 0; i < row.size(); i++) {
        if (row[i] == '\\' && i + 1 < row.size() && row[i + 1] == '|') {
            cell += '|';
            i++;
        } else if (row[i] == '|') {
            cells.push_back(trim(cell));
            cell.clear();
            last_boundary = i;
        } else {
            cell += row[i];
        }
    }
    cells.push_back(trim(cell));
    // A trailing unescaped pipe closes the last cell rather than opening
    // an empty one.
    if (!row.empty() && last_boundary == row.size() - 1) {
        cells.pop_back();
    }
    return cells;
}

void emit_pipe_table(const std::vector<std::string>& lines, const PageContext& page,
                     docv1::Document* doc) {
    docv1::TableItem* table = add_table(doc, page, page_prov(page));
    const size_t header_rows = lines.size() > 1 && is_separator_row(lines[1]) ? 1 : 0;
    std::vector<std::vector<std::string>> rows;
    for (size_t i = 0; i < lines.size(); i++) {
        if (header_rows == 1 && i == 1) {
            continue;  // the separator row itself
        }
        rows.push_back(split_pipe_row(lines[i]));
    }
    fill_table_data(table->mutable_data(), rows, header_rows);
}

}  // namespace

bool map_markdown(const std::string& text, const PageContext& page, docv1::Document* out,
                  std::string* error) {
    if (trim(text).empty()) {
        *error = "empty markdown response";
        return false;
    }

    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        size_t nl = text.find('\n', start);
        if (nl == std::string::npos) {
            nl = text.size();
        }
        std::string line = text.substr(start, nl - start);
        if (line.ends_with('\r')) {
            line.pop_back();
        }
        lines.push_back(std::move(line));
        start = nl + 1;
    }

    size_t items = 0;
    std::string paragraph;
    auto flush_paragraph = [&] {
        std::string body = trim(paragraph);
        paragraph.clear();
        if (!body.empty()) {
            add_text(out, page, docv1::DOC_ITEM_LABEL_PARAGRAPH, page_prov(page), body);
            items++;
        }
    };

    for (size_t i = 0; i < lines.size(); i++) {
        const std::string& line = lines[i];
        std::string stripped = trim(line);

        // Fenced code block.
        if (stripped.starts_with("```")) {
            flush_paragraph();
            std::string language = trim(stripped.substr(3));
            std::string code;
            while (++i < lines.size() && !trim(lines[i]).starts_with("```")) {
                code += lines[i];
                code += '\n';
            }
            add_code(out, page, page_prov(page),
                     code.empty() ? code : code.substr(0, code.size() - 1), language);
            items++;
            continue;
        }

        // ATX heading.
        size_t hashes = 0;
        while (hashes < stripped.size() && stripped[hashes] == '#') {
            hashes++;
        }
        if (hashes >= 1 && hashes <= 6 && hashes < stripped.size() &&
            std::isspace(static_cast<unsigned char>(stripped[hashes]))) {
            flush_paragraph();
            std::string heading = trim(stripped.substr(hashes));
            // ATX optional closing sequence: a trailing run of '#'s
            // preceded by whitespace (or the whole content) comes off.
            const size_t last_non_hash = heading.find_last_not_of('#');
            if (last_non_hash == std::string::npos) {
                heading.clear();
            } else if (last_non_hash + 1 < heading.size() &&
                       (heading[last_non_hash] == ' ' || heading[last_non_hash] == '\t')) {
                heading = trim(heading.substr(0, last_non_hash));
            }
            add_section_header(out, page, static_cast<int>(hashes), page_prov(page), heading);
            items++;
            continue;
        }

        // Pipe table: a run of lines starting with '|'.
        if (stripped.starts_with('|')) {
            flush_paragraph();
            std::vector<std::string> table_lines;
            while (i < lines.size() && trim(lines[i]).starts_with('|')) {
                table_lines.push_back(lines[i]);
                i++;
            }
            i--;
            emit_pipe_table(table_lines, page, out);
            items++;
            continue;
        }

        // List item, bulleted or enumerated.
        bool bulleted = stripped.size() > 2 &&
                        (stripped[0] == '-' || stripped[0] == '*' || stripped[0] == '+') &&
            std::isspace(static_cast<unsigned char>(stripped[1]));
        size_t number_end = 0;
        while (number_end < stripped.size() &&
               std::isdigit(static_cast<unsigned char>(stripped[number_end]))) {
            number_end++;
        }
        bool enumerated = number_end > 0 && number_end + 1 < stripped.size() &&
                          (stripped[number_end] == '.' || stripped[number_end] == ')') &&
            std::isspace(static_cast<unsigned char>(stripped[number_end + 1]));
        if (bulleted || enumerated) {
            flush_paragraph();
            const std::string marker = bulleted ? stripped.substr(0, 1)
                                                : stripped.substr(0, number_end + 1);
            const std::string body =
                trim(stripped.substr(bulleted ? 1 : number_end + 1));
            add_list_item(out, page, page_prov(page), body, enumerated, marker);
            items++;
            continue;
        }

        if (stripped.empty()) {
            flush_paragraph();
        } else {
            paragraph += stripped;
            paragraph += '\n';
        }
    }
    flush_paragraph();

    if (items == 0) {
        *error = "markdown response produced no document items";
        return false;
    }
    finalize_document(out, page);
    return true;
}

}  // namespace vlm::mapping
