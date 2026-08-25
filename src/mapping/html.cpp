#include "builder.h"
#include "mapper.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <vector>

namespace vlm::mapping {

namespace {

std::string strip_tags(const std::string& html) {
    static const std::regex kTag("<[^>]*>");
    std::string text = std::regex_replace(html, kTag, "");
    // The handful of entities VLM output actually carries.
    static const std::pair<const char*, const char*> kEntities[] = {
        {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"},
        {"&quot;", "\""}, {"&#39;", "'"}, {"&nbsp;", " "},
    };
    for (const auto& [entity, replacement] : kEntities) {
        size_t pos = 0;
        const std::string needle(entity);
        while ((pos = text.find(needle, pos)) != std::string::npos) {
            text.replace(pos, needle.size(), replacement);
            pos += 1;
        }
    }
    return trim(text);
}

// Rows and cells of a table body, in source order. A row whose cells are
// all <th> is a header row; the leading run of those becomes the table's
// column headers (*header_rows). Cell text is stripped like any other
// block body. Cells outside a <tr> are read as one row, so the shapes
// models actually emit still produce a grid.
std::vector<std::vector<std::string>> parse_table_rows(const std::string& table_html,
                                                       size_t* header_rows) {
    static const std::regex kRow("<tr[^>]*>([\\s\\S]*?)</tr>", std::regex::icase);
    static const std::regex kCell("<(t[hd])[^>]*>([\\s\\S]*?)</\\1>", std::regex::icase);
    auto cells_of = [](const std::string& row_html, bool* all_headers) {
        std::vector<std::string> cells;
        *all_headers = false;
        bool headers_so_far = true;
        for (auto it = std::sregex_iterator(row_html.begin(), row_html.end(), kCell);
             it != std::sregex_iterator(); ++it) {
            std::string tag = (*it)[1].str();
            std::ranges::transform(tag, tag.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
            headers_so_far = headers_so_far && tag == "th";
            cells.push_back(strip_tags((*it)[2].str()));
        }
        *all_headers = !cells.empty() && headers_so_far;
        return cells;
    };

    std::vector<std::vector<std::string>> rows;
    std::vector<bool> header_flags;
    for (auto it = std::sregex_iterator(table_html.begin(), table_html.end(), kRow);
         it != std::sregex_iterator(); ++it) {
        bool all_headers = false;
        std::vector<std::string> cells = cells_of((*it)[1].str(), &all_headers);
        if (cells.empty()) {
            continue;
        }
        rows.push_back(std::move(cells));
        header_flags.push_back(all_headers);
    }
    if (rows.empty()) {
        bool all_headers = false;
        std::vector<std::string> cells = cells_of(table_html, &all_headers);
        if (!cells.empty()) {
            rows.push_back(std::move(cells));
            header_flags.push_back(all_headers);
        }
    }
    *header_rows = 0;
    while (*header_rows < header_flags.size() && header_flags[*header_rows]) {
        (*header_rows)++;
    }
    return rows;
}

}  // namespace

bool map_html(const std::string& text, const PageContext& page, docv1::Document* out,
              std::string* error) {
    // Block-level constructs only — this is a snippet mapper, not an HTML
    // parser; anything richer belongs to the HTML collector upstream.
    // [\s\S] rather than . so a block that spans lines still matches:
    // model output wraps tables and paragraphs across newlines.
    static const std::regex kBlock(
        "<(h[1-6]|p|li|pre|code|table)[^>]*>([\\s\\S]*?)</\\1>",
        std::regex::icase);
    size_t items = 0;
    auto begin = std::sregex_iterator(text.begin(), text.end(), kBlock);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        std::string tag = (*it)[1].str();
        std::ranges::transform(tag, tag.begin(),
                               [](unsigned char c) { return std::tolower(c); });
        const std::string body = strip_tags((*it)[2].str());
        if (body.empty() && tag != "table") {
            continue;
        }
        if (tag[0] == 'h') {
            int level = tag[1] - '0';
            add_section_header(out, page, level, page_prov(page), body);
        } else if (tag == "li") {
            add_list_item(out, page, page_prov(page), body, /*enumerated=*/false, "");
        } else if (tag == "pre" || tag == "code") {
            add_code(out, page, page_prov(page), body, "");
        } else if (tag == "table") {
            // The rows and cells the model wrote become real TableData;
            // a TableItem with no data at all is an empty box where a
            // table was. A table whose markup holds no cells keeps its
            // text as a single cell rather than losing it.
            docv1::TableItem* table = add_table(out, page, page_prov(page));
            size_t header_rows = 0;
            std::vector<std::vector<std::string>> rows =
                parse_table_rows((*it)[2].str(), &header_rows);
            if (rows.empty() && !body.empty()) {
                rows.push_back({body});
            }
            fill_table_data(table->mutable_data(), rows, header_rows);
        } else {
            add_text(out, page, docv1::DOC_ITEM_LABEL_PARAGRAPH, page_prov(page), body);
        }
        items++;
    }
    if (items == 0) {
        *error = "HTML response held no block-level elements";
        return false;
    }
    finalize_document(out, page);
    return true;
}

}  // namespace vlm::mapping
