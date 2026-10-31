// NVIDIA Nemotron Parse native output (docling nemotron_parse_utils.py and
// nemotron_parse_v3_5_utils.py, v2.142.1). Blocks are
//   <x_X1><y_Y1>BODY<x_X2><y_Y2><class_CLASS>
// with coordinates normalized to [0, 1]. Version 2.0 draws on a padded
// 1664x2048 canvas (the padding is undone); 3.5 normalizes to the page
// content. Bodies are Markdown text, LaTeX tabular tables or, for charts,
// Markdown data tables.

#include <algorithm>
#include <cctype>
#include <charconv>
#include <memory>
#include <optional>
#include <set>

#include "layout.h"

namespace vlm::mapping {

namespace {

using Ref = LayoutWriter::Ref;
using Box = std::optional<docv1::BoundingBox>;

constexpr size_t kMaxLatexColumns = 1000;  // chandra _MAX_TABLE_GRID_CELLS

// ---- shared scanning ------------------------------------------------------

// `<x_N><y_N>` at `i` (N is \d+(?:\.\d+)?); the end offset and values.
std::optional<size_t> coord_token(std::string_view text, size_t i, double* x, double* y) {
    auto number = [&](size_t at, double* value) -> std::optional<size_t> {
        size_t j = at;
        while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j])) != 0) {
            j++;
        }
        if (j == at) {
            return std::nullopt;
        }
        if (j + 1 < text.size() && text[j] == '.' &&
            std::isdigit(static_cast<unsigned char>(text[j + 1])) != 0) {
            j++;
            while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j])) != 0) {
                j++;
            }
        }
        std::from_chars(text.data() + at, text.data() + j, *value);
        return j;
    };
    if (text.compare(i, 3, "<x_") != 0) {
        return std::nullopt;
    }
    const auto x_end = number(i + 3, x);
    if (!x_end || text.compare(*x_end, 4, "><y_") != 0) {
        return std::nullopt;
    }
    const auto y_end = number(*x_end + 4, y);
    if (!y_end || *y_end >= text.size() || text[*y_end] != '>') {
        return std::nullopt;
    }
    return *y_end + 1;
}

// `<class_NAME>` at `i` (NAME is [^>]+).
std::optional<size_t> class_token(std::string_view text, size_t i, std::string* name) {
    if (text.compare(i, 7, "<class_") != 0) {
        return std::nullopt;
    }
    const size_t close = text.find('>', i + 7);
    if (close == std::string_view::npos || close == i + 7) {
        return std::nullopt;
    }
    *name = std::string(text.substr(i + 7, close - i - 7));
    return close + 1;
}

std::string remove_artifacts(std::string text) {
    for (const std::string_view token : {std::string_view("<tbc>"), std::string_view("\\<|unk|\\>"),
                                         std::string_view("\\unknown")}) {
        for (size_t at = text.find(token); at != std::string::npos; at = text.find(token, at)) {
            text.erase(at, token.size());
        }
    }
    return text;
}

// ---- LaTeX tabular (nemotron_parse_v3_5_utils latex_tabular_to_html) -----

std::optional<std::pair<std::string, size_t>> read_group(const std::string& text, size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        pos++;
    }
    if (pos >= text.size() || text[pos] != '{') {
        return std::nullopt;
    }
    int depth = 0;
    for (size_t i = pos; i < text.size();) {
        const char c = text[i];
        if (c == '\\') {
            i += 2;
            continue;
        }
        if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0) {
                return std::pair{text.substr(pos + 1, i - pos - 1), i + 1};
            }
        }
        i++;
    }
    return std::nullopt;
}

std::vector<std::string> split_top_level(const std::string& text, std::string_view separator) {
    std::vector<std::string> parts;
    int depth = 0;
    size_t start = 0;
    size_t i = 0;
    while (i < text.size()) {
        if (depth == 0 && text.compare(i, separator.size(), separator) == 0) {
            parts.push_back(text.substr(start, i - start));
            i += separator.size();
            start = i;
            continue;
        }
        const char c = text[i];
        if (c == '\\') {
            i += 2;
            continue;
        }
        if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth = std::max(0, depth - 1);
        }
        i++;
    }
    parts.push_back(start <= text.size() ? text.substr(start) : std::string());
    return parts;
}

std::optional<std::vector<std::string>> unwrap_command(const std::string& text,
                                                       const std::string& command,
                                                       size_t arg_count) {
    const std::string stripped = trim(text);
    const std::string prefix = "\\" + command;
    if (!stripped.starts_with(prefix) ||
        (stripped.size() > prefix.size() &&
         std::isalpha(static_cast<unsigned char>(stripped[prefix.size()])) != 0)) {
        return std::nullopt;
    }
    std::vector<std::string> args;
    size_t pos = prefix.size();
    for (size_t n = 0; n < arg_count; n++) {
        auto group = read_group(stripped, pos);
        if (!group) {
            return std::nullopt;
        }
        args.push_back(std::move(group->first));
        pos = group->second;
    }
    std::string rest = trim(stripped.substr(pos));
    size_t braces = 0;
    while (braces < rest.size() && rest[braces] == '}') {
        braces++;
    }
    rest = trim(rest.substr(braces));
    if (!rest.empty()) {
        args.back() += " " + rest;
    }
    return args;
}

size_t span_value(const std::string& value) {
    std::string digits = trim(value);
    const size_t first = digits.find_first_not_of('0');
    digits = first == std::string::npos ? "" : digits.substr(first);
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) != 0;
        })) {
        return 1;
    }
    if (digits.size() > 4) {
        return kMaxLatexColumns + 1;
    }
    return std::min<size_t>(std::stoul(digits), kMaxLatexColumns + 1);
}

void html_escape_into(std::string_view text, std::string* out) {
    for (char c : text) {
        if (c == '&') {
            *out += "&amp;";
        } else if (c == '<') {
            *out += "&lt;";
        } else if (c == '>') {
            *out += "&gt;";
        } else {
            out->push_back(c);
        }
    }
}

// _escape_cell_markup: escape everything but a few LaTeX text commands
// and inline formatting tags.
std::string escape_cell_markup(const std::string& text) {
    static const std::pair<std::string_view, std::string_view> kCommands[] = {
        {"textless", "&lt;"},  {"textgreater", "&gt;"}, {"textasciitilde", "~"},
        {"linebreak", "<br>"}, {"mbox", ""}};
    static const std::set<std::string> kTags = {"b",  "br",     "code", "em", "i",
                                                "strong", "sub", "sup", "u"};
    std::string out;
    size_t plain = 0;
    auto flush = [&](size_t until) {
        html_escape_into(std::string_view(text).substr(plain, until - plain), &out);
    };
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == '\\' && i + 1 < text.size()) {
            const char next = text[i + 1];
            if (next == '%' || next == '&' || next == '#' || next == '$') {
                flush(i);
                out += next == '&' ? "&amp;" : std::string(1, next);
                plain = i + 2;
                i++;
                continue;
            }
            bool matched = false;
            for (const auto& [name, replacement] : kCommands) {
                if (text.compare(i + 1, name.size(), name) != 0) {
                    continue;
                }
                size_t end = i + 1 + name.size();
                if (end < text.size() && std::isalpha(static_cast<unsigned char>(text[end])) != 0) {
                    continue;
                }
                if (text.compare(end, 2, "{}") == 0) {
                    end += 2;
                }
                flush(i);
                out += replacement;
                plain = end;
                i = end - 1;
                matched = true;
                break;
            }
            if (matched) {
                continue;
            }
        }
        if (text[i] == '<') {
            size_t j = i + 1;
            if (j < text.size() && text[j] == '/') {
                j++;
            }
            size_t name_end = j;
            while (name_end < text.size() &&
                   std::isalpha(static_cast<unsigned char>(text[name_end])) != 0) {
                name_end++;
            }
            std::string name = text.substr(j, name_end - j);
            for (char& c : name) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            size_t k = name_end;
            while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k])) != 0) {
                k++;
            }
            if (k < text.size() && text[k] == '/') {
                k++;
            }
            if (kTags.contains(name) && k < text.size() && text[k] == '>') {
                flush(i);
                out += text.substr(i, k + 1 - i);
                plain = k + 1;
                i = k;
            }
        }
    }
    flush(text.size());
    return out;
}

struct LatexCell {
    std::string text;
    size_t row_span = 1;
    size_t col_span = 1;
    bool header = false;
};

LatexCell parse_latex_cell(std::string text) {
    LatexCell cell;
    while (true) {
        if (auto args = unwrap_command(text, "multicolumn", 3)) {
            cell.col_span = span_value((*args)[0]);
            text = (*args)[2];
        } else if (auto args = unwrap_command(text, "multirow", 3)) {
            cell.row_span = span_value((*args)[0]);
            text = (*args)[2];
        } else if (auto args = unwrap_command(text, "thead", 1)) {
            cell.header = true;
            text = (*args)[0];
        } else {
            break;
        }
    }
    cell.text = trim(escape_cell_markup(text));
    return cell;
}

// Nullopt when the table is wider than docling's 1000-column cap.
std::optional<std::string> latex_tabular_to_html(const std::string& latex) {
    std::string body = latex;
    const std::string begin_token = "\\begin{tabular}";
    const size_t begin = latex.find(begin_token);
    if (begin != std::string::npos) {
        body = latex.substr(begin + begin_token.size());
        if (auto spec = read_group(body, 0)) {
            body = body.substr(spec->second);
        }
    }
    const size_t end = body.find("\\end{tabular}");
    if (end != std::string::npos) {
        body.resize(end);
    }
    std::vector<std::string> rows;
    for (std::string& row : split_top_level(body, "\\\\")) {
        if (!trim(row).empty()) {
            rows.push_back(std::move(row));
        }
    }
    std::vector<std::unique_ptr<LatexCell>> storage;
    std::map<size_t, std::pair<LatexCell*, size_t>> owners;
    std::vector<std::vector<LatexCell*>> grid;
    for (size_t row_index = 0; row_index < rows.size(); row_index++) {
        size_t col = 0;
        std::vector<LatexCell*> cells;
        for (std::string& raw : split_top_level(rows[row_index], "&")) {
            storage.push_back(std::make_unique<LatexCell>(parse_latex_cell(std::move(raw))));
            LatexCell* cell = storage.back().get();
            const auto owner = owners.find(col);
            if (owner != owners.end() &&
                owner->second.second + owner->second.first->row_span > row_index) {
                if (cell->text.empty()) {
                    col += cell->col_span;
                    continue;
                }
                owner->second.first->row_span = row_index - owner->second.second;
            }
            if (col + cell->col_span > kMaxLatexColumns) {
                return std::nullopt;
            }
            for (size_t c = col; c < col + cell->col_span; c++) {
                owners[c] = {cell, row_index};
            }
            cells.push_back(cell);
            col += cell->col_span;
        }
        grid.push_back(std::move(cells));
    }
    std::string html = "<table>";
    for (const auto& cells : grid) {
        html += "<tr>";
        for (const LatexCell* cell : cells) {
            const char* tag = cell->header ? "th" : "td";
            html += "<";
            html += tag;
            if (cell->header) {
                html += " scope=\"col\"";
            }
            html += " rowspan=\"" + std::to_string(cell->row_span) + "\" colspan=\"" +
                    std::to_string(cell->col_span) + "\">";
            html += markdown_inline_html(cell->text);
            html += "</";
            html += tag;
            html += ">";
        }
        html += "</tr>";
    }
    html += "</table>";
    return html;
}

// ---- Markdown data tables (charts) ----------------------------------------

// _markdown_table_data: separator rows dropped, cells split on '|'.
std::vector<std::vector<std::string>> markdown_table_rows(const std::vector<std::string>& lines) {
    std::vector<std::vector<std::string>> grid;
    for (const std::string& line : lines) {
        if (line.find_first_not_of("|-: ") == std::string::npos) {
            continue;
        }
        std::string row = trim(line);
        const size_t first = row.find_first_not_of('|');
        const size_t last = row.find_last_not_of('|');
        row = first == std::string::npos ? "" : row.substr(first, last - first + 1);
        std::vector<std::string> cells;
        size_t start = 0;
        while (true) {
            const size_t bar = row.find('|', start);
            cells.push_back(trim(row.substr(start, bar == std::string::npos ? bar : bar - start)));
            if (bar == std::string::npos) {
                break;
            }
            start = bar + 1;
        }
        grid.push_back(std::move(cells));
    }
    return grid;
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find_first_of("\r\n", start);
        if (end == std::string::npos) {
            end = text.size();
        }
        lines.push_back(text.substr(start, end - start));
        start = end + (text.compare(end, 2, "\r\n") == 0 ? 2 : 1);
    }
    return lines;
}

std::string chart_label(const std::string& type) {
    auto has = [&](std::string_view word) { return type.find(word) != std::string::npos; };
    if (has("heatmap")) {
        return "heatmap";
    }
    if (has("stacked") && has("bar")) {
        return "stacked_bar_chart";
    }
    if (has("bar")) {
        return "bar_chart";
    }
    if (has("line")) {
        return "line_chart";
    }
    if (has("scatter") || has("bubble")) {
        return "scatter_chart";
    }
    if (has("pie") || has("donut")) {
        return "pie_chart";
    }
    if (has("box")) {
        return "box_plot";
    }
    return "other_chart";
}

// ---- Nemotron Parse 2.0 ---------------------------------------------------

struct Region {
    double x1, y1, x2, y2;
    std::string text;
    std::string label;
};

// finditer over <x_><y_>(.*?)<x_><y_><class_..>: the first closing
// triple after each opening pair. Without any closing triple after an
// opening, no later opening has one either.
std::vector<Region> nemotron_v2_regions(std::string_view text) {
    std::vector<Region> regions;
    size_t i = 0;
    while (i < text.size()) {
        const size_t open = text.find("<x_", i);
        if (open == std::string_view::npos) {
            break;
        }
        Region region{};
        const auto body_start = coord_token(text, open, &region.x1, &region.y1);
        if (!body_start) {
            i = open + 1;
            continue;
        }
        std::optional<size_t> match_end;
        size_t close = *body_start;
        for (size_t at = text.find("<x_", close); at != std::string_view::npos;
             at = text.find("<x_", at + 1)) {
            const auto after = coord_token(text, at, &region.x2, &region.y2);
            if (!after) {
                continue;
            }
            if (const auto end = class_token(text, *after, &region.label)) {
                close = at;
                match_end = end;
                break;
            }
        }
        if (!match_end) {
            break;
        }
        region.text = std::string(text.substr(*body_start, close - *body_start));
        if (region.label == "Inline-formula") {
            region.label = "Formula";
        }
        regions.push_back(std::move(region));
        i = *match_end;
    }
    return regions;
}

docv1::BoundingBox nemotron_v2_box(const Region& region, const PageContext& page) {
    constexpr double kCanvasW = 1664;
    constexpr double kCanvasH = 2048;
    const double image_w = page.width;
    const double image_h = page.height;
    const double aspect = image_h > 0 ? image_w / image_h : 1.0;
    double resized_w = image_w;
    double resized_h = image_h;
    if (resized_h > kCanvasH) {
        resized_h = kCanvasH;
        resized_w = std::trunc(resized_h * aspect);
    }
    if (resized_w > kCanvasW) {
        resized_w = kCanvasW;
        resized_h = std::trunc(resized_w / aspect);
    }
    const double pad_left = std::floor(std::max(0.0, kCanvasW - resized_w) / 2);
    const double pad_top = std::floor(std::max(0.0, kCanvasH - resized_h) / 2);
    const double sx = resized_w > 0 ? page.width / resized_w : 1.0;
    const double sy = resized_h > 0 ? page.height / resized_h : 1.0;
    auto clamp = [](double v, double hi) { return std::max(0.0, std::min(hi, v)); };
    docv1::BoundingBox box;
    box.set_l(clamp((region.x1 * kCanvasW - pad_left) * sx, page.width));
    box.set_r(clamp((region.x2 * kCanvasW - pad_left) * sx, page.width));
    box.set_t(clamp((region.y1 * kCanvasH - pad_top) * sy, page.height));
    box.set_b(clamp((region.y2 * kCanvasH - pad_top) * sy, page.height));
    box.set_coord_origin(docv1::COORD_ORIGIN_TOPLEFT);
    return box;
}

// Docling 2.0 reads a table through its LaTeX backend (first table,
// trailing empty rows dropped, header row 0, plain cell text). This reads
// the same tabular dialect the 3.5 path parses.
std::vector<SpanCell> nemotron_v2_table_cells(const std::string& latex) {
    const std::optional<std::string> html = latex_tabular_to_html(latex);
    if (!html) {
        return {};
    }
    const std::unique_ptr<MarkupNode> root = parse_markup(*html, MarkupMode::kHtml);
    const MarkupNode* table = find_element(*root, "table");
    if (table == nullptr) {
        return {};
    }
    std::vector<SpanCell> cells = chandra_table_cells(*table);
    size_t last_row = 0;
    bool any = false;
    for (SpanCell& cell : cells) {
        cell.text = markdown_to_text(collapse_whitespace(html_text(*cell.node)));
        cell.column_header = cell.row == 0;
        cell.row_header = false;
        cell.node = nullptr;
        if (!trim(cell.text).empty()) {
            last_row = std::max(last_row, cell.row + cell.row_span - 1);
            any = true;
        }
    }
    if (!any) {
        return {};
    }
    std::erase_if(cells, [&](const SpanCell& cell) { return cell.row > last_row; });
    return cells;
}

docv1::DocItemLabel nemotron_text_label(const std::string& label) {
    static const std::map<std::string, docv1::DocItemLabel> kLabels = {
        {"Text", docv1::DOC_ITEM_LABEL_TEXT},
        {"Caption", docv1::DOC_ITEM_LABEL_CAPTION},
        {"Footnote", docv1::DOC_ITEM_LABEL_FOOTNOTE},
        {"Page-header", docv1::DOC_ITEM_LABEL_PAGE_HEADER},
        {"Page-footer", docv1::DOC_ITEM_LABEL_PAGE_FOOTER},
        {"Bibliography", docv1::DOC_ITEM_LABEL_REFERENCE},
        {"Formula", docv1::DOC_ITEM_LABEL_FORMULA},
    };
    const auto it = kLabels.find(label);
    return it == kLabels.end() ? docv1::DOC_ITEM_LABEL_TEXT : it->second;
}

int markdown_heading_level(const std::string& text, int fallback) {
    size_t i = 0;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
        i++;
    }
    size_t hashes = 0;
    while (i + hashes < text.size() && text[i + hashes] == '#') {
        hashes++;
    }
    if (hashes == 0 || hashes > 6 || i + hashes >= text.size() ||
        std::isspace(static_cast<unsigned char>(text[i + hashes])) == 0) {
        return fallback;
    }
    return static_cast<int>(hashes);
}

// ---- Nemotron Parse 3.5 ---------------------------------------------------

struct Block {
    std::string label;
    double x1, y1, x2, y2;
    std::string text;
    bool operator<(const Block& o) const {
        return std::tie(label, x1, y1, x2, y2, text) <
               std::tie(o.label, o.x1, o.y1, o.x2, o.y2, o.text);
    }
};

std::vector<Block> nemotron_v35_blocks(std::string_view content) {
    std::vector<Block> blocks;
    std::set<Block> seen;
    size_t start = 0;
    for (size_t at = content.find("<class_"); at != std::string_view::npos;
         at = content.find("<class_", at + 1)) {
        std::string label;
        const auto end = class_token(content, at, &label);
        if (!end) {
            continue;
        }
        const std::string_view segment = content.substr(start, at - start);
        start = *end;
        struct Coord {
            size_t begin, end;
            double x, y;
        };
        std::vector<Coord> coords;
        for (size_t i = segment.find("<x_"); i != std::string_view::npos;
             i = segment.find("<x_", i)) {
            Coord c{i, 0, 0, 0};
            if (const auto e = coord_token(segment, i, &c.x, &c.y)) {
                c.end = *e;
                coords.push_back(c);
                i = *e;
            } else {
                i++;
            }
        }
        at = *end - 1;
        if (coords.size() < 2) {
            continue;
        }
        const Coord& first = coords.front();
        const Coord& last = coords.back();
        std::string body;
        size_t pos = first.end;
        for (size_t k = 1; k + 1 < coords.size(); k++) {
            body += segment.substr(pos, coords[k].begin - pos);
            pos = coords[k].end;
        }
        body += segment.substr(pos, last.begin - pos);
        Block block{label, first.x, first.y, last.x, last.y, trim(remove_artifacts(body))};
        if (seen.insert(block).second) {
            blocks.push_back(std::move(block));
        }
    }
    return blocks;
}

// _LIST_MARKER: the marker and the offset of the body after it.
std::optional<std::pair<std::string, size_t>> list_marker(const std::string& text,
                                                          bool* ordered) {
    size_t i = 0;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
        i++;
    }
    const size_t begin = i;
    auto ordered_marker = [&](size_t at) -> std::optional<size_t> {
        size_t j = at;
        if (j < text.size() && text[j] == '(') {
            j++;
        }
        auto close = [&](size_t k) -> std::optional<size_t> {
            if (k < text.size() && (text[k] == '.' || text[k] == ')')) {
                return k + 1;
            }
            return std::nullopt;
        };
        size_t k = j;
        while (k < text.size() && std::isdigit(static_cast<unsigned char>(text[k])) != 0) {
            k++;
        }
        if (k > j) {
            return close(k);
        }
        if (j < text.size() && std::isalpha(static_cast<unsigned char>(text[j])) != 0) {
            if (auto end = close(j + 1)) {
                return end;
            }
            static const std::string_view kRoman = "ivxlcdmIVXLCDM";
            k = j;
            while (k < text.size() && kRoman.find(text[k]) != std::string_view::npos) {
                k++;
            }
            if (k > j) {
                return close(k);
            }
        }
        return std::nullopt;
    };
    std::optional<size_t> end;
    *ordered = false;
    if ((end = ordered_marker(begin))) {
        *ordered = true;
    } else {
        static const std::string_view kBullets[] = {"-",  "\u2013", "\u2014", "\u2022",
                                                    "\u25ba", "\u25aa", "\u25a0", "\u25a1",
                                                    "\u25e6", "\u25cf", "\u25cb", "\u00b7",
                                                    "*",  "+"};
        for (std::string_view bullet : kBullets) {
            if (text.compare(begin, bullet.size(), bullet) == 0) {
                end = begin + bullet.size();
                break;
            }
        }
        if (!end && text.compare(begin, 3, "\\(\\") == 0) {
            size_t k = begin + 3;
            while (k < text.size() && std::isalpha(static_cast<unsigned char>(text[k])) != 0) {
                k++;
            }
            if (k > begin + 3 && text.compare(k, 2, "\\)") == 0) {
                end = k + 2;
            }
        }
    }
    if (!end || *end >= text.size() || std::isspace(static_cast<unsigned char>(text[*end])) == 0) {
        *ordered = false;
        return std::nullopt;
    }
    size_t body = *end;
    while (body < text.size() && std::isspace(static_cast<unsigned char>(text[body])) != 0) {
        body++;
    }
    return std::pair{text.substr(begin, *end - begin), body};
}

class V35Builder {
  public:
    V35Builder(LayoutWriter& w, docv1::Document* out) : w_(w), out_(out) {}

    void rich_text(const std::string& text, docv1::DocItemLabel label, const Box& box,
                   std::optional<Ref> parent) {
        const int texts_before = out_->texts_size();
        const int groups_before = out_->groups_size();
        if (!add_inline_text(w_, text, label, box, parent)) {
            const Ref item = w_.styled_text(label, box, unescape_markdown_literals(text), nullptr,
                                            "", parent.value_or(LayoutWriter::kBody));
            w_.set_orig(item, text);
        }
        if (label == docv1::DOC_ITEM_LABEL_PAGE_HEADER ||
            label == docv1::DOC_ITEM_LABEL_PAGE_FOOTER) {
            for (int i = texts_before; i < out_->texts_size(); i++) {
                w_.set_content_layer(Ref{BodyChild::TEXT, i}, docv1::CONTENT_LAYER_FURNITURE);
            }
            for (int i = groups_before; i < out_->groups_size(); i++) {
                w_.set_content_layer(Ref{BodyChild::GROUP, i}, docv1::CONTENT_LAYER_FURNITURE);
            }
        }
    }

    void latex_table(const std::string& latex, const Box& box, Ref parent) {
        const std::optional<std::string> html = latex_tabular_to_html(latex);
        if (!html || !add_html_table(w_, *html, box, parent)) {
            w_.table(box, {}, parent);
        }
    }

    void chart_picture(const Box& box, const std::string& type, const std::string& title,
                       const std::vector<std::string>& rows) {
        const Ref picture = w_.picture(box);
        w_.classify_picture(picture, chart_label(type), "");
        w_.set_tabular_chart(picture, title, markdown_table_rows(rows), "");
    }

    void chart(const std::string& text, const Box& box) {
        std::string type = "other";
        std::string body = text;
        // <chart\s+type="([^"]*)"\s*>
        for (size_t at = text.find("<chart"); at != std::string::npos;
             at = text.find("<chart", at + 1)) {
            size_t i = at + 6;
            const size_t spaces = i;
            while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
                i++;
            }
            if (i == spaces || text.compare(i, 6, "type=\"") != 0) {
                continue;
            }
            const size_t value = i + 6;
            const size_t quote = text.find('"', value);
            if (quote == std::string::npos) {
                continue;
            }
            size_t j = quote + 1;
            while (j < text.size() && std::isspace(static_cast<unsigned char>(text[j])) != 0) {
                j++;
            }
            if (j >= text.size() || text[j] != '>') {
                continue;
            }
            type = text.substr(value, quote - value);
            body = text.substr(j + 1);
            break;
        }
        const size_t close = body.find("</chart>");
        if (close != std::string::npos) {
            body.resize(close);
        }
        std::string title;
        struct Panel {
            std::string id, type, label;
        };
        std::vector<Panel> panels;
        std::map<std::string, std::vector<std::string>> tables;  // "" is the untitled table
        std::vector<std::string> table_order{""};
        std::string key;
        for (const std::string& line : split_lines(body)) {
            const std::string stripped = trim(line);
            if (stripped.starts_with("|")) {
                if (!tables.contains(key)) {
                    table_order.push_back(key);
                }
                tables[key].push_back(stripped);
                continue;
            }
            const size_t colon = stripped.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            const std::string name = stripped.substr(0, colon);
            const std::string value = stripped.substr(colon + 1);
            if (name == "title") {
                title = trim(value);
            } else if (name == "panel") {
                std::vector<std::string> parts;
                size_t start = 0;
                for (int n = 0; n < 2; n++) {
                    const size_t semi = value.find(';', start);
                    if (semi == std::string::npos) {
                        break;
                    }
                    parts.push_back(trim(value.substr(start, semi - start)));
                    start = semi + 1;
                }
                parts.push_back(trim(value.substr(start)));
                if (parts.size() >= 2) {
                    panels.push_back({parts[0], parts[1], parts.size() == 3 ? parts[2] : ""});
                }
            } else if (name.starts_with("table[") && name.ends_with("]") && name.size() > 7 &&
                       std::all_of(name.begin() + 6, name.end() - 1, [](char c) {
                           return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
                       })) {
                key = name.substr(6, name.size() - 7);
            }
        }
        bool any_panel = false;
        for (const Panel& panel : panels) {
            const auto it = tables.find(panel.id);
            if (!panel.id.empty() && it != tables.end() && !it->second.empty()) {
                chart_picture(box, panel.type, panel.label, it->second);
                any_panel = true;
            }
        }
        if (any_panel) {
            return;
        }
        std::vector<std::string> rows = tables[""];
        if (rows.empty()) {
            for (const std::string& k : table_order) {
                if (!tables[k].empty()) {
                    rows = tables[k];
                    break;
                }
            }
        }
        chart_picture(box, type, title, rows);
    }

  private:
    LayoutWriter& w_;
    docv1::Document* out_;
};

}  // namespace

bool map_nemotron_parse_v2(const std::string& text, const PageContext& page,
                           docv1::Document* out, std::string* error,
                           std::vector<vlmv1::PageWarning>* warnings) {
    LayoutWriter w(page, out, warnings);
    std::optional<Ref> list;
    for (const Region& region : nemotron_v2_regions(text)) {
        const Box box = nemotron_v2_box(region, page);
        const std::string raw = trim(remove_artifacts(region.text));
        if (region.label != "List-item") {
            list.reset();
        }
        Ref item;
        if (region.label == "List-item") {
            if (!list) {
                list = w.list_group();
            }
            item = w.list_item(box, markdown_to_text(raw), false, "", *list);
        } else if (region.label == "Picture") {
            w.picture(box);
            continue;
        } else if (region.label == "Table") {
            w.table(box, nemotron_v2_table_cells(raw));
            continue;
        } else if (region.label == "Chart") {
            const Ref picture = w.picture(box);
            std::vector<std::string> lines;
            for (const std::string& line : split_lines(raw)) {
                const std::string stripped = trim(line);
                if (stripped.starts_with("|")) {
                    lines.push_back(stripped);
                } else if (!lines.empty()) {
                    break;  // the first table only
                }
            }
            w.set_tabular_chart(picture, "", markdown_table_rows(lines), "");
            continue;
        } else if (region.label == "Title") {
            item = w.title(box, markdown_to_text(raw));
        } else if (region.label == "Section-header") {
            item = w.heading(markdown_heading_level(raw, 2), box, markdown_to_text(raw));
        } else {
            const docv1::DocItemLabel label = nemotron_text_label(region.label);
            item = label == docv1::DOC_ITEM_LABEL_FORMULA
                       ? w.formula(box, strip_math_delimiters(raw))
                       : w.styled_text(label, box, markdown_to_text(raw), nullptr, "");
        }
        w.set_orig(item, raw);
    }
    return w.finish(error, "Nemotron Parse", /*allow_empty=*/true);
}

bool map_nemotron_parse_v3_5(const std::string& text, const PageContext& page,
                             docv1::Document* out, std::string* error,
                             std::vector<vlmv1::PageWarning>* warnings) {
    LayoutWriter w(page, out, warnings);
    V35Builder builder(w, out);
    std::optional<Ref> list;
    std::optional<Ref> toc;
    std::vector<size_t> starts;
    std::vector<size_t> caption_blocks;
    const std::vector<Block> blocks = nemotron_v35_blocks(text);
    for (size_t index = 0; index < blocks.size(); index++) {
        const Block& block = blocks[index];
        starts.push_back(w.body_size());
        auto unit = [](double v) { return std::min(1.0, std::max(0.0, v)); };
        const double x1 = unit(block.x1);
        const double y1 = unit(block.y1);
        const double x2 = unit(block.x2);
        const double y2 = unit(block.y2);
        const Box box = w.scaled_box(std::min(x1, x2), std::min(y1, y2), std::max(x1, x2),
                                     std::max(y1, y2), page.width, page.height);
        const std::string& body = block.text;
        if (block.label != "List-item") {
            list.reset();
        }
        if (block.label != "TOC") {
            toc.reset();
        }
        if (block.label == "List-item") {
            if (!list) {
                list = w.list_group();
            }
            bool ordered = false;
            const auto marker = list_marker(body, &ordered);
            const std::string item_body = marker ? body.substr(marker->second) : body;
            const Ref item =
                w.list_item(box, "", ordered, marker ? marker->first : std::string(), *list);
            w.set_orig(item, body);
            if (!add_inline_text(w, item_body, docv1::DOC_ITEM_LABEL_TEXT, std::nullopt, item)) {
                w.set_text(item, unescape_markdown_literals(item_body), nullptr, "");
            }
        } else if (block.label == "TOC") {
            if (!toc) {
                toc = w.group("toc", docv1::GROUP_LABEL_UNSPECIFIED);
            }
            if (body.find("\\begin{tabular}") != std::string::npos) {
                builder.latex_table(body, box, *toc);
            } else {
                builder.rich_text(body, docv1::DOC_ITEM_LABEL_TEXT, box, *toc);
            }
        } else if (block.label == "Table") {
            builder.latex_table(body, box, LayoutWriter::kBody);
        } else if (block.label == "Chart") {
            builder.chart(body, box);
        } else if (block.label == "Picture") {
            const Ref picture = w.picture(box);
            if (!body.empty()) {
                builder.rich_text(body, docv1::DOC_ITEM_LABEL_TEXT, std::nullopt, picture);
            }
        } else if (block.label == "Title" || block.label == "Section-header") {
            int level = 0;
            const std::string clean = strip_markdown_heading(body, &level);
            const Ref heading = block.label == "Title"
                                    ? w.title(box, "")
                                    : w.heading(level > 0 ? level : 2, box, "");
            w.set_orig(heading, body);
            if (!add_inline_text(w, clean, docv1::DOC_ITEM_LABEL_TEXT, std::nullopt, heading)) {
                w.set_text(heading, unescape_markdown_literals(clean), nullptr, "");
            }
        } else if (block.label == "Formula") {
            const Ref item = w.formula(box, strip_math_delimiters(body));
            w.set_orig(item, body);
        } else if (block.label == "Code") {
            w.code(box, body, "");
        } else {
            if (block.label == "Caption") {
                caption_blocks.push_back(index);
            }
            static const std::map<std::string, docv1::DocItemLabel> kLabels = {
                {"Caption", docv1::DOC_ITEM_LABEL_CAPTION},
                {"Footnote", docv1::DOC_ITEM_LABEL_FOOTNOTE},
                {"Bibliography", docv1::DOC_ITEM_LABEL_REFERENCE},
                {"Page-header", docv1::DOC_ITEM_LABEL_PAGE_HEADER},
                {"Page-footer", docv1::DOC_ITEM_LABEL_PAGE_FOOTER},
            };
            const auto it = kLabels.find(block.label);
            builder.rich_text(body,
                              it == kLabels.end() ? docv1::DOC_ITEM_LABEL_TEXT : it->second, box,
                              std::nullopt);
        }
    }
    w.link_adjacent_captions(starts, caption_blocks);
    return w.finish(error, "Nemotron Parse 3.5", /*allow_empty=*/true);
}

}  // namespace vlm::mapping
