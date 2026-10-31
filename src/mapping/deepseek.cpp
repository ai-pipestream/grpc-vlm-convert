// DeepSeek-OCR and Unlimited-OCR grounded Markdown (docling
// deepseekocr_utils.py, v2.142.1): each block opens with an annotation
// line `<|ref|>label<|/ref|><|det|>[[x0, y0, x1, y1]]<|/det|>` on a
// 0..1000 grid; the block's text runs to the next annotation (a table
// keeps only its <table>...</table>). Unlimited-OCR's inline
// `<|det|>label [x0,y0,x1,y1]<|/det|>` form is rewritten to that shape
// first.

#include <cctype>
#include <charconv>
#include <optional>

#include "layout.h"

namespace vlm::mapping {

namespace {

constexpr double kGrid = 1000.0;

bool is_word(unsigned char c) {
    return std::isalnum(c) != 0 || c == '_' || c >= 0x80;
}

bool consume(std::string_view& text, std::string_view prefix) {
    if (text.starts_with(prefix)) {
        text.remove_prefix(prefix.size());
        return true;
    }
    return false;
}

std::string_view strip_view(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1);
    }
    return text;
}

// Python float() of one stripped piece of `[0-9.]` characters.
std::optional<double> parse_number(std::string_view piece) {
    piece = strip_view(piece);
    if (piece.empty() || piece == ".") {
        return std::nullopt;
    }
    double value = 0;
    const auto [end, ec] = std::from_chars(piece.data(), piece.data() + piece.size(), value);
    if (ec != std::errc() || end != piece.data() + piece.size()) {
        return std::nullopt;
    }
    return value;
}

struct Annotation {
    std::string label;
    std::vector<double> coords;  // empty: the coordinates did not parse to four numbers
};

// The annotation pattern
// ^(?:<|ref|>)?(\w+)(?:<|/ref|>)?(?:<|det|>)?\[\[([0-9., ]+)\]\](?:<|/det|>)?\s*$
// on a stripped line, as a linear scan.
std::optional<Annotation> match_annotation(std::string_view line) {
    consume(line, "<|ref|>");
    size_t word = 0;
    while (word < line.size() && is_word(static_cast<unsigned char>(line[word]))) {
        word++;
    }
    if (word == 0) {
        return std::nullopt;
    }
    Annotation annotation;
    annotation.label = std::string(line.substr(0, word));
    line.remove_prefix(word);
    consume(line, "<|/ref|>");
    consume(line, "<|det|>");
    if (!consume(line, "[[")) {
        return std::nullopt;
    }
    size_t body = 0;
    while (body < line.size() &&
           (std::isdigit(static_cast<unsigned char>(line[body])) != 0 || line[body] == '.' ||
            line[body] == ',' || line[body] == ' ')) {
        body++;
    }
    if (body == 0) {
        return std::nullopt;
    }
    const std::string_view coords = line.substr(0, body);
    line.remove_prefix(body);
    if (!consume(line, "]]")) {
        return std::nullopt;
    }
    consume(line, "<|/det|>");
    if (!strip_view(line).empty()) {
        return std::nullopt;
    }
    std::vector<double> numbers;
    size_t start = 0;
    while (start <= coords.size()) {
        size_t comma = coords.find(',', start);
        if (comma == std::string_view::npos) {
            comma = coords.size();
        }
        const auto number = parse_number(coords.substr(start, comma - start));
        if (!number) {
            return annotation;  // float() failed: docling skips the line
        }
        numbers.push_back(*number);
        start = comma + 1;
    }
    if (numbers.size() == 4) {
        annotation.coords = std::move(numbers);
    }
    return annotation;
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (true) {
        const size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            lines.push_back(text.substr(start));
            return lines;
        }
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
}

std::string rstrip(const std::string& text) {
    size_t end = text.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        end--;
    }
    return text.substr(0, end);
}

std::string strip_newlines(const std::string& text) {
    const size_t first = text.find_first_not_of('\n');
    if (first == std::string::npos) {
        return "";
    }
    return text.substr(first, text.find_last_not_of('\n') - first + 1);
}

bool contains_ci(const std::string& text, std::string_view needle) {
    if (needle.size() > text.size()) {
        return false;
    }
    for (size_t i = 0; i + needle.size() <= text.size(); i++) {
        bool same = true;
        for (size_t j = 0; j < needle.size() && same; j++) {
            same = std::tolower(static_cast<unsigned char>(text[i + j])) == needle[j];
        }
        if (same) {
            return true;
        }
    }
    return false;
}

std::string lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

void raw_text(const MarkupNode& node, std::string* out) {
    if (node.is_text()) {
        *out += node.text;
        return;
    }
    for (const auto& child : node.children) {
        raw_text(*child, out);
    }
}

void collect_rows(const MarkupNode& node, std::vector<const MarkupNode*>* rows) {
    for (const auto& child : node.children) {
        if (child->is_text()) {
            continue;
        }
        if (child->tag == "tr") {
            rows->push_back(child.get());
        }
        collect_rows(*child, rows);
    }
}

// Python int() of an attribute: optional sign, digits, surrounding space.
std::optional<long long> python_int(const std::string& value) {
    const std::string_view text = strip_view(value);
    if (text.empty()) {
        return std::nullopt;
    }
    long long number = 0;
    const char* begin = text.data() + (text.front() == '+' ? 1 : 0);
    const auto [end, ec] = std::from_chars(begin, text.data() + text.size(), number);
    if (ec != std::errc() || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return number;
}

}  // namespace

// _parse_table_html: the first <table ...>...</table>, every <tr> in it
// (nested ones included, as lxml's .//tr finds them), cells placed in the
// first free column; a header cell is a <th> in row 0, a row header a
// <th> in column 0. An unparsable span gives docling an empty table, and
// here too.
std::vector<SpanCell> deepseek_table_cells(const std::string& content) {
    const std::string folded = lower(content);
    size_t open = std::string::npos;
    for (size_t at = folded.find("<table"); at != std::string::npos;
         at = folded.find("<table", at + 1)) {
        open = at;
        break;
    }
    if (open == std::string::npos) {
        return {};
    }
    const size_t open_end = folded.find('>', open);
    if (open_end == std::string::npos) {
        return {};
    }
    const size_t close = folded.find("</table>", open_end);
    if (close == std::string::npos) {
        return {};
    }
    const std::unique_ptr<MarkupNode> tree =
        parse_markup(std::string_view(content).substr(open, close + 8 - open), MarkupMode::kHtml);
    std::vector<const MarkupNode*> rows;
    collect_rows(*tree, &rows);
    if (rows.empty()) {
        return {};
    }
    struct Raw {
        const MarkupNode* node;
        long long colspan;
        long long rowspan;
    };
    std::vector<std::vector<Raw>> grid_rows;
    size_t num_cols = 0;
    for (const MarkupNode* row : rows) {
        std::vector<Raw>& cells = grid_rows.emplace_back();
        size_t count = 0;
        for (const auto& cell : row->children) {
            if (cell->is_text() || (cell->tag != "td" && cell->tag != "th")) {
                continue;
            }
            const auto colspan = python_int(cell->attr("colspan", "1"));
            const auto rowspan = python_int(cell->attr("rowspan", "1"));
            if (!colspan || !rowspan) {
                return {};  // int() raised: docling logs it and keeps an empty table
            }
            // Clamped to a sane range so a hostile span cannot size the grid.
            const long long col_span = std::clamp<long long>(*colspan, 1, kMaxTableCells);
            const long long row_span = std::clamp<long long>(*rowspan, 1, kMaxTableCells);
            cells.push_back({cell.get(), col_span, row_span});
            count += static_cast<size_t>(col_span);
        }
        num_cols = std::max(num_cols, count);
    }
    const size_t num_rows = rows.size();
    num_cols = std::min(num_cols, kMaxTableCols * 4);
    std::vector<std::vector<bool>> taken(num_rows, std::vector<bool>(num_cols, false));
    std::vector<SpanCell> out;
    for (size_t r = 0; r < num_rows; r++) {
        size_t col = 0;
        for (const Raw& raw : grid_rows[r]) {
            while (col < num_cols && taken[r][col]) {
                col++;
            }
            if (col >= num_cols) {
                break;
            }
            const auto col_span = static_cast<size_t>(raw.colspan);
            const auto row_span = static_cast<size_t>(raw.rowspan);
            for (size_t rr = r; rr < std::min(r + row_span, num_rows); rr++) {
                for (size_t cc = col; cc < std::min(col + col_span, num_cols); cc++) {
                    taken[rr][cc] = true;
                }
            }
            SpanCell cell;
            cell.row = r;
            cell.col = col;
            // Spans stop at the table edge (docling keeps the raw span on
            // the cell; a span past the grid has nothing to cover).
            cell.row_span = std::min(row_span, num_rows - r);
            cell.col_span = std::min(col_span, num_cols - col);
            std::string text;
            raw_text(*raw.node, &text);
            cell.text = trim(text);
            const bool th = raw.node->tag == "th";
            cell.column_header = th && r == 0;
            cell.row_header = th && col == 0;
            out.push_back(std::move(cell));
            col += col_span;
        }
    }
    return out;
}

// normalize_unlimited_ocr_annotations: every
// `<|det|>\s*label\s*[d,d,d,d]<|/det|>` (label [a-z_]+, whitespace allowed
// around the commas) becomes `<|ref|>label<|/ref|><|det|>[[d,d,d,d]]<|/det|>\n`.
std::string normalize_unlimited_ocr(const std::string& content) {
    static constexpr std::string_view kOpen = "<|det|>";
    static constexpr std::string_view kClose = "<|/det|>";
    std::string out;
    out.reserve(content.size() + 64);
    size_t pos = 0;
    while (pos < content.size()) {
        const size_t at = content.find(kOpen, pos);
        if (at == std::string::npos) {
            out.append(content, pos, std::string::npos);
            break;
        }
        out.append(content, pos, at - pos);
        size_t i = at + kOpen.size();
        auto skip_space = [&] {
            while (i < content.size() && std::isspace(static_cast<unsigned char>(content[i])) != 0) {
                i++;
            }
        };
        auto digits = [&] {
            const size_t start = i;
            while (i < content.size() && std::isdigit(static_cast<unsigned char>(content[i])) != 0) {
                i++;
            }
            return i > start;
        };
        skip_space();
        const size_t label_start = i;
        while (i < content.size() && ((content[i] >= 'a' && content[i] <= 'z') || content[i] == '_')) {
            i++;
        }
        const size_t label_end = i;
        bool ok = label_end > label_start;
        size_t bbox_start = 0;
        size_t bbox_end = 0;
        if (ok) {
            skip_space();
            ok = i < content.size() && content[i] == '[';
        }
        if (ok) {
            i++;
            bbox_start = i;
            ok = digits();
            for (int n = 0; ok && n < 3; n++) {
                skip_space();
                ok = i < content.size() && content[i] == ',';
                if (ok) {
                    i++;
                    skip_space();
                    ok = digits();
                }
            }
            bbox_end = i;
            ok = ok && i < content.size() && content[i] == ']' &&
                 std::string_view(content).substr(i + 1).starts_with(kClose);
        }
        if (!ok) {
            out.append(kOpen);
            pos = at + kOpen.size();
            continue;
        }
        out += "<|ref|>";
        out.append(content, label_start, label_end - label_start);
        out += "<|/ref|><|det|>[[";
        out.append(content, bbox_start, bbox_end - bbox_start);
        out += "]]<|/det|>\n";
        pos = i + 1 + kClose.size();
    }
    return out;
}

bool map_deepseek_ocr(const std::string& text, const PageContext& page, docv1::Document* out,
                      std::string* error, std::vector<vlmv1::PageWarning>* warnings) {
    LayoutWriter writer(page, out, warnings);
    const std::vector<std::string> lines = split_lines(text);
    struct Block {
        std::string label;
        std::string content;
        docv1::BoundingBox box;
    };
    std::vector<Block> blocks;
    std::vector<bool> visited(lines.size(), false);
    size_t i = 0;
    while (i < lines.size()) {
        if (visited[i]) {
            i++;
            continue;
        }
        const auto annotation = match_annotation(strip_view(lines[i]));
        if (!annotation || annotation->coords.empty()) {
            i++;
            continue;
        }
        i++;
        std::string content;
        bool first = true;
        auto append = [&](const std::string& line) {
            if (!first) {
                content.push_back('\n');
            }
            content += rstrip(line);
            first = false;
        };
        if (annotation->label == "table") {
            bool started = false;
            for (size_t ii = i; ii < lines.size(); ii++) {
                if (contains_ci(lines[ii], "<table")) {
                    started = true;
                }
                if (started) {
                    visited[ii] = true;
                    append(lines[ii]);
                }
                if (started && contains_ci(lines[ii], "</table>")) {
                    break;
                }
            }
        } else {
            while (i < lines.size()) {
                if (match_annotation(strip_view(lines[i]))) {
                    break;
                }
                if (!visited[i]) {
                    append(lines[i]);
                    visited[i] = true;
                }
                i++;
            }
        }
        blocks.push_back(
            {annotation->label, strip_newlines(content), writer.grid_box(annotation->coords, kGrid)});
    }

    auto caption_of = [](const std::string& owner) -> std::string {
        if (owner == "table") {
            return "table_caption";
        }
        if (owner == "figure") {
            return "figure_caption";
        }
        if (owner == "image") {
            return "image_caption";
        }
        return "";
    };
    auto strip_hashes = [](const std::string& content, int* count) {
        size_t hashes = 0;
        while (hashes < content.size() && content[hashes] == '#') {
            hashes++;
        }
        *count = static_cast<int>(hashes);
        return hashes == 0 ? content : trim(content.substr(hashes));
    };
    for (size_t idx = 0; idx < blocks.size(); idx++) {
        const Block& block = blocks[idx];
        const std::string& label = block.label;
        std::optional<LayoutWriter::Ref> caption;
        const std::string caption_label = caption_of(label);
        if (!caption_label.empty() && idx + 1 < blocks.size() &&
            blocks[idx + 1].label == caption_label) {
            caption = writer.text(docv1::DOC_ITEM_LABEL_CAPTION, blocks[idx + 1].box,
                                  blocks[idx + 1].content);
        }
        if (idx > 0 && label.ends_with("_caption") &&
            caption_of(blocks[idx - 1].label) == label) {
            continue;  // already emitted ahead of its owner
        }
        LayoutWriter::Ref ref;
        if (label == "figure" || label == "image") {
            ref = writer.picture(block.box);
        } else if (label == "table") {
            ref = writer.table(block.box, deepseek_table_cells(block.content));
        } else if (label == "title") {
            int hashes = 0;
            ref = writer.title(block.box, strip_hashes(block.content, &hashes));
        } else if (label == "sub_title") {
            int hashes = 0;
            const std::string body = strip_hashes(block.content, &hashes);
            ref = writer.heading(hashes > 0 ? hashes : 2, block.box, body);
        } else {
            docv1::DocItemLabel item_label = docv1::DOC_ITEM_LABEL_TEXT;
            if (label.ends_with("_caption")) {
                item_label = docv1::DOC_ITEM_LABEL_CAPTION;
            } else if (label == "header") {
                item_label = docv1::DOC_ITEM_LABEL_PAGE_HEADER;
            } else if (label == "footer") {
                item_label = docv1::DOC_ITEM_LABEL_PAGE_FOOTER;
            }
            ref = writer.text(item_label, block.box, block.content);
        }
        if (caption) {
            writer.link_caption(ref, *caption);
        }
    }
    return writer.finish(error, "DeepSeek-OCR");
}

bool map_unlimited_ocr(const std::string& text, const PageContext& page, docv1::Document* out,
                       std::string* error, std::vector<vlmv1::PageWarning>* warnings) {
    return map_deepseek_ocr(normalize_unlimited_ocr(text), page, out, error, warnings);
}

}  // namespace vlm::mapping
