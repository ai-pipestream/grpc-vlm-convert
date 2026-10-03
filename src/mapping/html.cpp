#include "builder.h"
#include "mapper.h"

#include <cctype>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace vlm::mapping {

namespace {

// Markup is read with linear scans, not std::regex: libstdc++'s regex
// engine recurses once per character of a lazy or repeated match, and one
// block of about 60 KB (a looping model's table) overflowed the stack and
// took the whole process down.

char lower(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

// \s as the regexes this replaces read it.
bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

// True when `text` spells `name` (lowercase) at `pos`, ignoring case.
bool at_ci(std::string_view text, size_t pos, std::string_view name) {
    if (pos > text.size() || text.size() - pos < name.size()) {
        return false;
    }
    for (size_t i = 0; i < name.size(); i++) {
        if (lower(text[pos + i]) != name[i]) {
            return false;
        }
    }
    return true;
}

// The first `needle` (lowercase) at or after `from`, ignoring case.
size_t find_ci(std::string_view text, std::string_view needle, size_t from) {
    for (size_t i = from; i < text.size() && text.size() - i >= needle.size(); i++) {
        if (at_ci(text, i, needle)) {
            return i;
        }
    }
    return std::string_view::npos;
}

// One element: its tag name (lowercase, as listed) and its content.
struct Found {
    std::string_view name;
    std::string_view content;
};

// Walks `<name ...>content</name>` elements in order for a fixed set of
// tag names, ignoring case; the content runs to the first matching closing
// tag, and elements do not nest. A '<' opens an element only when a listed
// name follows and ends right there (whitespace, '>' or '/'), so <p is not
// <pre and <tr is not <track. Each search for a '>' or a closing tag
// resumes where the previous one for it ended, which keeps a page of
// unclosed tags linear.
class ElementScanner {
  public:
    ElementScanner(std::string_view text, std::vector<std::string_view> names)
        : text_(text), names_(std::move(names)), close_(names_.size()) {
        for (size_t i = 0; i < names_.size(); i++) {
            close_[i].tag = "</" + std::string(names_[i]) + ">";
        }
    }

    std::optional<Found> next() {
        while (pos_ < text_.size()) {
            const size_t open = text_.find('<', pos_);
            if (open == std::string_view::npos) {
                break;
            }
            pos_ = open + 1;
            for (size_t i = 0; i < names_.size(); i++) {
                const size_t name_end = open + 1 + names_[i].size();
                if (!at_ci(text_, open + 1, names_[i]) || name_end >= text_.size() ||
                    !(is_space(text_[name_end]) || text_[name_end] == '>' ||
                      text_[name_end] == '/')) {
                    continue;
                }
                const size_t gt = next_gt(name_end);
                if (gt == std::string_view::npos) {
                    pos_ = text_.size();  // no '>' anywhere further: nothing opens
                    return std::nullopt;
                }
                const size_t close = next_close(i, gt + 1);
                if (close == std::string_view::npos) {
                    break;  // never closed: not an element, keep looking
                }
                pos_ = close + close_[i].tag.size();
                return Found{names_[i], text_.substr(gt + 1, close - gt - 1)};
            }
        }
        return std::nullopt;
    }

  private:
    // A search result that later searches from further on can reuse.
    struct Cached {
        std::string tag;
        bool searched = false;
        size_t from = 0;
        size_t at = std::string_view::npos;
    };

    size_t next_gt(size_t from) {
        if (!gt_.searched || (gt_.at == std::string_view::npos ? from < gt_.from : gt_.at < from)) {
            gt_.at = text_.find('>', from);
            gt_.from = from;
            gt_.searched = true;
        }
        return gt_.at;
    }

    size_t next_close(size_t name, size_t from) {
        Cached& cached = close_[name];
        if (!cached.searched ||
            (cached.at == std::string_view::npos ? from < cached.from : cached.at < from)) {
            cached.at = find_ci(text_, cached.tag, from);
            cached.from = from;
            cached.searched = true;
        }
        return cached.at;
    }

    std::string_view text_;
    std::vector<std::string_view> names_;
    std::vector<Cached> close_;
    Cached gt_;
    size_t pos_ = 0;
};

std::string strip_tags(std::string_view html) {
    // A line break is whitespace, not nothing: stripping <br> bare fuses
    // the words around it ("Hello<br/>World" -> "HelloWorld"). Replace it
    // with a space before the generic tag strip, like docling's
    // chandra_utils._strip_tags — lowercase only, so <BR> still strips
    // bare. (<br\s*/?> in docling's terms.)
    std::string spaced;
    spaced.reserve(html.size());
    for (size_t pos = 0;;) {
        const size_t at = html.find("<br", pos);
        if (at == std::string_view::npos) {
            spaced.append(html.substr(pos));
            break;
        }
        size_t i = at + 3;
        while (i < html.size() && is_space(html[i])) {
            i++;
        }
        if (i < html.size() && html[i] == '/') {
            i++;
        }
        if (i < html.size() && html[i] == '>') {
            spaced.append(html.substr(pos, at - pos));
            spaced += ' ';
            pos = i + 1;
        } else {
            spaced.append(html.substr(pos, at + 1 - pos));
            pos = at + 1;
        }
    }
    // Then every tag comes off: a '<' through the next '>'. A '<' with no
    // '>' after it is text.
    std::string text;
    text.reserve(spaced.size());
    for (size_t pos = 0;;) {
        const size_t open = spaced.find('<', pos);
        const size_t close =
            open == std::string::npos ? std::string::npos : spaced.find('>', open + 1);
        if (close == std::string::npos) {
            text.append(spaced, pos);
            break;
        }
        text.append(spaced, pos, open - pos);
        pos = close + 1;
    }
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
    // Collapse whitespace runs to one space, then trim (docling ends with
    // re.sub(r"\s+", " ", ...).strip()); after the entity pass so &nbsp;
    // participates.
    std::string collapsed;
    collapsed.reserve(text.size());
    bool in_space = false;
    for (const char c : text) {
        if (is_space(c)) {
            if (!in_space) {
                collapsed += ' ';
            }
            in_space = true;
        } else {
            collapsed += c;
            in_space = false;
        }
    }
    return trim(collapsed);
}

// Rows and cells of a table body, in source order. A row whose cells are
// all <th> is a header row; the leading run of those becomes the table's
// column headers (*header_rows). Cell text is stripped like any other
// block body. Cells outside a <tr> are read as one row, so the shapes
// models actually emit still produce a grid.
std::vector<std::vector<std::string>> parse_table_rows(std::string_view table_html,
                                                       size_t* header_rows) {
    auto cells_of = [](std::string_view row_html, bool* all_headers) {
        std::vector<std::string> cells;
        *all_headers = false;
        bool headers_so_far = true;
        ElementScanner scanner(row_html, {"th", "td"});
        while (const std::optional<Found> cell = scanner.next()) {
            headers_so_far = headers_so_far && cell->name == "th";
            cells.push_back(strip_tags(cell->content));
        }
        *all_headers = !cells.empty() && headers_so_far;
        return cells;
    };

    std::vector<std::vector<std::string>> rows;
    std::vector<bool> header_flags;
    ElementScanner row_scanner(table_html, {"tr"});
    while (const std::optional<Found> row = row_scanner.next()) {
        bool all_headers = false;
        std::vector<std::string> cells = cells_of(row->content, &all_headers);
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
              std::string* error, std::vector<vlmv1::PageWarning>* warnings) {
    // Block-level constructs only — this is a snippet mapper, not an HTML
    // parser; anything richer belongs to the HTML collector upstream. A
    // block's content may span lines: model output wraps tables and
    // paragraphs across newlines.
    size_t items = 0;
    ElementScanner blocks(text,
                          {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "table"});
    while (const std::optional<Found> block = blocks.next()) {
        const std::string_view tag = block->name;
        const std::string body = strip_tags(block->content);
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
                parse_table_rows(block->content, &header_rows);
            if (rows.empty() && !body.empty()) {
                rows.push_back({body});
            }
            const TableCut cut = fill_table_data(table->mutable_data(), rows, header_rows);
            note_table_cut(cut, table->data(), "table",
                           body_child_ref(BodyChild::TABLE, out->tables_size() - 1), warnings);
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
