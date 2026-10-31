// Inline Markdown for the dots.ocr and Nemotron Parse 3.5 mappers
// (docling dots_utils _markdown_inline_html): emphasis becomes inline
// HTML for the Chandra builder; math, code spans and inline HTML pass
// through. Docling renders with marko restricted to paragraphs; this is
// the CommonMark emphasis algorithm over the same restricted grammar.

#include <cctype>
#include <map>
#include <optional>
#include <tuple>

#include "layout.h"

namespace vlm::mapping {

namespace {

bool md_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool md_punct(unsigned char c) {
    return c < 0x80 && std::ispunct(c) != 0;
}

void escape_into(std::string_view text, std::string* out) {
    for (char c : text) {
        switch (c) {
            case '&':
                *out += "&amp;";
                break;
            case '<':
                *out += "&lt;";
                break;
            case '>':
                *out += "&gt;";
                break;
            case '"':
                *out += "&quot;";
                break;
            default:
                out->push_back(c);
        }
    }
}

// The end of a literal span starting at `i` (math, an inline <code>
// element), or npos. Mirrors _LITERAL_INLINE_PATTERN.
size_t literal_end(std::string_view text, size_t i) {
    auto unescaped_find = [&](std::string_view needle, size_t from) -> size_t {
        for (size_t at = text.find(needle, from); at != std::string_view::npos;
             at = text.find(needle, at + 1)) {
            if (at == 0 || text[at - 1] != '\\') {
                return at;
            }
        }
        return std::string_view::npos;
    };
    if (i > 0 && text[i - 1] == '\\') {
        return std::string_view::npos;
    }
    if (text.substr(i).starts_with("$$")) {
        const size_t close = unescaped_find("$$", i + 2);
        return close == std::string_view::npos ? close : close + 2;
    }
    if (text[i] == '$') {
        for (size_t j = i + 1; j < text.size() && text[j] != '\n'; j++) {
            if (text[j] == '$') {
                if (j == i + 1) {
                    return std::string_view::npos;  // `[^\n$]+?` needs a character
                }
                if (text[j - 1] != '\\') {
                    return j + 1;
                }
            }
        }
        return std::string_view::npos;
    }
    if (text.substr(i).starts_with("\\(") || text.substr(i).starts_with("\\[")) {
        const std::string_view close = text[i + 1] == '(' ? "\\)" : "\\]";
        const size_t at = text.find(close, i + 2);
        return at == std::string_view::npos ? at : at + 2;
    }
    if (text.size() - i >= 5) {
        std::string head(text.substr(i, 5));
        for (char& c : head) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (head == "<code" &&
            (i + 5 == text.size() || !std::isalnum(static_cast<unsigned char>(text[i + 5])))) {
            std::string rest(text.substr(i));
            for (char& c : rest) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            const size_t gt = rest.find('>');
            const size_t close = gt == std::string::npos ? gt : rest.find("</code>", gt);
            return close == std::string::npos ? std::string_view::npos : i + close + 7;
        }
    }
    return std::string_view::npos;
}

// An inline HTML tag at `i` (CommonMark raw HTML, simplified to a tag name
// and a quoted-attribute-aware scan to '>').
size_t html_tag_end(std::string_view text, size_t i) {
    if (text[i] != '<' || i + 1 >= text.size()) {
        return std::string_view::npos;
    }
    size_t j = i + 1;
    if (text[j] == '/') {
        j++;
    }
    if (j >= text.size() || std::isalpha(static_cast<unsigned char>(text[j])) == 0) {
        return std::string_view::npos;
    }
    char quote = 0;
    for (; j < text.size(); j++) {
        const char c = text[j];
        if (quote != 0) {
            quote = c == quote ? 0 : quote;
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '>') {
            return j + 1;
        } else if (c == '<' || c == '\n') {
            return std::string_view::npos;
        }
    }
    return std::string_view::npos;
}

struct Piece {
    std::string html;  // text and raw pieces
    bool delim = false;
    char ch = 0;
    size_t count = 0;
    size_t orig = 0;
    bool can_open = false;
    bool can_close = false;
    std::string open_tags;   // inserted on the content side, newest outermost
    std::string close_tags;  // likewise
};

std::string render_paragraph(std::string_view text) {
    std::vector<Piece> pieces;
    std::string buffer;
    auto flush = [&] {
        if (!buffer.empty()) {
            Piece piece;
            piece.html = buffer;
            pieces.push_back(std::move(piece));
            buffer.clear();
        }
    };
    size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (c == '$' || c == '\\' || c == '<') {
            const size_t end = literal_end(text, i);
            if (end != std::string_view::npos) {
                const std::string_view literal = text.substr(i, end - i);
                if (literal.front() == '<') {
                    buffer += literal;  // <code> elements are kept as HTML
                } else {
                    escape_into(literal, &buffer);
                }
                i = end;
                continue;
            }
        }
        if (c == '`') {
            size_t run = 0;
            while (i + run < text.size() && text[i + run] == '`') {
                run++;
            }
            // A closing run of exactly the same length.
            size_t at = i + run;
            size_t close = std::string_view::npos;
            while (at < text.size()) {
                const size_t found = text.find('`', at);
                if (found == std::string_view::npos) {
                    break;
                }
                size_t len = 0;
                while (found + len < text.size() && text[found + len] == '`') {
                    len++;
                }
                if (len == run) {
                    close = found;
                    break;
                }
                at = found + len;
            }
            if (close != std::string_view::npos) {
                std::string code(text.substr(i + run, close - i - run));
                for (char& ch : code) {
                    ch = ch == '\n' ? ' ' : ch;
                }
                if (code.size() >= 2 && code.front() == ' ' && code.back() == ' ' &&
                    code.find_first_not_of(' ') != std::string::npos) {
                    code = code.substr(1, code.size() - 2);
                }
                buffer += "<code>";
                escape_into(code, &buffer);
                buffer += "</code>";
                i = close + run;
                continue;
            }
            buffer.append(run, '`');
            i += run;
            continue;
        }
        if (c == '\\' && i + 1 < text.size()) {
            const unsigned char next = static_cast<unsigned char>(text[i + 1]);
            if (next == '\n') {
                buffer += "<br />\n";
                i += 2;
                continue;
            }
            if (md_punct(next)) {
                escape_into(text.substr(i + 1, 1), &buffer);
                i += 2;
                continue;
            }
        }
        if (c == '<') {
            const size_t end = html_tag_end(text, i);
            if (end != std::string_view::npos) {
                buffer += text.substr(i, end - i);
                i = end;
                continue;
            }
        }
        if (c == '*' || c == '_') {
            size_t run = 0;
            while (i + run < text.size() && text[i + run] == c) {
                run++;
            }
            const unsigned char before = i == 0 ? ' ' : static_cast<unsigned char>(text[i - 1]);
            const unsigned char after =
                i + run >= text.size() ? ' ' : static_cast<unsigned char>(text[i + run]);
            const bool left = !md_space(after) && (!md_punct(after) || md_space(before) ||
                                                   md_punct(before));
            const bool right = !md_space(before) && (!md_punct(before) || md_space(after) ||
                                                     md_punct(after));
            flush();
            Piece piece;
            piece.delim = true;
            piece.ch = c;
            piece.count = run;
            piece.orig = run;
            if (c == '*') {
                piece.can_open = left;
                piece.can_close = right;
            } else {
                piece.can_open = left && (!right || md_punct(before));
                piece.can_close = right && (!left || md_punct(after));
            }
            pieces.push_back(std::move(piece));
            i += run;
            continue;
        }
        if (c == '\n') {
            // Two trailing spaces make a hard break; otherwise a soft one.
            size_t spaces = 0;
            while (!buffer.empty() && buffer.back() == ' ') {
                buffer.pop_back();
                spaces++;
            }
            buffer += spaces >= 2 ? "<br />\n" : "\n";
            i++;
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) {
                i++;
            }
            continue;
        }
        escape_into(text.substr(i, 1), &buffer);
        i++;
    }
    flush();

    // CommonMark "process emphasis": each closer looks back for the
    // nearest compatible opener; openers_bottom keeps the scan linear.
    std::vector<size_t> stack;
    std::map<std::tuple<char, bool, size_t>, size_t> bottom;
    for (size_t index = 0; index < pieces.size(); index++) {
        Piece& closer = pieces[index];
        if (!closer.delim) {
            continue;
        }
        if (closer.can_close) {
            const auto key = std::tuple{closer.ch, closer.can_open, closer.orig % 3};
            const size_t floor = bottom.contains(key) ? bottom[key] : 0;
            bool matched_any = false;
            while (closer.count > 0) {
                std::optional<size_t> found;
                for (size_t s = stack.size(); s > floor; s--) {
                    Piece& opener = pieces[stack[s - 1]];
                    if (opener.ch != closer.ch || !opener.can_open || opener.count == 0) {
                        continue;
                    }
                    const bool odd_match = (opener.can_close || closer.can_open) &&
                                           (opener.orig + closer.orig) % 3 == 0 &&
                                           !(opener.orig % 3 == 0 && closer.orig % 3 == 0);
                    if (!odd_match) {
                        found = s - 1;
                        break;
                    }
                }
                if (!found) {
                    break;
                }
                Piece& opener = pieces[stack[*found]];
                const size_t used = opener.count >= 2 && closer.count >= 2 ? 2 : 1;
                const char* open_tag = used == 2 ? "<strong>" : "<em>";
                const char* close_tag = used == 2 ? "</strong>" : "</em>";
                opener.open_tags = open_tag + opener.open_tags;
                closer.close_tags += close_tag;
                opener.count -= used;
                closer.count -= used;
                matched_any = true;
                // Delimiters between the pair become literal text.
                stack.resize(*found + 1);
                if (opener.count == 0) {
                    stack.pop_back();
                }
            }
            if (!matched_any) {
                bottom[key] = stack.size();
            }
        }
        if (closer.count > 0 && closer.can_open) {
            stack.push_back(index);
        }
    }
    std::string out;
    for (const Piece& piece : pieces) {
        if (!piece.delim) {
            out += piece.html;
            continue;
        }
        // Remaining delimiters sit outside the tags they took part in.
        out += piece.close_tags;
        out.append(piece.count, piece.ch);
        out += piece.open_tags;
    }
    return out;
}

}  // namespace

std::string markdown_inline_html(const std::string& text) {
    if (text.find('*') == std::string::npos && text.find('_') == std::string::npos) {
        return text;
    }
    // Paragraphs only: blank lines split them, nothing else is a block.
    std::string rendered;
    size_t start = 0;
    std::string paragraph;
    auto emit = [&] {
        const std::string body = trim(paragraph);
        if (!body.empty()) {
            rendered += "<p>" + render_paragraph(body) + "</p>\n";
        }
        paragraph.clear();
    };
    while (start <= text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string line = text.substr(start, end - start);
        if (trim(line).empty()) {
            emit();
        } else {
            if (!paragraph.empty()) {
                paragraph.push_back('\n');
            }
            paragraph += line;
        }
        if (end == text.size()) {
            break;
        }
        start = end + 1;
    }
    emit();
    if (rendered.find("<strong>") == std::string::npos &&
        rendered.find("<em>") == std::string::npos) {
        return unescape_markdown_literals(text);
    }
    std::string html = trim(rendered);
    std::string out;
    for (size_t i = 0; i < html.size(); i++) {
        if (html.compare(i, 7, "<br />\n") == 0) {
            out += "<br>";
            i += 6;
        } else if (html[i] == '\n') {
            out += "<br>";
        } else {
            out.push_back(html[i]);
        }
    }
    return out;
}

bool has_inline_html_tag(const std::string& html) {
    static constexpr const char* kTags[] = {"a",     "b",     "br",    "code", "del", "div",
                                            "em",    "i",     "p",     "s",    "span", "strong",
                                            "sub",   "sup",   "table", "tbody", "td",  "tfoot",
                                            "th",    "thead", "tr",    "u"};
    for (size_t at = html.find('<'); at != std::string::npos; at = html.find('<', at + 1)) {
        size_t i = at + 1;
        if (i < html.size() && html[i] == '/') {
            i++;
        }
        size_t j = i;
        while (j < html.size() && std::isalpha(static_cast<unsigned char>(html[j])) != 0) {
            j++;
        }
        if (j == i || j >= html.size()) {
            continue;
        }
        const char follow = html[j];
        if (!(md_space(static_cast<unsigned char>(follow)) || follow == '/' || follow == '>')) {
            continue;
        }
        std::string name = html.substr(i, j - i);
        for (char& c : name) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        for (const char* tag : kTags) {
            if (name == tag) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace vlm::mapping
