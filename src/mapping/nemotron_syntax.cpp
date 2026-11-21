// Shared syntax of Nemotron Parse native output (docling
// nemotron_parse_v3_5_utils.py): coordinate and class tokens, the 3.5 block
// split, and LaTeX tabular cells. The page mapper and the multi-pass merge
// both read model output through these.

#include "nemotron_syntax.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <set>
#include <tuple>

#include "builder.h"

namespace vlm::mapping {

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

// ---- LaTeX tabular cells ----------------------------------------------------

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

namespace {

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

}  // namespace

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

// ---- Nemotron Parse 3.5 blocks ---------------------------------------------

bool NemotronBlock::operator<(const NemotronBlock& o) const {
    return std::tie(label, x1, y1, x2, y2, text) <
           std::tie(o.label, o.x1, o.y1, o.x2, o.y2, o.text);
}

std::vector<NemotronBlock> nemotron_v35_blocks(std::string_view content) {
    std::vector<NemotronBlock> blocks;
    std::set<NemotronBlock> seen;
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
        NemotronBlock block{label, first.x, first.y, last.x, last.y, trim(remove_artifacts(body))};
        if (seen.insert(block).second) {
            blocks.push_back(std::move(block));
        }
    }
    return blocks;
}
}  // namespace vlm::mapping
