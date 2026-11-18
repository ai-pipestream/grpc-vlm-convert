#include "nemotron_multipass.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <set>
#include <utility>

#include "builder.h"
#include "sequence_ratio.h"

namespace vlm::mapping {
namespace {

bool structural_label(const std::string& label) { return label == "Table" || label == "Chart"; }

double intersection(const NemotronBlock& a, const NemotronBlock& b) {
    return std::max(0.0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
           std::max(0.0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
}

double iou(const NemotronBlock& a, const NemotronBlock& b) {
    const double shared = intersection(a, b);
    const double area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1);
    return area > shared ? shared / (area - shared) : 0.0;
}

// Braces balance, with a backslash escaping the next character other than
// a newline (Python's `\\.` without re.S).
bool balanced_braces(const std::string& text) {
    int depth = 0;
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == '\\' && i + 1 < text.size() && text[i + 1] != '\n') {
            i++;
        } else if (text[i] == '{') {
            depth++;
        } else if (text[i] == '}' && --depth < 0) {
            return false;
        }
    }
    return depth == 0;
}

size_t count_of(const std::string& text, std::string_view token) {
    size_t count = 0;
    for (size_t at = text.find(token); at != std::string::npos; at = text.find(token, at + 1)) {
        count++;
    }
    return count;
}

// The column spec and body of the one `\begin{tabular}{SPEC}BODY\end{tabular}`.
std::optional<std::pair<std::string, std::string>> single_tabular(const std::string& text) {
    constexpr std::string_view kBegin = "\\begin{tabular}";
    constexpr std::string_view kEnd = "\\end{tabular}";
    if (count_of(text, kBegin) != 1 || count_of(text, kEnd) != 1) {
        return std::nullopt;
    }
    size_t at = text.find(kBegin) + kBegin.size();
    if (at >= text.size() || text[at] != '{') {
        return std::nullopt;
    }
    const size_t close = text.find('}', at + 1);
    if (close == std::string::npos || close == at + 1) {
        return std::nullopt;
    }
    std::string spec = text.substr(at + 1, close - at - 1);
    for (const char c : spec) {
        // `[clrx|\s]`: ASCII whitespace plus the separators \x1c-\x1f.
        if (std::string_view("clrx| \t\n\r\v\f\x1c\x1d\x1e\x1f").find(c) == std::string_view::npos) {
            return std::nullopt;
        }
    }
    const size_t end = text.find(kEnd, close + 1);
    if (end == std::string::npos) {
        return std::nullopt;
    }
    return std::pair{std::move(spec), text.substr(close + 1, end - close - 1)};
}

// Every `\multirow{N}` / `\multicolumn{N}` count in a raw cell is a
// positive integer.
bool span_counts_valid(const std::string& raw) {
    for (const std::string_view command : {"\\multirow{", "\\multicolumn{"}) {
        for (size_t at = raw.find(command); at != std::string::npos;
             at = raw.find(command, at + 1)) {
            const size_t start = at + command.size();
            const size_t close = raw.find_first_of("{}", start);
            if (close == std::string::npos || raw[close] != '}') {
                continue;  // not `{[^{}]*}`: the pattern does not match here
            }
            const std::string count = trim(raw.substr(start, close - start));
            if (count.empty() || !std::all_of(count.begin(), count.end(), [](char c) {
                    return std::isdigit(static_cast<unsigned char>(c)) != 0;
                })) {
                return false;
            }
            if (count.find_first_not_of('0') == std::string::npos) {
                return false;  // zero
            }
        }
    }
    return true;
}

// The length of the line break at `at`, 0 when there is none: Python's
// str.splitlines set (\r\n, \n, \r, \v, \f, \x1c-\x1e, U+0085, U+2028,
// U+2029).
size_t line_break(const std::string& text, size_t at) {
    const std::string_view rest = std::string_view(text).substr(at);
    for (const std::string_view wide : {"\r\n", "\xC2\x85", "\xE2\x80\xA8", "\xE2\x80\xA9"}) {
        if (rest.starts_with(wide)) {
            return wide.size();
        }
    }
    return std::string_view("\n\r\v\f\x1c\x1d\x1e").find(rest[0]) != std::string_view::npos ? 1 : 0;
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::string line;
    for (size_t i = 0; i < text.size();) {
        if (const size_t length = line_break(text, i); length > 0) {
            lines.push_back(std::move(line));
            line.clear();
            i += length;
        } else {
            line.push_back(text[i++]);
        }
    }
    if (!line.empty()) {
        lines.push_back(std::move(line));
    }
    return lines;
}

bool valid_chart(const std::string& text) {
    // `<chart\b[^>]*>` matches, found left to right without overlap.
    size_t headers = 0;
    for (size_t at = text.find("<chart"); at != std::string::npos;) {
        const size_t after = at + 6;
        const bool boundary =
            after >= text.size() ||
            !(std::isalnum(static_cast<unsigned char>(text[after])) != 0 || text[after] == '_');
        const size_t close = boundary ? text.find('>', after) : std::string::npos;
        if (close == std::string::npos) {
            at = text.find("<chart", at + 1);
            continue;
        }
        headers++;
        at = text.find("<chart", close + 1);
    }
    if (headers != 1 || count_of(text, "</chart>") != 1) {
        return false;
    }
    std::vector<size_t> widths;
    for (const std::string& line : split_lines(text)) {
        const std::string row = trim(line);
        if (!row.starts_with("|")) {
            continue;
        }
        const size_t first = row.find_first_not_of('|');
        const size_t last = row.find_last_not_of('|');
        const std::string inner =
            first == std::string::npos ? std::string() : row.substr(first, last - first + 1);
        widths.push_back(static_cast<size_t>(std::count(inner.begin(), inner.end(), '|')) + 1);
    }
    return widths.size() >= 3 && std::all_of(widths.begin(), widths.end(),
                                             [&](size_t w) { return w == widths[0]; }) &&
           widths[0] >= 2;
}

std::string serialize_blocks(const std::vector<NemotronBlock>& blocks) {
    std::string out;
    char coords[96];
    for (size_t i = 0; i < blocks.size(); i++) {
        const NemotronBlock& b = blocks[i];
        if (i > 0) {
            out.push_back('\n');
        }
        std::snprintf(coords, sizeof(coords), "<x_%.8f><y_%.8f>", b.x1, b.y1);
        out += coords;
        out += b.text;
        std::snprintf(coords, sizeof(coords), "<x_%.8f><y_%.8f>", b.x2, b.y2);
        out += coords;
        out += "<class_" + b.label + ">";
    }
    return out;
}

}  // namespace

std::vector<NemotronBlock> nemotron_structural_regions(std::string_view content) {
    std::vector<NemotronBlock> regions;
    for (NemotronBlock& block : nemotron_v35_blocks(content)) {
        if (structural_label(block.label) && block.x1 >= 0 && block.x1 < block.x2 &&
            block.x2 <= 1 && block.y1 >= 0 && block.y1 < block.y2 && block.y2 <= 1) {
            regions.push_back(std::move(block));
        }
    }
    return regions;
}

std::vector<NemotronCrop> plan_nemotron_crops(int64_t width, int64_t height,
                                              std::string_view parser,
                                              const std::string* detection,
                                              bool union_regions) {
    std::vector<NemotronBlock> regions;
    for (NemotronBlock& region :
         nemotron_structural_regions(detection != nullptr ? *detection : parser)) {
        if (std::none_of(regions.begin(), regions.end(), [&](const NemotronBlock& other) {
                return region.label == other.label && iou(region, other) >= 0.85;
            })) {
            regions.push_back(std::move(region));
        }
    }
    if (detection != nullptr && union_regions) {
        for (NemotronBlock& region : nemotron_structural_regions(parser)) {
            if (std::none_of(regions.begin(), regions.end(), [&](const NemotronBlock& other) {
                    return region.label == other.label && iou(region, other) >= 0.5;
                })) {
                regions.push_back(std::move(region));
            }
        }
    }
    if (regions.size() > kMaxNemotronCrops) {
        regions.resize(kMaxNemotronCrops);
    }
    const auto w = static_cast<double>(width);
    const auto h = static_cast<double>(height);
    std::vector<NemotronCrop> crops;
    for (NemotronBlock& region : regions) {
        const double mx = 12 + (region.x2 - region.x1) * w * 0.04;
        const double my = 12 + (region.y2 - region.y1) * h * 0.04;
        NemotronCrop crop;
        crop.left = std::max<int64_t>(0, static_cast<int64_t>(std::floor(region.x1 * w - mx)));
        crop.top = std::max<int64_t>(0, static_cast<int64_t>(std::floor(region.y1 * h - my)));
        crop.right = std::min<int64_t>(width, static_cast<int64_t>(std::ceil(region.x2 * w + mx)));
        crop.bottom =
            std::min<int64_t>(height, static_cast<int64_t>(std::ceil(region.y2 * h + my)));
        if (crop.right - crop.left < 8 || crop.bottom - crop.top < 8) {
            continue;
        }
        crop.x1 = static_cast<double>(crop.left) / w;
        crop.y1 = static_cast<double>(crop.top) / h;
        crop.x2 = static_cast<double>(crop.right) / w;
        crop.y2 = static_cast<double>(crop.bottom) / h;
        crop.region = std::move(region);
        crops.push_back(std::move(crop));
    }
    return crops;
}

std::optional<size_t> nemotron_table_columns(const std::string& text) {
    if (!balanced_braces(text)) {
        return std::nullopt;
    }
    const auto tabular = single_tabular(text);
    if (!tabular) {
        return std::nullopt;
    }
    const auto& [spec, body] = *tabular;
    const auto columns = static_cast<size_t>(
        std::count_if(spec.begin(), spec.end(),
                      [](char c) { return std::string_view("clrx").find(c) != std::string_view::npos; }));
    std::vector<std::string> rows;
    for (std::string& row : split_top_level(body, "\\\\")) {
        if (!trim(row).empty()) {
            rows.push_back(std::move(row));
        }
    }
    if (rows.empty() || rows.size() > 4096 || columns == 0 || columns > 512) {
        return std::nullopt;
    }
    std::set<std::pair<size_t, size_t>> occupied;
    for (size_t row_index = 0; row_index < rows.size(); row_index++) {
        size_t column = 0;
        for (std::string& raw : split_top_level(rows[row_index], "&")) {
            if (!span_counts_valid(raw)) {
                return std::nullopt;
            }
            const LatexCell cell = parse_latex_cell(std::move(raw));
            if (cell.text.find("\\multirow") != std::string::npos ||
                cell.text.find("\\multicolumn") != std::string::npos) {
                return std::nullopt;
            }
            if (row_index + cell.row_span > rows.size() || column + cell.col_span > columns) {
                return std::nullopt;
            }
            bool overlaps = false;
            bool covered = true;
            for (size_t r = row_index; r < row_index + cell.row_span; r++) {
                for (size_t c = column; c < column + cell.col_span; c++) {
                    const bool taken = occupied.contains({r, c});
                    overlaps = overlaps || taken;
                    covered = covered && taken;
                }
            }
            if (overlaps) {
                // Only an empty placeholder under a \multirow may sit on a
                // taken cell, and only exactly on it.
                if ((cell.text != "" && cell.text != "\\mbox{}") || cell.header ||
                    cell.row_span != 1 || !covered) {
                    return std::nullopt;
                }
            } else {
                for (size_t r = row_index; r < row_index + cell.row_span; r++) {
                    for (size_t c = column; c < column + cell.col_span; c++) {
                        occupied.insert({r, c});
                    }
                }
            }
            column += cell.col_span;
        }
        if (column != columns) {
            return std::nullopt;
        }
    }
    return columns;
}

bool nemotron_valid_structure(const NemotronBlock& block) {
    if (block.label == "Table") {
        return nemotron_table_columns(block.text).has_value();
    }
    return valid_chart(block.text);
}

std::vector<NemotronBlock> remap_nemotron_crop(std::string_view content,
                                               const NemotronCrop& crop) {
    std::vector<NemotronBlock> candidates;
    const double width = crop.x2 - crop.x1;
    const double height = crop.y2 - crop.y1;
    for (NemotronBlock& block : nemotron_structural_regions(content)) {
        NemotronBlock mapped{block.label,
                             crop.x1 + block.x1 * width,
                             crop.y1 + block.y1 * height,
                             crop.x1 + block.x2 * width,
                             crop.y1 + block.y2 * height,
                             std::move(block.text)};
        if (mapped.label == crop.region.label && iou(mapped, crop.region) >= 0.5 &&
            nemotron_valid_structure(mapped)) {
            candidates.push_back(std::move(mapped));
        }
    }
    return candidates;
}

std::string merge_nemotron_crops(const std::string& content,
                                 const std::vector<NemotronBlock>& candidates) {
    std::vector<NemotronBlock> blocks = nemotron_v35_blocks(content);
    if (blocks.empty() || candidates.empty()) {
        return content;
    }
    std::vector<bool> used(candidates.size(), false);
    bool changed = false;
    for (NemotronBlock& original : blocks) {
        const std::optional<size_t> old_columns =
            original.label == "Table" ? nemotron_table_columns(original.text) : std::nullopt;
        // The best candidate by overlap, the later one on a tie.
        std::optional<std::pair<double, size_t>> best;
        for (size_t index = 0; index < candidates.size(); index++) {
            const NemotronBlock& candidate = candidates[index];
            if (used[index] || candidate.label != original.label ||
                !nemotron_valid_structure(candidate)) {
                continue;
            }
            const double overlap = iou(candidate, original);
            if (overlap < 0.5) {
                continue;
            }
            if (old_columns && old_columns != nemotron_table_columns(candidate.text)) {
                continue;
            }
            if (sequence_ratio(original.text, candidate.text) < 0.65) {
                continue;
            }
            if (!best || std::pair{overlap, index} > *best) {
                best = std::pair{overlap, index};
            }
        }
        if (best) {
            original.text = candidates[best->second].text;
            used[best->second] = true;
            changed = true;
        }
    }
    for (size_t index = 0; index < candidates.size(); index++) {
        const NemotronBlock& candidate = candidates[index];
        if (used[index] || !nemotron_valid_structure(candidate)) {
            continue;
        }
        if (std::any_of(blocks.begin(), blocks.end(), [&](const NemotronBlock& block) {
                return candidate.label == block.label && intersection(candidate, block) > 0;
            })) {
            continue;
        }
        // A new detector object goes before the first block below or right
        // of it; the page blocks keep their order.
        const auto position = std::find_if(blocks.begin(), blocks.end(), [&](const NemotronBlock& b) {
            return std::pair{b.y1, b.x1} > std::pair{candidate.y1, candidate.x1};
        });
        blocks.insert(position, candidate);
        changed = true;
    }
    return changed ? serialize_blocks(blocks) : content;
}

}  // namespace vlm::mapping
