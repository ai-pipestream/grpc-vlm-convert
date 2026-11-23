#pragma once

// Shared syntax of Nemotron Parse native output (docling
// nemotron_parse_utils.py and nemotron_parse_v3_5_utils.py). Blocks are
//   <x_X1><y_Y1>BODY<x_X2><y_Y2><class_CLASS>
// with coordinates normalized to [0, 1]; table bodies are LaTeX tabulars.

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vlm::mapping {

// Widest LaTeX table read, as chandra's _MAX_TABLE_GRID_CELLS.
constexpr size_t kMaxLatexColumns = 1000;

// `<x_N><y_N>` at `i` (N is \d+(?:\.\d+)?); the end offset and values.
std::optional<size_t> coord_token(std::string_view text, size_t i, double* x, double* y);

// `<class_NAME>` at `i` (NAME is [^>]+); the end offset and the name.
std::optional<size_t> class_token(std::string_view text, size_t i, std::string* name);

// The text with docling's generation artifacts (<tbc>, \<|unk|\>,
// \unknown) removed.
std::string remove_artifacts(std::string text);

// The brace group starting at `pos` after optional whitespace: its content
// and the offset after the closing brace. Backslash escapes one character.
std::optional<std::pair<std::string, size_t>> read_group(const std::string& text, size_t pos);

// `text` split on `separator` outside brace groups. A stray closing brace
// never takes the depth below zero.
std::vector<std::string> split_top_level(const std::string& text, std::string_view separator);

// One LaTeX tabular cell with its \multicolumn, \multirow and \thead
// wrappers read off; text is escaped for HTML except docling's inline tags.
struct LatexCell {
    std::string text;
    size_t row_span = 1;
    size_t col_span = 1;
    bool header = false;
};

LatexCell parse_latex_cell(std::string text);

// One Nemotron Parse 3.5 block: its class, normalized corners, and body
// with stray coordinate tokens and artifacts removed and trimmed.
struct NemotronBlock {
    std::string label;
    double x1, y1, x2, y2;
    std::string text;
    bool operator<(const NemotronBlock& o) const;
};

// What nemotron_v35_blocks read past without building a block from it.
struct NemotronUnread {
    // Class tokens whose segment holds text but fewer than two coordinate
    // tokens: blocks with no corners.
    size_t blocks_without_corners = 0;
    // Bytes of text after the last class token, coordinate tokens and
    // artifacts not counted: a block the answer broke off inside.
    size_t trailing_bytes = 0;
    bool any() const { return blocks_without_corners > 0 || trailing_bytes > 0; }
};

// Every well-formed block in output order, repeats dropped. `unread`, when
// given, receives what was not one.
std::vector<NemotronBlock> nemotron_v35_blocks(std::string_view content,
                                               NemotronUnread* unread = nullptr);

}  // namespace vlm::mapping
