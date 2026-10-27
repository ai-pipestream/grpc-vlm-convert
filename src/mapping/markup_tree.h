#pragma once

// A tolerant element tree for model markup: Chandra's HTML dialect and
// DocLang XML. Models emit truncated and unbalanced markup, so the parser
// never fails: an unclosed element ends at the end of input, a stray
// closing tag is dropped, and nesting deeper than kMaxMarkupDepth is
// flattened onto the deepest open element (the walkers recurse, and a
// page of 100k nested <div>s must not take the stack down). No DTDs, no
// external entities, no resource fetching: a <!DOCTYPE> or <!ENTITY> is
// skipped as a comment, and only the five XML entities, &nbsp; and
// numeric references are decoded.

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vlm::mapping {

constexpr size_t kMaxMarkupDepth = 128;

struct MarkupNode {
    // Empty for a text node; otherwise the tag name (lowercase in HTML
    // mode, as written in XML mode).
    std::string tag;
    // Decoded character data of a text node.
    std::string text;
    std::vector<std::pair<std::string, std::string>> attrs;
    std::vector<std::unique_ptr<MarkupNode>> children;

    bool is_text() const { return tag.empty(); }
    bool has_attr(std::string_view name) const;
    // The attribute's value, or `fallback` when absent.
    std::string attr(std::string_view name, std::string_view fallback = "") const;
};

enum class MarkupMode {
    // Tag and attribute names fold to lowercase; void elements (<br>,
    // <img>, ...) never open; <td>/<th>/<tr>/<li>/<p> close an open
    // sibling the way Chandra's tree builder does.
    kHtml,
    // Names keep their case; only "<x/>" is empty.
    kXml,
};

// Parses `text` into a synthetic root (tag "root") holding the top-level
// nodes.
std::unique_ptr<MarkupNode> parse_markup(std::string_view text, MarkupMode mode);

// Appends `codepoint` to `out` as UTF-8.
void append_utf8(std::string* out, unsigned codepoint);

// Decodes the entities parse_markup decodes; an unknown one stays as is.
std::string decode_entities(std::string_view text);

}  // namespace vlm::mapping
