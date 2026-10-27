#include "markup_tree.h"

#include <algorithm>
#include <array>
#include <cctype>

namespace vlm::mapping {

namespace {

char lower(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

constexpr size_t kMaxTagBytes = 64 * 1024;

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

bool is_name_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '_' ||
           c == ':' || c == '.' || (static_cast<unsigned char>(c) >= 0x80);
}

constexpr std::array<std::string_view, 14> kVoidTags = {
    "area", "base", "br",   "col",   "embed",  "hr",    "img",
    "input", "link", "meta", "param", "source", "track", "wbr"};

constexpr std::array<std::string_view, 18> kHtmlBlockTags = {
    "div", "p",  "h1", "h2",  "h3",     "h4",   "h5",      "h6",         "table",
    "ul",  "ol", "li", "pre", "figure", "form", "caption", "figcaption", "hr"};

bool contains(const auto& list, std::string_view name) {
    return std::ranges::find(list, name) != list.end();
}

std::string fold(std::string_view name, MarkupMode mode) {
    std::string out(name);
    if (mode == MarkupMode::kHtml) {
        std::ranges::transform(out, out.begin(), lower);
    }
    return out;
}

struct Tag {
    std::string name;
    bool closing = false;
    bool self_closing = false;
    std::vector<std::pair<std::string, std::string>> attrs;
};

// Reads one tag starting at text[pos] == '<'. Returns false when the
// '<' does not open a tag (it is then character data). `end` is the
// position past the tag.
bool read_tag(std::string_view text, size_t pos, MarkupMode mode, Tag* tag, size_t* end) {
    size_t i = pos + 1;
    if (i < text.size() && text[i] == '/') {
        tag->closing = true;
        i++;
    }
    const size_t name_start = i;
    while (i < text.size() && is_name_char(text[i])) {
        i++;
    }
    if (i == name_start ||
        std::isalpha(static_cast<unsigned char>(text[name_start])) == 0) {
        return false;
    }
    tag->name = fold(text.substr(name_start, i - name_start), mode);
    while (i < text.size()) {
        while (i < text.size() && is_space(text[i])) {
            i++;
        }
        if (i >= text.size()) {
            break;
        }
        if (text[i] == '>') {
            *end = i + 1;
            return true;
        }
        if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '>') {
            tag->self_closing = true;
            *end = i + 2;
            return true;
        }
        const size_t attr_start = i;
        while (i < text.size() && !is_space(text[i]) && text[i] != '=' && text[i] != '>' &&
               !(text[i] == '/' && i + 1 < text.size() && text[i + 1] == '>')) {
            i++;
        }
        if (i == attr_start) {
            i++;  // a lone '/' or '=': skip it
            continue;
        }
        std::string name = fold(text.substr(attr_start, i - attr_start), mode);
        while (i < text.size() && is_space(text[i])) {
            i++;
        }
        std::string value;
        if (i < text.size() && text[i] == '=') {
            i++;
            while (i < text.size() && is_space(text[i])) {
                i++;
            }
            if (i < text.size() && (text[i] == '"' || text[i] == '\'')) {
                const char quote = text[i];
                const size_t close = text.find(quote, i + 1);
                if (close == std::string_view::npos) {
                    return false;  // an unterminated quote: not a tag
                }
                value = decode_entities(text.substr(i + 1, close - i - 1));
                i = close + 1;
            } else {
                const size_t value_start = i;
                while (i < text.size() && !is_space(text[i]) && text[i] != '>') {
                    i++;
                }
                value = decode_entities(text.substr(value_start, i - value_start));
            }
        }
        if (!tag->closing) {
            tag->attrs.emplace_back(std::move(name), std::move(value));
        }
    }
    return false;  // no '>' before the end: a truncated tag is text
}

class TreeBuilder {
  public:
    explicit TreeBuilder(MarkupMode mode) : mode_(mode), root_(std::make_unique<MarkupNode>()) {
        root_->tag = "root";
        stack_.push_back(root_.get());
    }

    void text(std::string_view raw) {
        if (raw.empty()) {
            return;
        }
        MarkupNode* top = stack_.back();
        if (!top->children.empty() && top->children.back()->is_text()) {
            top->children.back()->text += decode_entities(raw);
            return;
        }
        auto node = std::make_unique<MarkupNode>();
        node->text = decode_entities(raw);
        top->children.push_back(std::move(node));
    }

    void open(Tag& tag) {
        if (mode_ == MarkupMode::kHtml) {
            implicit_close(tag.name);
        }
        auto node = std::make_unique<MarkupNode>();
        node->tag = tag.name;
        node->attrs = std::move(tag.attrs);
        MarkupNode* raw = node.get();
        stack_.back()->children.push_back(std::move(node));
        const bool empty = tag.self_closing ||
                           (mode_ == MarkupMode::kHtml && contains(kVoidTags, tag.name));
        if (!empty && stack_.size() <= kMaxMarkupDepth) {
            stack_.push_back(raw);
        }
    }

    void close(const std::string& name) {
        for (size_t i = stack_.size() - 1; i > 0; i--) {
            if (stack_[i]->tag == name) {
                stack_.resize(i);
                return;
            }
        }
    }

    std::unique_ptr<MarkupNode> take() { return std::move(root_); }

  private:
    // Chandra's tree builder: a new cell closes an open cell up to its
    // row, a row closes a row up to its table, an item closes an item up
    // to its list, and a paragraph closes a paragraph up to any other
    // block.
    void implicit_close(const std::string& name) {
        auto close_up_to = [&](std::initializer_list<std::string_view> targets,
                               auto&& is_boundary) {
            for (size_t i = stack_.size() - 1; i > 0; i--) {
                if (std::ranges::find(targets, stack_[i]->tag) != targets.end()) {
                    stack_.resize(i);
                    return;
                }
                if (is_boundary(stack_[i]->tag)) {
                    return;
                }
            }
        };
        if (name == "td" || name == "th") {
            close_up_to({"td", "th"},
                        [](const std::string& tag) { return tag == "tr" || tag == "table"; });
        } else if (name == "tr") {
            close_up_to({"tr"}, [](const std::string& tag) { return tag == "table"; });
        } else if (name == "li") {
            close_up_to({"li"}, [](const std::string& tag) { return tag == "ul" || tag == "ol"; });
        } else if (name == "p") {
            close_up_to({"p"}, [](const std::string& tag) {
                return tag != "p" && contains(kHtmlBlockTags, tag);
            });
        }
    }

    MarkupMode mode_;
    std::unique_ptr<MarkupNode> root_;
    std::vector<MarkupNode*> stack_;
};

}  // namespace

bool MarkupNode::has_attr(std::string_view name) const {
    return std::ranges::any_of(attrs, [&](const auto& attr) { return attr.first == name; });
}

std::string MarkupNode::attr(std::string_view name, std::string_view fallback) const {
    for (const auto& [key, value] : attrs) {
        if (key == name) {
            return value;
        }
    }
    return std::string(fallback);
}

void append_utf8(std::string* out, unsigned cp) {
    if (cp == 0 || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        cp = 0xFFFD;
    }
    if (cp < 0x80) {
        out->push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string decode_entities(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] != '&') {
            out.push_back(text[i++]);
            continue;
        }
        const size_t semi = text.find(';', i + 1);
        if (semi == std::string_view::npos || semi - i > 10) {
            out.push_back(text[i++]);
            continue;
        }
        const std::string_view name = text.substr(i + 1, semi - i - 1);
        bool decoded = true;
        if (name == "amp") {
            out.push_back('&');
        } else if (name == "lt") {
            out.push_back('<');
        } else if (name == "gt") {
            out.push_back('>');
        } else if (name == "quot") {
            out.push_back('"');
        } else if (name == "apos") {
            out.push_back('\'');
        } else if (name == "nbsp") {
            append_utf8(&out, 0xA0);
        } else if (name.size() >= 2 && name[0] == '#') {
            const bool hex = name[1] == 'x' || name[1] == 'X';
            const std::string_view digits = name.substr(hex ? 2 : 1);
            unsigned long value = 0;
            bool ok = !digits.empty();
            for (char c : digits) {
                const int digit = std::isdigit(static_cast<unsigned char>(c)) != 0 ? c - '0'
                                  : hex && std::isxdigit(static_cast<unsigned char>(c)) != 0
                                      ? lower(c) - 'a' + 10
                                      : -1;
                if (digit < 0 || value > 0x10FFFF) {
                    ok = false;
                    break;
                }
                value = value * (hex ? 16 : 10) + static_cast<unsigned long>(digit);
            }
            if (ok) {
                append_utf8(&out, static_cast<unsigned>(value));
            } else {
                decoded = false;
            }
        } else {
            decoded = false;
        }
        if (decoded) {
            i = semi + 1;
        } else {
            out.push_back(text[i++]);
        }
    }
    return out;
}

std::unique_ptr<MarkupNode> parse_markup(std::string_view text, MarkupMode mode) {
    TreeBuilder builder(mode);
    size_t pos = 0;
    size_t text_start = 0;
    while (pos < text.size()) {
        const size_t open = text.find('<', pos);
        if (open == std::string_view::npos) {
            break;
        }
        // Comments, CDATA, doctype and processing instructions.
        if (text.substr(open, 4) == "<!--") {
            builder.text(text.substr(text_start, open - text_start));
            const size_t close = text.find("-->", open + 4);
            pos = text_start = close == std::string_view::npos ? text.size() : close + 3;
            continue;
        }
        if (text.substr(open, 9) == "<![CDATA[") {
            builder.text(text.substr(text_start, open - text_start));
            const size_t close = text.find("]]>", open + 9);
            const size_t data_end = close == std::string_view::npos ? text.size() : close;
            // CDATA is literal: bypass entity decoding by escaping '&'.
            std::string literal;
            for (char c : text.substr(open + 9, data_end - open - 9)) {
                literal += c == '&' ? std::string("&amp;") : std::string(1, c);
            }
            builder.text(literal);
            pos = text_start = close == std::string_view::npos ? text.size() : close + 3;
            continue;
        }
        if (open + 1 < text.size() && (text[open + 1] == '!' || text[open + 1] == '?')) {
            builder.text(text.substr(text_start, open - text_start));
            const size_t close = text.find('>', open + 2);
            pos = text_start = close == std::string_view::npos ? text.size() : close + 1;
            continue;
        }
        Tag tag;
        size_t end = 0;
        // A tag is read within a bounded window, so a page of unterminated
        // '<a x="' openings stays linear rather than rescanning to the end
        // for each one.
        const std::string_view window = text.substr(0, std::min(text.size(), open + kMaxTagBytes));
        if (!read_tag(window, open, mode, &tag, &end)) {
            pos = open + 1;
            continue;
        }
        builder.text(text.substr(text_start, open - text_start));
        if (tag.closing) {
            builder.close(tag.name);
        } else {
            const bool raw_text =
                mode == MarkupMode::kHtml && (tag.name == "script" || tag.name == "style");
            const std::string name = tag.name;
            builder.open(tag);
            if (raw_text) {
                // Script and style bodies are not markup; skip to the close.
                const std::string closer = "</" + name;
                size_t close = end;
                while (true) {
                    close = text.find('<', close);
                    if (close == std::string_view::npos) {
                        break;
                    }
                    std::string probe(text.substr(close, closer.size()));
                    std::ranges::transform(probe, probe.begin(), lower);
                    if (probe == closer) {
                        break;
                    }
                    close++;
                }
                if (close == std::string_view::npos) {
                    end = text.size();
                } else {
                    builder.text(text.substr(end, close - end));
                    const size_t gt = text.find('>', close);
                    end = gt == std::string_view::npos ? text.size() : gt + 1;
                    builder.close(name);
                }
            }
        }
        pos = text_start = end;
    }
    builder.text(text.substr(text_start));
    return builder.take();
}

}  // namespace vlm::mapping
