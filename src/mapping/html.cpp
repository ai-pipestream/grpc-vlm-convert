#include "builder.h"
#include "mapper.h"

#include <algorithm>
#include <cctype>
#include <regex>

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

}  // namespace

bool map_html(const std::string& text, const PageContext& page, docv1::Document* out,
              std::string* error) {
    // Block-level constructs only — this is a snippet mapper, not an HTML
    // parser; anything richer belongs to the HTML collector upstream.
    static const std::regex kBlock(
        "<(h[1-6]|p|li|pre|code|table)[^>]*>(.*?)</\\1>",
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
            // Cell-level HTML tables are the HTML collector's job; here a
            // table becomes one item with its text content.
            add_table(out, page, page_prov(page));
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
