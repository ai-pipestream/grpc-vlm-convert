#include "builder.h"
#include "mapper.h"

namespace vlm::mapping {

bool map_plaintext(const std::string& text, const PageContext& page, docv1::Document* out,
                   std::string* error) {
    std::string body = trim(text);
    if (body.empty()) {
        *error = "empty plaintext response";
        return false;
    }
    // Plaintext carries no structure at all: exactly one TextItem per page.
    add_text(out, page, docv1::DOC_ITEM_LABEL_TEXT, page_prov(page), body);
    finalize_document(out, page);
    return true;
}

bool map_response(vlmv1::ResponseFormat format, const std::string& text,
                  const PageContext& page, docv1::Document* out, std::string* error) {
    switch (format) {
        case vlmv1::RESPONSE_FORMAT_DOCTAGS:
            return map_doctags(text, page, out, error);
        case vlmv1::RESPONSE_FORMAT_MARKDOWN:
            return map_markdown(text, page, out, error);
        case vlmv1::RESPONSE_FORMAT_HTML:
            return map_html(text, page, out, error);
        case vlmv1::RESPONSE_FORMAT_OTSL:
            return map_otsl(text, page, out, error);
        case vlmv1::RESPONSE_FORMAT_PLAINTEXT:
            return map_plaintext(text, page, out, error);
        default:
            *error = "unresolved response format";
            return false;
    }
}

}  // namespace vlm::mapping
