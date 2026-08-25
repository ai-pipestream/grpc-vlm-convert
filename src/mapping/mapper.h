#pragma once

#include <string>

#include "ai/pipestream/document/v1/document.pb.h"
#include "ai/pipestream/vlm/v1/vlm_convert.pb.h"

namespace vlm::mapping {

namespace docv1 = ai::pipestream::document::v1;
namespace vlmv1 = ai::pipestream::vlm::v1;

// What one page's model output is mapped against.
struct PageContext {
    // 1-based page number; provenance on every item.
    uint32_t page_no = 0;
    // Raster dimensions; DocTags locations scale into these pixels.
    uint32_t width = 0;
    uint32_t height = 0;
    // The page raster itself, when the caller sent one: the DocTags mapper
    // crops picture regions out of it for ImageRef attachments. Empty for
    // mappers/tests that have no image.
    std::string png;
    // Collector attribution stamped on every item ("vlm-convert", the
    // model name, this server's version, and the page's raw model score
    // when the endpoint reported logprobs).
    docv1::CollectorSource source;
    // The model invocation that produced this page: which model answered,
    // from which endpoint, how the generation stopped, what it cost. Rides
    // every item's source list next to the collector source, so a consumer
    // holding one item can tell a truncated answer from a complete one.
    docv1::GenerationSource generation;
    // False for mappers and tests with no generation to attribute (the
    // fragment then carries the collector source alone).
    bool has_generation = false;
};

// Maps one page's model response into a Document fragment. Returns true
// with `out` filled; returns false with `error` set when the text does
// not parse as the declared format — the caller then emits PageRaw.
bool map_response(vlmv1::ResponseFormat format, const std::string& text,
                  const PageContext& page, docv1::Document* out, std::string* error);

// Individual format mappers, exposed for unit tests. Same contract.
bool map_doctags(const std::string& text, const PageContext& page, docv1::Document* out,
                 std::string* error);
bool map_markdown(const std::string& text, const PageContext& page, docv1::Document* out,
                  std::string* error);
bool map_html(const std::string& text, const PageContext& page, docv1::Document* out,
              std::string* error);
bool map_otsl(const std::string& text, const PageContext& page, docv1::Document* out,
              std::string* error);
bool map_plaintext(const std::string& text, const PageContext& page, docv1::Document* out,
                   std::string* error);

}  // namespace vlm::mapping
