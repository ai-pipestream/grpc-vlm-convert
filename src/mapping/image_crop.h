#pragma once

// Page-image crops for picture provenance: Docling attaches the picture's
// region cropped from the page raster as the item's ImageRef. Decode or
// encode failure must never fail the page — callers emit the PictureItem
// without an image when this returns empty.

#include <cstdint>
#include <string>

#include "ai/pipestream/document/v1/document.pb.h"

namespace vlm::mapping {

// Crops [left, top, right, bottom] (pixels in the declared page raster
// coordinates, TOPLEFT) out of `png` and fills `image` with the crop as a
// PNG data URI. Returns false on any decode/crop/encode failure or when
// the region is empty; `image` is then untouched.
bool crop_png_image(const std::string& png, double left, double top, double right,
                    double bottom, uint32_t page_width, uint32_t page_height,
                    ai::pipestream::document::v1::ImageRef* image);

}  // namespace vlm::mapping
