#pragma once

// Page-image crops for picture provenance: Docling attaches the picture's
// region cropped from the page raster as the item's ImageRef. Decode or
// encode failure must never fail the page — callers emit the PictureItem
// without an image when no crop comes back.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "ai/pipestream/document/v1/document.pb.h"

namespace vlm::mapping {

// Rasters whose header claims more pixels than this are never decoded: a
// small, highly compressible PNG can declare a gigapixel image, and the
// decode is 4 bytes per pixel. 40 million covers a 600 DPI A4 or US letter
// page.
constexpr uint64_t kMaxRasterPixels = 40'000'000;

// What one page may spend on crops: a model repeating <picture> must not
// multiply the page into its own fragment (each crop is re-encoded and
// inlined as a data URI).
struct CropBudget {
    // Crops per page.
    size_t max_crops = 100;
    // Cropped pixels per page, as a multiple of the page raster's pixels.
    double max_area_pages = 2.0;
};

// The page raster picture crops come out of, decoded at most once per
// page and only when the first crop is asked for: a page whose model
// output names fifty pictures pays for one decode, not fifty.
class PageRaster {
  public:
    enum class Crop {
        kAttached,        // `image` holds the crop
        kFailed,          // undecodable raster, empty region, or encode failure
        kOverBudget,      // the page's CropBudget ran out
        kRasterTooLarge,  // the raster is above kMaxRasterPixels
    };

    // `png` must outlive the raster.
    PageRaster(std::string_view png, uint32_t page_width, uint32_t page_height,
               CropBudget budget = {});
    ~PageRaster();
    PageRaster(const PageRaster&) = delete;
    PageRaster& operator=(const PageRaster&) = delete;

    // Crops [left, top, right, bottom] (pixels in the declared page raster
    // coordinates, TOPLEFT) and fills `image` with the crop as a PNG data
    // URI when it returns kAttached; `image` is untouched otherwise.
    Crop crop(double left, double top, double right, double bottom,
              ai::pipestream::document::v1::ImageRef* image);

    // How many times the raster was decoded (0 or 1).
    int decodes() const { return decodes_; }
    // The raster's size as its header declares it; zero until a crop asked.
    uint32_t width() const { return static_cast<uint32_t>(width_); }
    uint32_t height() const { return static_cast<uint32_t>(height_); }

  private:
    // Decodes on first use. False when the raster is undecodable or too
    // large (too_large_ says which).
    bool decode();

    std::string_view png_;
    uint32_t page_width_;
    uint32_t page_height_;
    CropBudget budget_;
    bool tried_ = false;
    bool too_large_ = false;
    unsigned char* pixels_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int decodes_ = 0;
    size_t crops_ = 0;
    uint64_t cropped_pixels_ = 0;
};

// One crop from a raster decoded for this call alone, for callers with a
// single region. Returns false on any decode/crop/encode failure, when the
// raster is above kMaxRasterPixels, or when the region is empty; `image` is
// then untouched.
bool crop_png_image(const std::string& png, double left, double top, double right,
                    double bottom, uint32_t page_width, uint32_t page_height,
                    ai::pipestream::document::v1::ImageRef* image);

}  // namespace vlm::mapping
