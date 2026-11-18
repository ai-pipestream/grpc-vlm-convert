#pragma once

// Region images for a second model call on part of a page: Nemotron Parse's
// multi-pass crops. The page raster is decoded once; each region is cut
// out, put through a JPEG quality 95 round trip as docling does before it
// sends a crop (the model was trained on such crops), and encoded as the
// PNG the call carries.

#include <cstdint>
#include <string>
#include <string_view>

namespace vlm::mapping {

class RegionImages {
  public:
    // `png` must outlive the object. A raster that does not decode, or
    // whose header claims more than kMaxRasterPixels, leaves it empty.
    explicit RegionImages(std::string_view png);
    ~RegionImages();
    RegionImages(const RegionImages&) = delete;
    RegionImages& operator=(const RegionImages&) = delete;

    bool decoded() const { return pixels_ != nullptr; }
    int64_t width() const { return width_; }
    int64_t height() const { return height_; }

    // The pixel rectangle [left, right) x [top, bottom) as an RGB PNG;
    // empty when the rectangle is outside the raster or an encode fails.
    std::string crop(int64_t left, int64_t top, int64_t right, int64_t bottom) const;

  private:
    unsigned char* pixels_ = nullptr;
    int64_t width_ = 0;
    int64_t height_ = 0;
};

}  // namespace vlm::mapping
