#include "image_crop.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

// stb is single-header, public domain / MIT. The implementation lives in
// this TU only; the headers come in as SYSTEM includes so their warnings
// cannot trip -Werror.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace docv1 = ai::pipestream::document::v1;

namespace vlm::mapping {

namespace {

std::string base64_encode(const unsigned char* bytes, size_t size) {
    static const char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((size + 2) / 3 * 4);
    for (size_t i = 0; i < size; i += 3) {
        unsigned group = static_cast<unsigned>(bytes[i]) << 16;
        if (i + 1 < size) {
            group |= static_cast<unsigned>(bytes[i + 1]) << 8;
        }
        if (i + 2 < size) {
            group |= static_cast<unsigned>(bytes[i + 2]);
        }
        out.push_back(kAlphabet[(group >> 18) & 63]);
        out.push_back(kAlphabet[(group >> 12) & 63]);
        out.push_back(i + 1 < size ? kAlphabet[(group >> 6) & 63] : '=');
        out.push_back(i + 2 < size ? kAlphabet[group & 63] : '=');
    }
    return out;
}

struct PngSink {
    std::string bytes;
};

void png_sink_write(void* context, void* data, int size) {
    auto* sink = static_cast<PngSink*>(context);
    sink->bytes.append(static_cast<const char*>(data), static_cast<size_t>(size));
}

}  // namespace

PageRaster::PageRaster(std::string_view png, uint32_t page_width, uint32_t page_height,
                       CropBudget budget)
    : png_(png), page_width_(page_width), page_height_(page_height), budget_(budget) {}

PageRaster::~PageRaster() {
    if (pixels_ != nullptr) {
        stbi_image_free(pixels_);
    }
}

bool PageRaster::decode() {
    if (tried_) {
        return pixels_ != nullptr;
    }
    tried_ = true;
    const auto* bytes = reinterpret_cast<const stbi_uc*>(png_.data());
    const int size = static_cast<int>(std::min<size_t>(png_.size(), INT32_MAX));
    // The header first: what the raster claims to be is checked before a
    // byte of it is decoded.
    int width = 0, height = 0, channels = 0;
    if (stbi_info_from_memory(bytes, size, &width, &height, &channels) == 0 || width <= 0 ||
        height <= 0) {
        return false;
    }
    width_ = width;
    height_ = height;
    if (static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > kMaxRasterPixels) {
        too_large_ = true;
        return false;
    }
    decodes_++;
    pixels_ = stbi_load_from_memory(bytes, size, &width, &height, &channels, 4);
    if (pixels_ == nullptr || width != width_ || height != height_) {
        if (pixels_ != nullptr) {
            stbi_image_free(pixels_);
            pixels_ = nullptr;
        }
        return false;
    }
    return true;
}

PageRaster::Crop PageRaster::crop(double left, double top, double right, double bottom,
                                  docv1::ImageRef* image) {
    if (!decode()) {
        return too_large_ ? Crop::kRasterTooLarge : Crop::kFailed;
    }

    // The box is in the declared page raster coordinates; scale into the
    // decoded image's actual pixels before clamping. Clamping happens in
    // the double domain: casting an out-of-range double to int is UB, and
    // hostile loc values can drive the box far outside the raster.
    const double sx = page_width_ > 0 ? static_cast<double>(width_) / page_width_ : 1.0;
    const double sy = page_height_ > 0 ? static_cast<double>(height_) / page_height_ : 1.0;
    const int x1 = static_cast<int>(std::clamp(std::floor(left * sx), 0.0,
                                               static_cast<double>(width_)));
    const int y1 = static_cast<int>(std::clamp(std::floor(top * sy), 0.0,
                                               static_cast<double>(height_)));
    const int x2 = static_cast<int>(std::clamp(std::ceil(right * sx), 0.0,
                                               static_cast<double>(width_)));
    const int y2 = static_cast<int>(std::clamp(std::ceil(bottom * sy), 0.0,
                                               static_cast<double>(height_)));
    if (x2 <= x1 || y2 <= y1) {
        return Crop::kFailed;
    }

    const int crop_w = x2 - x1;
    const int crop_h = y2 - y1;
    const uint64_t area = static_cast<uint64_t>(crop_w) * static_cast<uint64_t>(crop_h);
    const double area_budget = budget_.max_area_pages * static_cast<double>(width_) *
                               static_cast<double>(height_);
    if (crops_ >= budget_.max_crops ||
        static_cast<double>(cropped_pixels_ + area) > area_budget) {
        return Crop::kOverBudget;
    }
    crops_++;
    cropped_pixels_ += area;

    std::vector<stbi_uc> pixels(static_cast<size_t>(crop_w) * crop_h * 4);
    for (int row = 0; row < crop_h; row++) {
        const stbi_uc* src = pixels_ + (static_cast<size_t>(y1 + row) * width_ + x1) * 4;
        std::copy_n(src, static_cast<size_t>(crop_w) * 4,
                    pixels.data() + static_cast<size_t>(row) * crop_w * 4);
    }

    PngSink sink;
    if (stbi_write_png_to_func(png_sink_write, &sink, crop_w, crop_h, 4, pixels.data(),
                               crop_w * 4) == 0 ||
        sink.bytes.empty()) {
        return Crop::kFailed;
    }

    image->set_mimetype("image/png");
    image->set_dpi(72);  // docling's ImageRef.from_pil default
    image->mutable_size()->set_width(crop_w);
    image->mutable_size()->set_height(crop_h);
    image->set_uri("data:image/png;base64," +
                   base64_encode(reinterpret_cast<const unsigned char*>(sink.bytes.data()),
                                 sink.bytes.size()));
    return Crop::kAttached;
}

bool crop_png_image(const std::string& png, double left, double top, double right,
                    double bottom, uint32_t page_width, uint32_t page_height,
                    docv1::ImageRef* image) {
    PageRaster raster(png, page_width, page_height);
    return raster.crop(left, top, right, bottom, image) == PageRaster::Crop::kAttached;
}

}  // namespace vlm::mapping
