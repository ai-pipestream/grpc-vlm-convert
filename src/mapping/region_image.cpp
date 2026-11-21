#include "region_image.h"

#include <algorithm>
#include <climits>
#include <vector>

// The stb implementations are compiled in image_crop.cpp.
#include <stb_image.h>
#include <stb_image_write.h>

#include "image_crop.h"

namespace vlm::mapping {
namespace {

constexpr int kChannels = 3;
constexpr int kJpegQuality = 95;

void append_bytes(void* context, void* data, int size) {
    static_cast<std::string*>(context)->append(static_cast<const char*>(data),
                                               static_cast<size_t>(size));
}

}  // namespace

RegionImages::RegionImages(std::string_view png) {
    const auto* bytes = reinterpret_cast<const stbi_uc*>(png.data());
    const int size = static_cast<int>(std::min<size_t>(png.size(), INT_MAX));
    int width = 0, height = 0, channels = 0;
    if (stbi_info_from_memory(bytes, size, &width, &height, &channels) == 0 || width <= 0 ||
        height <= 0 ||
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > kMaxRasterPixels) {
        return;
    }
    // Three channels: PIL's convert("RGB") drops alpha the same way.
    pixels_ = stbi_load_from_memory(bytes, size, &width, &height, &channels, kChannels);
    width_ = width;
    height_ = height;
}

RegionImages::~RegionImages() {
    if (pixels_ != nullptr) {
        stbi_image_free(pixels_);
    }
}

std::string RegionImages::crop(int64_t left, int64_t top, int64_t right, int64_t bottom) const {
    if (pixels_ == nullptr || left < 0 || top < 0 || right > width_ || bottom > height_ ||
        right <= left || bottom <= top) {
        return {};
    }
    const int crop_w = static_cast<int>(right - left);
    const int crop_h = static_cast<int>(bottom - top);
    const size_t stride = static_cast<size_t>(crop_w) * kChannels;
    std::vector<stbi_uc> region(stride * crop_h);
    for (int row = 0; row < crop_h; row++) {
        const stbi_uc* source =
            pixels_ + ((static_cast<size_t>(top) + row) * width_ + left) * kChannels;
        std::copy_n(source, stride, region.data() + row * stride);
    }

    std::string jpeg;
    if (stbi_write_jpg_to_func(append_bytes, &jpeg, crop_w, crop_h, kChannels, region.data(),
                               kJpegQuality) == 0) {
        return {};
    }
    int width = 0, height = 0, channels = 0;
    stbi_uc* decoded =
        stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(jpeg.data()),
                              static_cast<int>(jpeg.size()), &width, &height, &channels, kChannels);
    if (decoded == nullptr) {
        return {};
    }
    std::string png;
    const int written = stbi_write_png_to_func(append_bytes, &png, width, height, kChannels,
                                               decoded, width * kChannels);
    stbi_image_free(decoded);
    return written != 0 ? png : std::string();
}

}  // namespace vlm::mapping
