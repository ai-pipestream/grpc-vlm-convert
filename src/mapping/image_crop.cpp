#include "image_crop.h"

#include <algorithm>
#include <cmath>
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

bool crop_png_image(const std::string& png, double left, double top, double right,
                    double bottom, uint32_t page_width, uint32_t page_height,
                    docv1::ImageRef* image) {
    int width = 0, height = 0, channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(png.data()), static_cast<int>(png.size()), &width,
        &height, &channels, 4);
    if (pixels == nullptr || width <= 0 || height <= 0) {
        return false;
    }

    // The box is in the declared page raster coordinates; scale into the
    // decoded image's actual pixels before clamping.
    const double sx = page_width > 0 ? static_cast<double>(width) / page_width : 1.0;
    const double sy = page_height > 0 ? static_cast<double>(height) / page_height : 1.0;
    const int x1 = std::clamp(static_cast<int>(std::floor(left * sx)), 0, width);
    const int y1 = std::clamp(static_cast<int>(std::floor(top * sy)), 0, height);
    const int x2 = std::clamp(static_cast<int>(std::ceil(right * sx)), 0, width);
    const int y2 = std::clamp(static_cast<int>(std::ceil(bottom * sy)), 0, height);
    if (x2 <= x1 || y2 <= y1) {
        stbi_image_free(pixels);
        return false;
    }

    const int crop_w = x2 - x1;
    const int crop_h = y2 - y1;
    std::vector<stbi_uc> crop(static_cast<size_t>(crop_w) * crop_h * 4);
    for (int row = 0; row < crop_h; row++) {
        const stbi_uc* src = pixels + (static_cast<size_t>(y1 + row) * width + x1) * 4;
        std::copy_n(src, static_cast<size_t>(crop_w) * 4,
                    crop.data() + static_cast<size_t>(row) * crop_w * 4);
    }
    stbi_image_free(pixels);

    PngSink sink;
    if (stbi_write_png_to_func(png_sink_write, &sink, crop_w, crop_h, 4, crop.data(),
                               crop_w * 4) == 0 ||
        sink.bytes.empty()) {
        return false;
    }

    image->set_mimetype("image/png");
    image->set_dpi(72);  // docling's ImageRef.from_pil default
    image->mutable_size()->set_width(crop_w);
    image->mutable_size()->set_height(crop_h);
    image->set_uri("data:image/png;base64," +
                   base64_encode(reinterpret_cast<const unsigned char*>(sink.bytes.data()),
                                 sink.bytes.size()));
    return true;
}

}  // namespace vlm::mapping
