#pragma once

// Nemotron Parse 3.5 double and triple pass (docling
// nemotron_parse_multipass.py): Table and Chart regions from the page parse
// (and, for triple, from a no-text detection pass) are cropped with
// NVIDIA's margins, parsed again on their own, mapped back to the page, and
// allowed to replace or add a page block only when their structure checks
// out. The page block keeps its position and bounding box.
//
// Everything here is pure: the service makes the model calls and the
// raster crops, these decide what to crop and what to keep.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ai/pipestream/vlm/v1/vlm_convert.pb.h"
#include "nemotron_syntax.h"

namespace vlm::mapping {

// Adds the page warnings for what a Nemotron Parse 3.5 answer held outside
// its well-formed blocks: PAGE_WARNING_CODE_ELEMENTS_SKIPPED for blocks
// without corners, PAGE_WARNING_CODE_OUTPUT_TRUNCATED for an unfinished
// last block. Nothing is added when `unread` is empty or `warnings` null.
void report_nemotron_unread(const NemotronUnread& unread,
                            std::vector<ai::pipestream::vlm::v1::PageWarning>* warnings);

// docling's Nemotron Parse 3.5 page prompt: boxes, classes and markdown
// text, including text inside pictures.
inline constexpr std::string_view kNemotronPagePrompt =
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>";

// The detection pass prompt: boxes and classes, no text.
inline constexpr std::string_view kNemotronDetectPrompt =
    "</s><s><predict_bbox><predict_classes><output_no_text><predict_no_text_in_pic>";

// At most this many regions are cropped per page.
inline constexpr size_t kMaxNemotronCrops = 80;

// One region to parse again: the block it came from, the pixel rectangle
// to cut from the page raster (margin included, clamped), and that
// rectangle normalized to the page.
struct NemotronCrop {
    NemotronBlock region;
    int64_t left = 0, top = 0, right = 0, bottom = 0;
    double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

// Table and Chart blocks whose corners are ordered and inside [0, 1].
std::vector<NemotronBlock> nemotron_structural_regions(std::string_view content);

// The crops for one page raster of `width` x `height` pixels. Regions come
// from `detection` when it is set (triple pass) and from `parser`
// otherwise; with `union_regions`, parser regions the detector missed are
// added. Near-duplicates of the same class are dropped and crops under
// 8 pixels on a side are skipped. At most kMaxNemotronCrops regions are
// planned; `over_cap`, when given, receives how many more there were.
std::vector<NemotronCrop> plan_nemotron_crops(int64_t width, int64_t height,
                                              std::string_view parser,
                                              const std::string* detection,
                                              bool union_regions, size_t* over_cap = nullptr);

// The column count of a LaTeX tabular whose grid is complete and
// consistent (spans, multirow placeholders); nullopt otherwise.
std::optional<size_t> nemotron_table_columns(const std::string& text);

// A Table with a valid grid, or a Chart with one closed header and a
// rectangular data table of at least three rows and two columns.
bool nemotron_valid_structure(const NemotronBlock& block);

// The crop pass output mapped back to page coordinates: the blocks of the
// crop's class that overlap its region and pass the structure check.
std::vector<NemotronBlock> remap_nemotron_crop(std::string_view content,
                                               const NemotronCrop& crop);

// The page output with guarded replacements and additions applied, in the
// page's reading order. Returns `content` unchanged when nothing applies.
std::string merge_nemotron_crops(const std::string& content,
                                 const std::vector<NemotronBlock>& candidates);

}  // namespace vlm::mapping
