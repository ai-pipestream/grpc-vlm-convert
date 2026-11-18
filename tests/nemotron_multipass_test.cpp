// Unit tests for the Nemotron Parse 3.5 double and triple pass logic. Every
// expected value was produced by docling's own functions
// (docling/utils/nemotron_parse_multipass.py and difflib) on the same input.

#include <cmath>
#include <string>
#include <vector>

#include "fixture.h"
#include "mapping/nemotron_multipass.h"
#include "mapping/sequence_ratio.h"

using vlm::mapping::NemotronBlock;
using vlm::mapping::NemotronCrop;

namespace {

bool near(double a, double b) { return std::abs(a - b) < 1e-9; }

void verify_sequence_ratio_matches_difflib() {
    struct Case {
        std::string a, b;
        double ratio;
    };
    const std::vector<Case> cases = {
        {"abcd", "bcde", 0.75},
        {"", "", 1.0},
        {"abc", "", 0.0},
        {"kitten", "sitting", 0.6153846153846154},
        {"\u00fcn\u00efc\u00f6d\u00e9 t\u00ebxt", "unicode text", 0.5833333333333334},
        {"a b c a b c", "c b a", 0.5},
        {R"(\begin{tabular}{cc}a & b \\ c & d \\\end{tabular})",
         R"(\begin{tabular}{cc}a & b \\ c & e \\\end{tabular})", 0.9795918367346939},
        // A byte that is not UTF-8 counts as one element, like the U+00FF
        // Python would see after a latin-1 decode.
        {"\xff" "ab", "ab", 0.8},
    };
    for (const Case& c : cases) {
        require(near(vlm::mapping::sequence_ratio(c.a, c.b), c.ratio),
                "sequence ratio of '" + c.a + "' and '" + c.b + "'");
    }
}

void verify_table_columns_follow_the_strict_grid_guard() {
    struct Case {
        std::string latex;
        std::optional<size_t> columns;
    };
    const std::vector<Case> cases = {
        {R"(\begin{tabular}{cc}a & b \\ c & d \\\end{tabular})", 2},
        {R"(\begin{tabular}{|l|c|r|}\multicolumn{2}{c}{ab} & c \\ d & e & f \\\end{tabular})", 3},
        {R"(\begin{tabular}{cc}\multirow{2}{*}{a} & b \\ & d \\\end{tabular})", 2},
        {R"(\begin{tabular}{cc}\multirow{2}{*}{a} & b \\ \mbox{} & d \\\end{tabular})", 2},
        // Text on a cell the multirow covers.
        {R"(\begin{tabular}{cc}\multirow{2}{*}{a} & b \\ x & d \\\end{tabular})", std::nullopt},
        {R"(\begin{tabular}{cc}\multicolumn{0}{c}{a} & b \\\end{tabular})", std::nullopt},
        {R"(\begin{tabular}{cc}\multicolumn{ x}{c}{a} & b \\\end{tabular})", std::nullopt},
        {R"(\begin{tabular}{cc}a & b & c \\\end{tabular})", std::nullopt},
        {R"(\begin{tabular}{cc}a & {b \\\end{tabular})", std::nullopt},
        {R"(\begin{tabular}{cc}a & b \\\end{tabular}\begin{tabular}{cc}a & b \\\end{tabular})",
         std::nullopt},
        {R"(\begin{tabular}{cq}a & b \\\end{tabular})", std::nullopt},
        // A multirow running past the last row.
        {R"(\begin{tabular}{cc}\multirow{3}{*}{a} & b \\ & d \\\end{tabular})", std::nullopt},
        {"\\begin{tabular}{c\x1c" "c}a & b \\\\\\end{tabular}", 2},
        // An escaped brace does not unbalance the text.
        {R"(text \{ \begin{tabular}{c}a \\\end{tabular})", 1},
    };
    for (const Case& c : cases) {
        require(vlm::mapping::nemotron_table_columns(c.latex) == c.columns,
                "table columns of " + c.latex);
    }
}

void verify_chart_structure_needs_one_header_and_a_rectangular_table() {
    struct Case {
        std::string text;
        bool valid;
    };
    const std::vector<Case> cases = {
        {"<chart type=\"bar\">\n| x | y |\n| 1 | 2 |\n| 3 | 4 |\n</chart>", true},
        {"<chart type=\"bar\">\n| x | y |\n| 1 | 2 |\n</chart>", false},
        {"<chart type=\"bar\">\n| x | y |\n| 1 | 2 | 3 |\n| 3 | 4 |\n</chart>", false},
        {"<chart>\n| x |\n| 1 |\n| 3 |\n</chart>", false},
        // `<chart <chart type="bar">` is one header match, not two.
        {"<chart <chart type=\"bar\">\n| x | y |\n| 1 | 2 |\n| 3 | 4 |\n</chart>", true},
        {"<chart type=\"a\"><chart type=\"b\">\n| x | y |\n| 1 | 2 |\n| 3 | 4 |\n</chart>", false},
        {"<charts>\n| x | y |\n| 1 | 2 |\n| 3 | 4 |\n</chart>", false},
        {"<chart type=\"bar\">\r| x | y |\v| 1 | 2 |\u2028| 3 | 4 |\n</chart>", true},
    };
    for (const Case& c : cases) {
        require(vlm::mapping::nemotron_valid_structure({.label = "Chart", .text = c.text}) == c.valid,
                "chart structure of " + c.text);
    }
}

const std::string kTable = R"(\begin{tabular}{cc}a & b \\ c & d \\\end{tabular})";

// A page parse with a title, a table, a chart and its near-duplicate, a
// tiny table and one with inverted corners.
const std::string kParser =
    "<x_0.1><y_0.1>Title<x_0.9><y_0.15><class_Title>\n"
    "<x_0.1><y_0.2>" + kTable + "<x_0.6><y_0.5><class_Table>\n"
    "<x_0.5><y_0.6>chart<x_0.9><y_0.9><class_Chart>\n"
    "<x_0.5><y_0.6>dup<x_0.9><y_0.9001><class_Chart>\n"
    "<x_0.95><y_0.95>tiny<x_0.951><y_0.951><class_Table>\n"
    "<x_0.7><y_0.1>bad<x_0.2><y_0.3><class_Table>";

const std::string kDetection =
    "<x_0.12><y_0.21><x_0.6><y_0.5><class_Table>\n"
    "<x_0.0><y_0.0><x_0.3><y_0.05><class_Chart>";

struct ExpectedCrop {
    std::string label;
    double region_x1;
    int64_t left, top, right, bottom;
};

void require_crops(const std::vector<NemotronCrop>& crops,
                   const std::vector<ExpectedCrop>& expected, const std::string& what) {
    require(crops.size() == expected.size(), what + ": crop count");
    for (size_t i = 0; i < crops.size(); i++) {
        const NemotronCrop& crop = crops[i];
        const ExpectedCrop& want = expected[i];
        const std::string at = what + ": crop " + std::to_string(i);
        require(crop.region.label == want.label && near(crop.region.x1, want.region_x1),
                at + " region");
        require(crop.left == want.left && crop.top == want.top && crop.right == want.right &&
                    crop.bottom == want.bottom,
                at + " pixels");
        require(near(crop.x1, want.left / 1000.0) && near(crop.y1, want.top / 800.0) &&
                    near(crop.x2, want.right / 1000.0) && near(crop.y2, want.bottom / 800.0),
                at + " normalized box");
    }
}

void verify_crop_plans_match_docling() {
    using vlm::mapping::plan_nemotron_crops;
    // Double pass: parser regions, the near-duplicate chart dropped.
    require_crops(plan_nemotron_crops(1000, 800, kParser, nullptr, true),
                  {{"Table", 0.1, 68, 138, 632, 422},
                   {"Chart", 0.5, 472, 458, 928, 742},
                   {"Table", 0.95, 937, 747, 964, 773}},
                  "double pass");
    // Triple pass, detector regions only.
    require_crops(plan_nemotron_crops(1000, 800, kParser, &kDetection, false),
                  {{"Table", 0.12, 88, 146, 632, 422}, {"Chart", 0.0, 0, 0, 324, 54}},
                  "triple detector");
    // Triple pass, union: parser regions the detector missed follow.
    require_crops(plan_nemotron_crops(1000, 800, kParser, &kDetection, true),
                  {{"Table", 0.12, 88, 146, 632, 422},
                   {"Chart", 0.0, 0, 0, 324, 54},
                   {"Chart", 0.5, 472, 458, 928, 742},
                   {"Table", 0.95, 937, 747, 964, 773}},
                  "triple union");
    // A raster too narrow for an 8 pixel crop.
    require(plan_nemotron_crops(6, 800, kParser, nullptr, true).empty(), "narrow page: no crops");
}

void verify_crop_output_maps_back_to_the_page() {
    const std::vector<NemotronCrop> crops =
        vlm::mapping::plan_nemotron_crops(1000, 800, kParser, nullptr, true);
    const std::string crop_output =
        "<x_0.05><y_0.05>" R"(\begin{tabular}{cc}a & b \\ c & D \\\end{tabular})"
        "<x_0.95><y_0.95><class_Table>\n"
        "<x_0.0><y_0.0>Caption<x_1.0><y_0.04><class_Caption>\n"
        "<x_0.05><y_0.05>" R"(\begin{tabular}{cc}a & b \\ c \\\end{tabular})"
        "<x_0.95><y_0.95><class_Table>\n"
        "<x_0.0><y_0.0>" + kTable + "<x_0.2><y_0.2><class_Table>";
    // Only the first block keeps the class, overlaps the region and has a
    // valid grid.
    const std::vector<NemotronBlock> mapped = vlm::mapping::remap_nemotron_crop(crop_output, crops[0]);
    require(mapped.size() == 1, "remap: one candidate");
    require(mapped[0].label == "Table" &&
                mapped[0].text == R"(\begin{tabular}{cc}a & b \\ c & D \\\end{tabular})",
            "remap: the valid table");
    require(near(mapped[0].x1, 0.09620000000000001) && near(mapped[0].y1, 0.19024999999999997) &&
                near(mapped[0].x2, 0.6038000000000001) && near(mapped[0].y2, 0.5097499999999999),
            "remap: page coordinates");
}

std::string page(const std::string& table_text, const std::string& prefix = "") {
    return prefix +
           "<x_0.10000000><y_0.10000000>Title<x_0.90000000><y_0.15000000><class_Title>\n"
           "<x_0.10000000><y_0.20000000>" + table_text +
           "<x_0.60000000><y_0.50000000><class_Table>\n"
           "<x_0.50000000><y_0.60000000>chart<x_0.90000000><y_0.90000000><class_Chart>\n"
           "<x_0.50000000><y_0.60000000>dup<x_0.90000000><y_0.90010000><class_Chart>\n"
           "<x_0.95000000><y_0.95000000>tiny<x_0.95100000><y_0.95100000><class_Table>\n"
           "<x_0.70000000><y_0.10000000>bad<x_0.20000000><y_0.30000000><class_Table>";
}

NemotronBlock table_at_page_block(std::string text) {
    return {.label = "Table", .x1 = 0.1, .y1 = 0.2, .x2 = 0.6, .y2 = 0.5, .text = std::move(text)};
}

void verify_merge_replaces_only_compatible_tables() {
    using vlm::mapping::merge_nemotron_crops;
    const std::string fixed = R"(\begin{tabular}{cc}a & b \\ c & D \\\end{tabular})";
    require(merge_nemotron_crops(kParser, {table_at_page_block(fixed)}) == page(fixed),
            "merge: a compatible crop replaces the page table text");
    // Three columns where the page had two.
    const std::string wider = R"(\begin{tabular}{ccc}a & b & x \\ c & d & y \\\end{tabular})";
    require(merge_nemotron_crops(kParser, {table_at_page_block(wider)}) == kParser,
            "merge: a column count change is refused");
    // Too little text in common (ratio 0.543).
    const std::string single =
        R"(<x_0.1><y_0.2>\begin{tabular}{cc}Revenue by quarter & Net income \\ 2024 & 2025 \\\end{tabular}<x_0.6><y_0.5><class_Table>)";
    const std::string unrelated =
        R"(\begin{tabular}{cc}Zygomorphic xylophones & pqjwvkfm \\ hhhhhhh & 7777777777 \\\end{tabular})";
    require(merge_nemotron_crops(single, {table_at_page_block(unrelated)}) == single,
            "merge: an unrelated crop is refused");
    // Equal overlap: the later candidate wins.
    const std::string e = R"(\begin{tabular}{cc}a & b \\ c & e \\\end{tabular})";
    const std::string f = R"(\begin{tabular}{cc}a & b \\ c & f \\\end{tabular})";
    require(merge_nemotron_crops(kParser, {table_at_page_block(e), table_at_page_block(f)}) ==
                page(f),
            "merge: ties go to the later candidate");
    require(merge_nemotron_crops(kParser, {}) == kParser, "merge: no candidates, no change");
}

void verify_merge_inserts_new_regions_in_reading_order() {
    const std::string chart = "<chart type=\"bar\">\n| x | y |\n| 1 | 2 |\n| 3 | 4 |\n</chart>";
    const std::vector<NemotronBlock> candidates = {
        {.label = "Chart", .x1 = 0.05, .y1 = 0.55, .x2 = 0.4, .y2 = 0.58, .text = chart},
        {.label = "Table", .x1 = 0.0, .y1 = 0.0, .x2 = 0.05, .y2 = 0.05, .text = kTable},
    };
    const std::string expected =
        "<x_0.00000000><y_0.00000000>" + kTable +
        "<x_0.05000000><y_0.05000000><class_Table>\n"
        "<x_0.10000000><y_0.10000000>Title<x_0.90000000><y_0.15000000><class_Title>\n"
        "<x_0.10000000><y_0.20000000>" + kTable +
        "<x_0.60000000><y_0.50000000><class_Table>\n"
        "<x_0.05000000><y_0.55000000>" + chart +
        "<x_0.40000000><y_0.58000000><class_Chart>\n"
        "<x_0.50000000><y_0.60000000>chart<x_0.90000000><y_0.90000000><class_Chart>\n"
        "<x_0.50000000><y_0.60000000>dup<x_0.90000000><y_0.90010000><class_Chart>\n"
        "<x_0.95000000><y_0.95000000>tiny<x_0.95100000><y_0.95100000><class_Table>\n"
        "<x_0.70000000><y_0.10000000>bad<x_0.20000000><y_0.30000000><class_Table>";
    require(vlm::mapping::merge_nemotron_crops(kParser, candidates) == expected,
            "merge: new regions inserted before the first block below them");
    // On the same top edge, the block further right comes after.
    const std::string two_blocks =
        "<x_0.1><y_0.1>Title<x_0.9><y_0.15><class_Title>\n"
        "<x_0.1><y_0.2>" + kTable + "<x_0.6><y_0.5><class_Table>";
    const std::string same_row =
        "<x_0.10000000><y_0.10000000>Title<x_0.90000000><y_0.15000000><class_Title>\n"
        "<x_0.05000000><y_0.20000000>" + chart +
        "<x_0.09000000><y_0.25000000><class_Chart>\n"
        "<x_0.10000000><y_0.20000000>" + kTable + "<x_0.60000000><y_0.50000000><class_Table>";
    require(vlm::mapping::merge_nemotron_crops(
                two_blocks,
                {{.label = "Chart", .x1 = 0.05, .y1 = 0.2, .x2 = 0.09, .y2 = 0.25, .text = chart}}) ==
                same_row,
            "merge: a new region left of a block on its row goes first");
}

}  // namespace

int main() {
    try {
        verify_sequence_ratio_matches_difflib();
        verify_table_columns_follow_the_strict_grid_guard();
        verify_chart_structure_needs_one_header_and_a_rectangular_table();
        verify_crop_plans_match_docling();
        verify_crop_output_maps_back_to_the_page();
        verify_merge_replaces_only_compatible_tables();
        verify_merge_inserts_new_regions_in_reading_order();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("nemotron-multipass-test passed");
    return 0;
}
