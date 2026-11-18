#pragma once

#include <string_view>

namespace vlm::mapping {

// Python difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() over
// the code points of two UTF-8 strings: 2 * matches / (len(a) + len(b)),
// 1.0 when both are empty. Bytes that are not valid UTF-8 count as one
// code point each.
double sequence_ratio(std::string_view a, std::string_view b);

}  // namespace vlm::mapping
