#pragma once

// Shared OTSL grid parser: turns the cell token stream between <otsl> and
// </otsl> into TableData. Used by the OTSL format mapper and by the
// DocTags mapper for <table> elements whose body is OTSL.

#include <string>

#include "ai/pipestream/document/v1/document.pb.h"

namespace vlm::mapping {

// Parses OTSL cell tokens into a table grid with docling's span
// resolution: <lcel>/<ucel>/<xcel> filler cells extend their anchor
// cell's col_span/row_span (and end offsets) and are not emitted;
// <srow> starts a new (section) row; <nl> ends a row. The grid is the
// full num_rows × num_cols matrix with anchors stamped over every
// position their span covers. Returns false when the body holds no
// complete row.
bool parse_otsl_grid(const std::string& body,
                     ai::pipestream::document::v1::TableData* data);

}  // namespace vlm::mapping
