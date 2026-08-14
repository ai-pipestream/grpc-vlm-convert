#pragma once

// Shared OTSL grid parser: turns the cell token stream between <otsl> and
// </otsl> into TableData. Used by the OTSL format mapper and by the
// DocTags mapper for <table> elements whose body is OTSL.

#include <string>

#include "ai/pipestream/document/v1/document.pb.h"

namespace vlm::mapping {

// Parses OTSL cell tokens (<fcel>, <ecel>, <ched>, <rhed>, <nl>) into a
// table grid. Returns false when the body holds no complete row.
bool parse_otsl_grid(const std::string& body,
                     ai::pipestream::document::v1::TableData* data);

}  // namespace vlm::mapping
