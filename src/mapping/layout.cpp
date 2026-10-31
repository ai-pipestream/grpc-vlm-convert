#include "layout.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>

namespace vlm::mapping {

namespace {

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

int32_t char_count(const std::string& text) {
    int32_t count = 0;
    for (unsigned char c : text) {
        if ((c & 0xC0) != 0x80) {
            count++;
        }
    }
    return count;
}

docv1::TextItemBase* base_of(docv1::BaseTextItem* item) {
    switch (item->item_case()) {
        case docv1::BaseTextItem::kTitle:
            return item->mutable_title()->mutable_base();
        case docv1::BaseTextItem::kSectionHeader:
            return item->mutable_section_header()->mutable_base();
        case docv1::BaseTextItem::kListItem:
            return item->mutable_list_item()->mutable_base();
        case docv1::BaseTextItem::kFormula:
            return item->mutable_formula()->mutable_base();
        case docv1::BaseTextItem::kText:
            return item->mutable_text()->mutable_base();
        case docv1::BaseTextItem::kFieldHeading:
            return item->mutable_field_heading()->mutable_base();
        case docv1::BaseTextItem::kFieldValue:
            return item->mutable_field_value()->mutable_base();
        default:
            return nullptr;
    }
}

constexpr const char* kHtmlBlockTags[] = {"div", "p",       "h1",         "h2",  "h3",
                                          "h4",  "h5",      "h6",         "table", "ul",
                                          "ol",  "li",      "pre",        "figure", "form",
                                          "caption", "figcaption", "hr"};

bool is_block_tag(const std::string& tag) {
    return std::ranges::find(kHtmlBlockTags, tag) != std::end(kHtmlBlockTags);
}

void html_text_into(const MarkupNode& node, std::string* out) {
    if (node.is_text()) {
        *out += node.text;
        return;
    }
    if (node.tag == "br") {
        *out += "\n";
        return;
    }
    if (node.tag == "input") {
        std::string type = node.attr("type");
        std::ranges::transform(type, type.begin(),
                               [](char c) { return static_cast<char>(std::tolower(c)); });
        if (type == "checkbox" || type == "radio") {
            *out += node.has_attr("checked") ? "\u2611" : "\u2610";
        } else {
            *out += node.attr("value");
        }
        return;
    }
    for (const auto& child : node.children) {
        html_text_into(*child, out);
        if (!child->is_text() && is_block_tag(child->tag)) {
            *out += "\n";
        }
    }
}

// A positive span from an attribute, capped (chandra_utils _span): a bad
// value is 1.
size_t parse_span(const std::string& value, size_t maximum) {
    size_t i = 0;
    while (i < value.size() && is_space(value[i])) {
        i++;
    }
    size_t number = 0;
    bool any = false;
    while (i < value.size() && std::isdigit(static_cast<unsigned char>(value[i])) != 0) {
        number = std::min<size_t>(number * 10 + static_cast<size_t>(value[i] - '0'), 1u << 30);
        any = true;
        i++;
    }
    while (i < value.size() && is_space(value[i])) {
        i++;
    }
    if (!any || i != value.size()) {
        return 1;
    }
    return std::min(maximum, std::max<size_t>(1, number));
}

}  // namespace

std::string html_text(const MarkupNode& node) {
    std::string out;
    html_text_into(node, &out);
    return out;
}

std::string collapse_whitespace(std::string_view text) {
    std::string out;
    bool pending = false;
    for (char c : text) {
        if (is_space(c)) {
            pending = !out.empty();
            continue;
        }
        if (pending) {
            out.push_back(' ');
            pending = false;
        }
        out.push_back(c);
    }
    return out;
}

const MarkupNode* find_element(const MarkupNode& node, std::string_view tag) {
    for (const auto& child : node.children) {
        if (child->is_text()) {
            continue;
        }
        if (child->tag == tag) {
            return child.get();
        }
        if (const MarkupNode* found = find_element(*child, tag)) {
            return found;
        }
    }
    return nullptr;
}

std::vector<SpanCell> chandra_table_cells(const MarkupNode& table) {
    struct Row {
        const MarkupNode* node;
        bool in_header;
        size_t group_end;  // 0: the table end
    };
    std::vector<Row> rows;
    for (const auto& child : table.children) {
        if (child->is_text()) {
            continue;
        }
        if (child->tag == "tr") {
            rows.push_back({child.get(), false, 0});
        } else if (child->tag == "thead" || child->tag == "tbody" || child->tag == "tfoot") {
            size_t count = 0;
            for (const auto& row : child->children) {
                count += !row->is_text() && row->tag == "tr" ? 1 : 0;
            }
            const size_t group_end = rows.size() + count;
            for (const auto& row : child->children) {
                if (!row->is_text() && row->tag == "tr") {
                    rows.push_back({row.get(), child->tag == "thead", group_end});
                }
            }
        }
    }
    std::vector<SpanCell> cells;
    std::map<size_t, size_t> occupied;  // column -> first free row
    const size_t max_span = kMaxTableCells;
    for (size_t row_index = 0; row_index < rows.size(); row_index++) {
        const Row& row = rows[row_index];
        size_t col = 0;
        for (const auto& node : row.node->children) {
            if (node->is_text() || (node->tag != "td" && node->tag != "th")) {
                continue;
            }
            const size_t col_span = parse_span(node->attr("colspan", "1"), max_span);
            const size_t remaining = (row.group_end != 0 ? row.group_end : rows.size()) - row_index;
            const size_t row_span = node->attr("rowspan") == "0"
                                        ? remaining
                                        : parse_span(node->attr("rowspan", "1"), remaining);
            while (true) {
                const auto it = occupied.find(col);
                if (it == occupied.end() || it->second <= row_index) {
                    break;
                }
                col++;
            }
            if (col + col_span > kMaxTableCols * 4) {
                break;  // far past the column cap: the rest of the row is dropped
            }
            for (size_t c = col; c < col + col_span; c++) {
                size_t& until = occupied[c];
                until = std::max(until, row_index + row_span);
            }
            SpanCell cell;
            cell.row = row_index;
            cell.col = col;
            cell.row_span = row_span;
            cell.col_span = col_span;
            cell.text = collapse_whitespace(html_text(*node));
            cell.node = node.get();
            const bool th = node->tag == "th";
            cell.column_header =
                node->attr("scope") == "col" || (th && (row.in_header || row_index == 0));
            cell.row_header = node->attr("scope") == "row" ||
                              (th && col == 0 && row_index > 0 && !row.in_header);
            cells.push_back(std::move(cell));
            col += col_span;
        }
    }
    return cells;
}

TableCut fill_span_table(docv1::TableData* data, const std::vector<SpanCell>& cells) {
    TableCut cut;
    for (const SpanCell& cell : cells) {
        cut.source_rows = std::max(cut.source_rows, cell.row + cell.row_span);
        cut.source_cols = std::max(cut.source_cols, cell.col + cell.col_span);
    }
    const auto [kept_rows, kept_cols] = kept_table_shape(cut.source_rows, cut.source_cols);
    cut.truncated = kept_rows < cut.source_rows || kept_cols < cut.source_cols;
    const auto num_rows = static_cast<int32_t>(kept_rows);
    const auto num_cols = static_cast<int32_t>(kept_cols);
    data->set_num_rows(num_rows);
    data->set_num_cols(num_cols);
    for (const SpanCell& source : cells) {
        if (source.row >= kept_rows || source.col >= kept_cols) {
            continue;
        }
        const size_t end_row = std::min(source.row + source.row_span, kept_rows);
        const size_t end_col = std::min(source.col + source.col_span, kept_cols);
        docv1::TableCell* cell = data->add_table_cells();
        cell->set_text(source.text);
        cell->set_column_header(source.column_header);
        cell->set_row_header(source.row_header);
        cell->set_row_span(static_cast<int32_t>(end_row - source.row));
        cell->set_col_span(static_cast<int32_t>(end_col - source.col));
        cell->set_start_row_offset_idx(static_cast<int32_t>(source.row));
        cell->set_end_row_offset_idx(static_cast<int32_t>(end_row));
        cell->set_start_col_offset_idx(static_cast<int32_t>(source.col));
        cell->set_end_col_offset_idx(static_cast<int32_t>(end_col));
    }
    size_t grid_bytes = 0;
    for (const docv1::TableCell& cell : data->table_cells()) {
        grid_bytes += static_cast<size_t>(cell.row_span()) * static_cast<size_t>(cell.col_span()) *
                      cell.ByteSizeLong();
        if (grid_bytes > kMaxGridBytes) {
            cut.grid_omitted = true;
            return cut;
        }
    }
    for (int32_t r = 0; r < num_rows; r++) {
        docv1::TableRow* grid_row = data->add_grid();
        for (int32_t c = 0; c < num_cols; c++) {
            docv1::TableCell* empty = grid_row->add_cells();
            empty->set_row_span(1);
            empty->set_col_span(1);
            empty->set_start_row_offset_idx(r);
            empty->set_end_row_offset_idx(r + 1);
            empty->set_start_col_offset_idx(c);
            empty->set_end_col_offset_idx(c + 1);
        }
    }
    for (const docv1::TableCell& cell : data->table_cells()) {
        for (int32_t r = cell.start_row_offset_idx(); r < cell.end_row_offset_idx(); r++) {
            for (int32_t c = cell.start_col_offset_idx(); c < cell.end_col_offset_idx(); c++) {
                *data->mutable_grid(r)->mutable_cells(c) = cell;
            }
        }
    }
    return cut;
}

std::string strip_markdown_heading(const std::string& text, int* level) {
    *level = 0;
    size_t hashes = 0;
    while (hashes < text.size() && text[hashes] == '#') {
        hashes++;
    }
    if (hashes == 0 || hashes > 6 || hashes >= text.size() ||
        (text[hashes] != ' ' && text[hashes] != '\t')) {
        return text;
    }
    *level = static_cast<int>(hashes);
    size_t start = hashes;
    while (start < text.size() && (text[start] == ' ' || text[start] == '\t')) {
        start++;
    }
    size_t end = text.size();
    while (end > start && is_space(text[end - 1])) {
        end--;
    }
    return text.substr(start, end - start);
}

std::string strip_math_delimiters(const std::string& text) {
    std::string out;
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == '\\' && i + 1 < text.size()) {
            if (text[i + 1] == '\\') {
                out += "\\\\";
                i++;
                continue;
            }
            if (text[i + 1] == '(' || text[i + 1] == ')') {
                i++;
                continue;
            }
        }
        out.push_back(text[i]);
    }
    out = trim(out);
    while (true) {
        std::string inner;
        if (out.size() >= 4 && out.starts_with("$$") && out.ends_with("$$") &&
            out.substr(2, out.size() - 4).find("$$") == std::string::npos) {
            inner = out.substr(2, out.size() - 4);
        } else if (out.size() >= 4 && out.starts_with("\\[") && out.ends_with("\\]") &&
                   out.substr(2, out.size() - 4).find("\\[") == std::string::npos &&
                   out.substr(2, out.size() - 4).find("\\]") == std::string::npos) {
            inner = out.substr(2, out.size() - 4);
        } else if (out.size() >= 2 && out.front() == '$' && out.back() == '$' &&
                   out.substr(1, out.size() - 2).find('$') == std::string::npos) {
            inner = out.substr(1, out.size() - 2);
        } else {
            break;
        }
        out = trim(inner);
    }
    return out;
}

std::string unescape_markdown_literals(const std::string& text) {
    std::string out;
    size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        // Code spans and math pass through untouched.
        if (c == '`') {
            size_t run = 0;
            while (i + run < text.size() && text[i + run] == '`') {
                run++;
            }
            const std::string fence(run, '`');
            const size_t close = text.find(fence, i + run);
            if (close != std::string::npos) {
                out += text.substr(i, close + run - i);
                i = close + run;
                continue;
            }
            out += fence;
            i += run;
            continue;
        }
        if (c == '$') {
            const bool display = i + 1 < text.size() && text[i + 1] == '$';
            const std::string delim = display ? "$$" : "$";
            const size_t close = text.find(delim, i + delim.size());
            if (close != std::string::npos &&
                (display || text.substr(i, close - i).find('\n') == std::string::npos)) {
                out += text.substr(i, close + delim.size() - i);
                i = close + delim.size();
                continue;
            }
        }
        if (c == '\\' && i + 1 < text.size()) {
            const char next = text[i + 1];
            if (next == '*' || next == '_') {
                out.push_back(next);
                i += 2;
                continue;
            }
            if (next == '\\') {
                out += "\\\\";
                i += 2;
                continue;
            }
            if (next == '(' || next == '[') {
                const std::string closer = next == '(' ? "\\)" : "\\]";
                const size_t close = text.find(closer, i + 2);
                if (close != std::string::npos) {
                    out += text.substr(i, close + 2 - i);
                    i = close + 2;
                    continue;
                }
            }
        }
        out.push_back(c);
        i++;
    }
    return out;
}

std::string markdown_to_text(const std::string& markdown) {
    std::string out;
    size_t line_start = 0;
    while (line_start <= markdown.size()) {
        size_t line_end = markdown.find('\n', line_start);
        if (line_end == std::string::npos) {
            line_end = markdown.size();
        }
        std::string line = trim(markdown.substr(line_start, line_end - line_start));
        line_start = line_end + 1;
        int level = 0;
        line = strip_markdown_heading(line, &level);
        std::string plain;
        for (size_t i = 0; i < line.size(); i++) {
            const char c = line[i];
            if (c == '\\' && i + 1 < line.size() &&
                std::ispunct(static_cast<unsigned char>(line[i + 1])) != 0) {
                plain.push_back(line[++i]);
                continue;
            }
            if (c == '*' || c == '`') {
                continue;  // emphasis and code-span markers
            }
            if (c == '_' && (i + 1 < line.size() && line[i + 1] == '_')) {
                i++;
                continue;
            }
            if (c == '_' && ((i == 0 || is_space(line[i - 1])) ||
                             (i + 1 == line.size() || is_space(line[i + 1])))) {
                continue;  // _emphasis_ at a word edge, not snake_case
            }
            if (c == '[') {
                const size_t close = line.find("](", i + 1);
                const size_t paren = close == std::string::npos ? close : line.find(')', close);
                if (close != std::string::npos && paren != std::string::npos) {
                    plain += line.substr(i + 1, close - i - 1);
                    i = paren;
                    continue;
                }
            }
            plain.push_back(c);
        }
        plain = collapse_whitespace(plain);
        if (!plain.empty()) {
            if (!out.empty()) {
                out.push_back('\n');
            }
            out += plain;
        }
        if (line_end == markdown.size()) {
            break;
        }
    }
    return out;
}

std::pair<double, double> qwen2vl_image_size(double width, double height) {
    constexpr double kFactor = 28;
    constexpr double kMinPixels = 200704;
    constexpr double kMaxPixels = 2'500'000;
    const double mw = std::floor(width);
    const double mh = std::floor(height);
    if (mw <= 0 || mh <= 0) {
        return {width, height};
    }
    // Python's round() is round-half-even, as nearbyint is by default.
    double h_bar = std::nearbyint(mh / kFactor) * kFactor;
    double w_bar = std::nearbyint(mw / kFactor) * kFactor;
    if (h_bar * w_bar > kMaxPixels) {
        const double beta = std::sqrt((mh * mw) / kMaxPixels);
        h_bar = std::max(kFactor, std::floor(mh / beta / kFactor) * kFactor);
        w_bar = std::max(kFactor, std::floor(mw / beta / kFactor) * kFactor);
    } else if (h_bar * w_bar < kMinPixels) {
        const double beta = std::sqrt(kMinPixels / (mh * mw));
        h_bar = std::ceil(mh * beta / kFactor) * kFactor;
        w_bar = std::ceil(mw * beta / kFactor) * kFactor;
    }
    return {w_bar, h_bar};
}

LayoutWriter::LayoutWriter(const PageContext& page, docv1::Document* out,
                           std::vector<vlmv1::PageWarning>* warnings)
    : page_(page),
      out_(out),
      warnings_(warnings),
      raster_(page.png, page.width, page.height, page.crops) {}

docv1::BoundingBox LayoutWriter::grid_box(const std::vector<double>& coords, double grid) const {
    if (coords.size() < 4 || grid <= 0) {
        return full_page_box(page_);
    }
    return scaled_box(coords[0], coords[1], coords[2], coords[3], page_.width / grid,
                      page_.height / grid);
}

docv1::BoundingBox LayoutWriter::scaled_box(double l, double t, double r, double b, double sx,
                                            double sy) const {
    const double width = page_.width;
    const double height = page_.height;
    docv1::BoundingBox box;
    box.set_l(std::clamp(l * sx, 0.0, width));
    box.set_t(std::clamp(t * sy, 0.0, height));
    box.set_r(std::clamp(r * sx, 0.0, width));
    box.set_b(std::clamp(b * sy, 0.0, height));
    box.set_coord_origin(docv1::COORD_ORIGIN_TOPLEFT);
    return box;
}

docv1::ProvenanceItem LayoutWriter::prov(const std::optional<docv1::BoundingBox>& box,
                                         int32_t chars) const {
    docv1::ProvenanceItem item = make_prov(page_, box ? *box : full_page_box(page_));
    item.mutable_charspan()->set_start(0);
    item.mutable_charspan()->set_end(chars);
    return item;
}

LayoutWriter::Ref LayoutWriter::add_entry(BodyChild::Kind kind, int index, Ref parent) {
    const Ref ref{kind, index};
    entries_.push_back({ref, parent});
    if (parent.is_body()) {
        body_.push_back(ref);
    } else {
        child_counts_[{static_cast<int>(parent.kind), parent.index}]++;
    }
    return ref;
}

size_t LayoutWriter::children_of(Ref ref) const {
    const auto it = child_counts_.find({static_cast<int>(ref.kind), ref.index});
    return it == child_counts_.end() ? 0 : it->second;
}

std::string LayoutWriter::text_of(Ref ref) const {
    if (ref.kind != BodyChild::TEXT) {
        return "";
    }
    const docv1::BaseTextItem& item = out_->texts(ref.index);
    if (item.has_code()) {
        return item.code().text();
    }
    docv1::BaseTextItem copy = item;
    const docv1::TextItemBase* base = base_of(&copy);
    return base != nullptr ? base->text() : "";
}

void LayoutWriter::set_text(Ref ref, const std::string& body, const docv1::Formatting* formatting,
                            const std::string& hyperlink) {
    if (ref.kind != BodyChild::TEXT) {
        return;
    }
    docv1::BaseTextItem* item = text_item(ref);
    if (item->has_code()) {
        item->mutable_code()->set_text(body);
        if (item->code().orig().empty()) {
            item->mutable_code()->set_orig(body);
        }
        return;
    }
    docv1::TextItemBase* base = base_of(item);
    if (base == nullptr) {
        return;
    }
    base->set_text(body);
    if (base->orig().empty()) {
        base->set_orig(body);
    }
    if (formatting != nullptr) {
        *base->mutable_formatting() = *formatting;
    }
    if (!hyperlink.empty()) {
        base->set_hyperlink(hyperlink);
    }
    // The charspan follows the text the item now holds.
    for (docv1::ProvenanceItem& prov : *base->mutable_prov()) {
        prov.mutable_charspan()->set_end(char_count(body));
    }
}

void LayoutWriter::set_orig(Ref ref, const std::string& orig) {
    if (ref.kind != BodyChild::TEXT) {
        return;
    }
    docv1::BaseTextItem* item = text_item(ref);
    if (item->has_code()) {
        item->mutable_code()->set_orig(orig);
    } else if (docv1::TextItemBase* base = base_of(item)) {
        base->set_orig(orig);
    }
}

void LayoutWriter::set_heading_level(int text_index, int level) {
    docv1::BaseTextItem* item = out_->mutable_texts(text_index);
    if (item->has_section_header()) {
        item->mutable_section_header()->set_level(level);
    }
}

void LayoutWriter::set_content_layer(Ref ref, docv1::ContentLayer layer) {
    switch (ref.kind) {
        case BodyChild::TEXT: {
            docv1::BaseTextItem* item = text_item(ref);
            if (item->has_code()) {
                item->mutable_code()->set_content_layer(layer);
            } else if (docv1::TextItemBase* base = base_of(item)) {
                base->set_content_layer(layer);
            }
            break;
        }
        case BodyChild::GROUP:
            out_->mutable_groups(ref.index)->set_content_layer(layer);
            break;
        default:
            break;
    }
}

LayoutWriter::Ref LayoutWriter::field_region(const std::optional<docv1::BoundingBox>& box,
                                             Ref parent) {
    docv1::FieldRegionItem* region = out_->add_field_regions();
    region->set_label(docv1::DOC_ITEM_LABEL_FIELD_REGION);
    region->set_content_layer(docv1::CONTENT_LAYER_BODY);
    *region->add_prov() = prov(box, 0);
    add_sources(region->mutable_source(), page_);
    return add_entry(BodyChild::FIELD_REGION, out_->field_regions_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::inline_group(Ref parent) {
    return group("group", docv1::GROUP_LABEL_INLINE, parent);
}

LayoutWriter::Ref LayoutWriter::styled_text(docv1::DocItemLabel label,
                                            const std::optional<docv1::BoundingBox>& box,
                                            const std::string& body,
                                            const docv1::Formatting* formatting,
                                            const std::string& hyperlink, Ref parent) {
    Ref ref;
    switch (label) {
        case docv1::DOC_ITEM_LABEL_TITLE:
            ref = title(box, body, parent);
            break;
        case docv1::DOC_ITEM_LABEL_SECTION_HEADER:
            ref = heading(1, box, body, parent);
            break;
        case docv1::DOC_ITEM_LABEL_FORMULA:
            ref = formula(box, body, parent);
            break;
        case docv1::DOC_ITEM_LABEL_CODE:
            ref = code(box, body, "", parent);
            break;
        case docv1::DOC_ITEM_LABEL_FIELD_VALUE: {
            docv1::BaseTextItem* item = out_->add_texts();
            fill_text_base(item->mutable_field_value()->mutable_base(), page_, label,
                           prov(box, char_count(body)), body);
            item->mutable_field_value()->set_kind("fillable");
            ref = add_entry(BodyChild::TEXT, out_->texts_size() - 1, parent);
            break;
        }
        default:
            ref = text(label, box, body, parent);
            break;
    }
    docv1::BaseTextItem* item = text_item(ref);
    if (docv1::TextItemBase* base = base_of(item)) {
        if (formatting != nullptr) {
            *base->mutable_formatting() = *formatting;
        }
        if (!hyperlink.empty()) {
            base->set_hyperlink(hyperlink);
        }
    }
    return ref;
}

void LayoutWriter::stamp_layer(docv1::BaseTextItem* item, docv1::DocItemLabel label) {
    if (label != docv1::DOC_ITEM_LABEL_PAGE_HEADER && label != docv1::DOC_ITEM_LABEL_PAGE_FOOTER) {
        return;
    }
    if (docv1::TextItemBase* base = base_of(item)) {
        base->set_content_layer(docv1::CONTENT_LAYER_FURNITURE);
    }
}

LayoutWriter::Ref LayoutWriter::text(docv1::DocItemLabel label,
                                     const std::optional<docv1::BoundingBox>& box,
                                     const std::string& body, Ref parent) {
    docv1::BaseTextItem* item = add_text(out_, page_, label, prov(box, char_count(body)), body);
    stamp_layer(item, label);
    return add_entry(BodyChild::TEXT, out_->texts_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::title(const std::optional<docv1::BoundingBox>& box,
                                      const std::string& body, Ref parent) {
    add_title(out_, page_, prov(box, char_count(body)), body);
    return add_entry(BodyChild::TEXT, out_->texts_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::heading(int level, const std::optional<docv1::BoundingBox>& box,
                                        const std::string& body, Ref parent) {
    add_section_header(out_, page_, level, prov(box, char_count(body)), body);
    return add_entry(BodyChild::TEXT, out_->texts_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::formula(const std::optional<docv1::BoundingBox>& box,
                                        const std::string& body, Ref parent) {
    add_formula(out_, page_, prov(box, char_count(body)), body);
    return add_entry(BodyChild::TEXT, out_->texts_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::code(const std::optional<docv1::BoundingBox>& box,
                                     const std::string& body, const std::string& language_raw,
                                     Ref parent) {
    add_code(out_, page_, prov(box, char_count(body)), body, language_raw);
    return add_entry(BodyChild::TEXT, out_->texts_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::list_item(const std::optional<docv1::BoundingBox>& box,
                                          const std::string& body, bool enumerated,
                                          const std::string& marker, Ref parent) {
    add_list_item(out_, page_, prov(box, char_count(body)), body, enumerated, marker);
    return add_entry(BodyChild::TEXT, out_->texts_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::list_group(Ref parent) {
    return group("list", docv1::GROUP_LABEL_LIST, parent);
}

LayoutWriter::Ref LayoutWriter::group(const std::string& name, docv1::GroupLabel label,
                                      Ref parent) {
    docv1::GroupItem* item = out_->add_groups();
    item->set_name(name);
    item->set_label(label);
    item->set_content_layer(docv1::CONTENT_LAYER_BODY);
    return add_entry(BodyChild::GROUP, out_->groups_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::picture(const std::optional<docv1::BoundingBox>& box,
                                        Ref parent) {
    docv1::PictureItem* item = add_picture(out_, page_, prov(box, 0));
    if (box && !page_.png.empty()) {
        docv1::ImageRef crop;
        switch (raster_.crop(box->l(), box->t(), box->r(), box->b(), &crop)) {
            case PageRaster::Crop::kAttached:
                *item->mutable_image() = std::move(crop);
                break;
            case PageRaster::Crop::kOverBudget:
                crops_over_budget_++;
                break;
            case PageRaster::Crop::kOverByteCap:
                crops_over_bytes_++;
                break;
            case PageRaster::Crop::kRasterTooLarge:
                crops_too_large_++;
                break;
            case PageRaster::Crop::kFailed:
                break;
        }
    }
    return add_entry(BodyChild::PICTURE, out_->pictures_size() - 1, parent);
}

LayoutWriter::Ref LayoutWriter::table(const std::optional<docv1::BoundingBox>& box,
                                      const std::vector<SpanCell>& cells, Ref parent) {
    docv1::TableItem* item = add_table(out_, page_, prov(box, 0));
    const Ref ref = add_entry(BodyChild::TABLE, out_->tables_size() - 1, parent);
    if (cells.empty()) {
        item->mutable_data();
        return ref;
    }
    const TableCut cut = fill_span_table(item->mutable_data(), cells);
    note_table_cut(cut, item->data(), "table", body_child_ref(BodyChild::TABLE, ref.index),
                   warnings_);
    return ref;
}

void LayoutWriter::describe_picture(Ref ref, const std::string& description,
                                    const std::string& created_by) {
    if (description.empty()) {
        return;
    }
    docv1::PictureItem* picture = picture_item(ref);
    auto* meta = picture->mutable_meta()->mutable_description();
    meta->set_text(description);
    meta->set_created_by(created_by);
    auto* annotation = picture->add_annotations()->mutable_description();
    annotation->set_kind("description");
    annotation->set_text(description);
    annotation->set_provenance(created_by);
}

void LayoutWriter::classify_picture(Ref ref, const std::string& class_name,
                                    const std::string& created_by) {
    if (class_name.empty()) {
        return;
    }
    docv1::PictureItem* picture = picture_item(ref);
    // A bare label carries no probability, so no confidence is claimed.
    auto* prediction = picture->mutable_meta()->mutable_classification()->add_predictions();
    prediction->set_class_name(class_name);
    prediction->set_created_by(created_by);
    auto* data = picture->add_annotations()->mutable_classification();
    data->set_kind("classification");
    data->set_provenance(created_by);
    data->add_predicted_classes()->set_class_name(class_name);
}

void LayoutWriter::set_molecule(Ref ref, const std::string& smiles,
                                const std::string& created_by) {
    if (smiles.empty()) {
        return;
    }
    docv1::PictureItem* picture = picture_item(ref);
    auto* meta = picture->mutable_meta()->mutable_molecule();
    meta->set_smi(smiles);
    meta->set_created_by(created_by);
    auto* annotation = picture->add_annotations()->mutable_molecule();
    annotation->set_kind("molecule_data");
    annotation->set_smi(smiles);
    annotation->set_provenance(created_by);
}

void LayoutWriter::set_tabular_chart(Ref ref, const std::string& title,
                                     const std::vector<std::vector<std::string>>& rows,
                                     const std::string& created_by) {
    if (rows.empty()) {
        return;
    }
    docv1::TableData data;
    const TableCut cut = fill_table_data(&data, rows, 1);
    note_table_cut(cut, data, "chart data", body_child_ref(BodyChild::PICTURE, ref.index),
                   warnings_);
    docv1::PictureItem* picture = picture_item(ref);
    auto* meta = picture->mutable_meta()->mutable_tabular_chart();
    if (!title.empty()) {
        meta->set_title(title);
    }
    if (!created_by.empty()) {
        meta->set_created_by(created_by);
    }
    *meta->mutable_chart_data() = data;
    auto* annotation = picture->add_annotations()->mutable_tabular_chart();
    annotation->set_kind("tabular_chart_data");
    if (!title.empty()) {
        annotation->set_title(title);
    }
    *annotation->mutable_chart_data() = std::move(data);
}

void LayoutWriter::link_caption(Ref owner, Ref caption) {
    const std::string ref = body_child_ref(caption.kind, caption.index);
    if (owner.kind == BodyChild::PICTURE) {
        picture_item(owner)->add_captions()->set_ref(ref);
    } else if (owner.kind == BodyChild::TABLE) {
        table_item(owner)->add_captions()->set_ref(ref);
    } else if (owner.kind == BodyChild::TEXT && text_item(owner)->has_code()) {
        text_item(owner)->mutable_code()->add_captions()->set_ref(ref);
    }
}

void LayoutWriter::link_footnote(Ref owner, Ref footnote) {
    const std::string ref = body_child_ref(footnote.kind, footnote.index);
    if (owner.kind == BodyChild::PICTURE) {
        picture_item(owner)->add_footnotes()->set_ref(ref);
    } else if (owner.kind == BodyChild::TABLE) {
        table_item(owner)->add_footnotes()->set_ref(ref);
    } else if (owner.kind == BodyChild::TEXT && text_item(owner)->has_code()) {
        text_item(owner)->mutable_code()->add_footnotes()->set_ref(ref);
    }
}

bool LayoutWriter::has_captions(Ref owner) const {
    if (owner.kind == BodyChild::PICTURE) {
        return out_->pictures(owner.index).captions_size() > 0;
    }
    if (owner.kind == BodyChild::TABLE) {
        return out_->tables(owner.index).captions_size() > 0;
    }
    return false;
}

docv1::DocItemLabel LayoutWriter::label_of(Ref ref) const {
    switch (ref.kind) {
        case BodyChild::PICTURE:
            return docv1::DOC_ITEM_LABEL_PICTURE;
        case BodyChild::TABLE:
            return docv1::DOC_ITEM_LABEL_TABLE;
        case BodyChild::TEXT: {
            const docv1::BaseTextItem& item = out_->texts(ref.index);
            if (item.has_code()) {
                return docv1::DOC_ITEM_LABEL_CODE;
            }
            switch (item.item_case()) {
                case docv1::BaseTextItem::kTitle:
                    return item.title().base().label();
                case docv1::BaseTextItem::kSectionHeader:
                    return item.section_header().base().label();
                case docv1::BaseTextItem::kListItem:
                    return item.list_item().base().label();
                case docv1::BaseTextItem::kFormula:
                    return item.formula().base().label();
                case docv1::BaseTextItem::kText:
                    return item.text().base().label();
                default:
                    return docv1::DOC_ITEM_LABEL_UNSPECIFIED;
            }
        }
        default:
            return docv1::DOC_ITEM_LABEL_UNSPECIFIED;
    }
}

void LayoutWriter::link_adjacent_captions(const std::vector<size_t>& starts,
                                          const std::vector<size_t>& caption_blocks,
                                          bool code_owners) {
    const size_t blocks = starts.size();
    auto block = [&](size_t i) {
        const size_t begin = starts[i];
        const size_t end = i + 1 < blocks ? starts[i + 1] : body_.size();
        return std::pair<size_t, size_t>{begin, end};
    };
    for (size_t index : caption_blocks) {
        if (index >= blocks) {
            continue;
        }
        const auto [begin, end] = block(index);
        if (end - begin != 1) {
            continue;
        }
        const Ref caption = body_[begin];
        if (caption.kind != BodyChild::TEXT ||
            label_of(caption) != docv1::DOC_ITEM_LABEL_CAPTION) {
            continue;
        }
        const docv1::BaseTextItem& caption_item = out_->texts(caption.index);
        if (!caption_item.has_text() || trim(caption_item.text().base().text()).empty() ||
            children_of(caption) != 0) {
            continue;
        }
        std::vector<Ref> owners;
        for (const size_t neighbor : {index - 1, index + 1}) {
            if (neighbor >= blocks) {  // index - 1 wraps for index 0
                continue;
            }
            const auto [nb, ne] = block(neighbor);
            if (ne - nb != 1) {
                continue;
            }
            const Ref owner = body_[nb];
            if (owner.kind == BodyChild::PICTURE || owner.kind == BodyChild::TABLE ||
                (code_owners && owner.kind == BodyChild::TEXT &&
                 out_->texts(owner.index).has_code())) {
                owners.push_back(owner);
            }
        }
        // A code owner counts toward ambiguity but is never linked (docling
        // core cannot serialize code caption heads).
        if (owners.size() == 1 && owners[0].kind != BodyChild::TEXT &&
            !has_captions(owners[0])) {
            link_caption(owners[0], caption);
        }
    }
}

bool LayoutWriter::finish(std::string* error, const std::string& format_name,
                          bool allow_empty) {
    if (entries_.empty() && !allow_empty) {
        *error = format_name + " response produced no document items";
        return false;
    }
    auto children_of = [&](Ref ref) -> google::protobuf::RepeatedPtrField<docv1::RefItem>* {
        switch (ref.kind) {
            case BodyChild::GROUP:
                return out_->mutable_groups(ref.index)->mutable_children();
            case BodyChild::PICTURE:
                return out_->mutable_pictures(ref.index)->mutable_children();
            case BodyChild::TABLE:
                return out_->mutable_tables(ref.index)->mutable_children();
            case BodyChild::FIELD_REGION:
                return out_->mutable_field_regions(ref.index)->mutable_children();
            case BodyChild::TEXT: {
                docv1::BaseTextItem* item = text_item(ref);
                if (item->has_code()) {
                    return item->mutable_code()->mutable_children();
                }
                docv1::TextItemBase* base = base_of(item);
                return base != nullptr ? base->mutable_children() : nullptr;
            }
            default:
                return nullptr;
        }
    };
    // The body itself, the page entry and every body child come from the
    // shared finalizer; nested entries are stamped here.
    std::vector<BodyChild> order;
    order.reserve(body_.size());
    for (const Ref& ref : body_) {
        order.push_back({ref.kind, ref.index});
    }
    finalize_document(out_, page_, order);
    for (const Entry& entry : entries_) {
        if (entry.parent.is_body()) {
            continue;
        }
        const std::string self_ref = body_child_ref(entry.ref.kind, entry.ref.index);
        const std::string parent_ref = body_child_ref(entry.parent.kind, entry.parent.index);
        switch (entry.ref.kind) {
            case BodyChild::TEXT:
                internal::stamp_text_item(*text_item(entry.ref), self_ref, parent_ref);
                break;
            case BodyChild::PICTURE:
                picture_item(entry.ref)->set_self_ref(self_ref);
                picture_item(entry.ref)->mutable_parent()->set_ref(parent_ref);
                break;
            case BodyChild::TABLE:
                table_item(entry.ref)->set_self_ref(self_ref);
                table_item(entry.ref)->mutable_parent()->set_ref(parent_ref);
                break;
            case BodyChild::GROUP:
                out_->mutable_groups(entry.ref.index)->set_self_ref(self_ref);
                out_->mutable_groups(entry.ref.index)->mutable_parent()->set_ref(parent_ref);
                break;
            case BodyChild::FIELD_REGION:
                out_->mutable_field_regions(entry.ref.index)->set_self_ref(self_ref);
                out_->mutable_field_regions(entry.ref.index)->mutable_parent()->set_ref(parent_ref);
                break;
            default:
                break;
        }
        if (auto* children = children_of(entry.parent)) {
            children->Add()->set_ref(self_ref);
        }
    }
    if (crops_over_budget_ > 0) {
        add_warning(warnings_, vlmv1::PAGE_WARNING_CODE_PICTURE_IMAGES_SKIPPED,
                    std::to_string(crops_over_budget_) +
                        " picture(s) carry no image: the page's crop budget (" +
                        std::to_string(page_.crops.max_crops) + " crops, " +
                        std::to_string(static_cast<int>(page_.crops.max_area_pages)) +
                        "x the page's pixels) ran out",
                    "");
    }
    if (crops_over_bytes_ > 0) {
        add_warning(warnings_, vlmv1::PAGE_WARNING_CODE_PICTURE_IMAGES_SKIPPED,
                    std::to_string(crops_over_bytes_) +
                        " picture(s) carry no image: the page's inline image cap (" +
                        std::to_string(page_.crops.max_inline_bytes) + " bytes, " +
                        std::to_string(raster_.inline_bytes()) + " attached) ran out",
                    "");
    }
    if (crops_too_large_ > 0) {
        add_warning(warnings_, vlmv1::PAGE_WARNING_CODE_PICTURE_IMAGES_SKIPPED,
                    std::to_string(crops_too_large_) +
                        " picture(s) carry no image: the page raster is " +
                        std::to_string(raster_.width()) + "x" + std::to_string(raster_.height()) +
                        ", above the " + std::to_string(kMaxRasterPixels) +
                        "-pixel decode cap",
                    "");
    }
    return true;
}

}  // namespace vlm::mapping
