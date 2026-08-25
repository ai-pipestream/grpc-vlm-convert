// Unit tests for the DocTags mapper against canned model output: labels,
// real location boxes scaled from the 0..500 grid, furniture layers,
// tables via OTSL, and the no-markup failure that drives PageRaw.

#include "fixture.h"
#include "mapping/mapper.h"

namespace docv1 = ai::pipestream::document::v1;
namespace vlmv1 = ai::pipestream::vlm::v1;

namespace {

vlm::mapping::PageContext page_context() {
    vlm::mapping::PageContext page;
    page.page_no = 1;
    page.width = 1000;
    page.height = 1000;  // square page: grid units scale by exactly 2
    page.source.set_collector("vlm-convert");
    page.source.set_model("test-model");
    return page;
}

// Generation provenance is stamped by the item builders, so it must reach
// every item kind that has a source list, not only the text items the
// service test walks.
void verify_generation_source_on_every_item() {
    vlm::mapping::PageContext page = page_context();
    page.has_generation = true;
    page.generation.set_model("served-model");
    page.generation.set_endpoint("http://vlm:8080");
    page.generation.set_finish_reason("length");
    page.generation.set_completion_tokens(4096);

    const std::string text =
        "<doctag>"
        "<text><loc_50><loc_200><loc_400><loc_260>Body text.</text>"
        "<picture><loc_100><loc_100><loc_300><loc_300><logo></picture>"
        "<otsl><loc_10><loc_10><loc_400><loc_400><fcel>a<fcel>b<nl></otsl>"
        "<key_value_region><loc_10><loc_10><loc_400><loc_400>"
        "<key_0><loc_10><loc_10><loc_100><loc_40>Name</key_0>"
        "<value_0><loc_110><loc_10><loc_300><loc_40>Ada</value_0>"
        "</key_value_region>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page, &doc, &error),
            "mixed page maps: " + error);

    auto check = [](const google::protobuf::RepeatedPtrField<docv1::SourceType>& sources,
                    const std::string& what) {
        require(sources.size() == 2, what + " carries both sources");
        require(sources.Get(0).has_collector(), what + " keeps its collector source");
        require(sources.Get(1).has_generation(), what + " gains a generation source");
        require(sources.Get(1).generation().model() == "served-model" &&
                    sources.Get(1).generation().endpoint() == "http://vlm:8080" &&
                    sources.Get(1).generation().finish_reason() == "length" &&
                    sources.Get(1).generation().completion_tokens() == 4096,
                what + " carries the generation facts");
    };
    require(doc.texts_size() == 1 && doc.pictures_size() == 1 && doc.tables_size() == 1 &&
                doc.key_value_items_size() == 1,
            "one item of each kind");
    check(doc.texts(0).text().base().source(), "a text item");
    check(doc.pictures(0).source(), "a picture item");
    check(doc.tables(0).source(), "a table item");
    check(doc.key_value_items(0).source(), "a key-value item");

    // A page with no generation to attribute stamps the collector alone
    // rather than an empty shell.
    docv1::Document plain;
    require(vlm::mapping::map_doctags(text, page_context(), &plain, &error),
            "mixed page maps without a generation: " + error);
    require(plain.texts(0).text().base().source_size() == 1,
            "no generation means no generation source");
}

void verify_heading_paragraph_boxes() {
    const std::string text =
        "<doctag>"
        "<section_header_level_1><loc_50><loc_100><loc_400><loc_150>Introduction"
        "</section_header_level_1>"
        "<text><loc_50><loc_200><loc_400><loc_260>Some body text.</text>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "doctags page maps: " + error);
    require(doc.texts_size() == 2, "one heading + one paragraph");
    require(doc.texts(0).has_section_header(), "first item is a section header");
    require(doc.texts(0).section_header().level() == 1, "heading level 1");
    require(doc.texts(0).section_header().base().text() == "Introduction", "heading text");
    require(doc.texts(1).has_text(), "second item is text");
    require(doc.texts(1).text().base().label() == docv1::DOC_ITEM_LABEL_TEXT, "text label");

    const docv1::BoundingBox& box = doc.texts(0).section_header().base().prov(0).bbox();
    require(box.coord_origin() == docv1::COORD_ORIGIN_TOPLEFT, "boxes are TOPLEFT");
    require(box.l() == 100 && box.t() == 200 && box.r() == 800 && box.b() == 300,
            "grid locations scale to page pixels");
    require(doc.texts(0).section_header().base().prov(0).page_no() == 1, "page provenance");
    require(doc.texts(0).section_header().base().source(0).collector().collector() ==
                "vlm-convert",
            "collector source stamped");
    require(doc.texts(0).section_header().base().source(0).collector().model() == "test-model",
            "model source stamped");

    require(doc.body().self_ref() == "#/body", "body group present");
    require(doc.body().children_size() == 2, "body lists the items");
    require(doc.body().children(0).ref() == "#/texts/0", "child refs in order");
    require(doc.pages().at(1).size().width() == 1000, "pages entry carries raster size");
    require(doc.schema_name() == "docling_document_v2", "root schema name stamped");
    require(doc.version() == "1.10.0", "root schema version stamped");
}

void verify_table_picture_furniture() {
    const std::string text =
        "<doctag>"
        "<page_header><loc_50><loc_10><loc_400><loc_30>ACME Corp</page_header>"
        "<picture><loc_100><loc_100><loc_300><loc_300></picture>"
        "<table><loc_50><loc_400><loc_450><loc_480>"
        "<otsl><ched>Name<ched>Qty<nl><fcel>bolts<fcel>12<nl></otsl>"
        "</table>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "doctags with table maps: " + error);
    require(doc.texts_size() == 1, "page header is a text item");
    require(doc.texts(0).text().base().label() == docv1::DOC_ITEM_LABEL_PAGE_HEADER,
            "page header label");
    require(doc.texts(0).text().base().content_layer() == docv1::CONTENT_LAYER_FURNITURE,
            "page header is furniture");
    require(doc.pictures_size() == 1, "picture item");
    require(doc.pictures(0).label() == docv1::DOC_ITEM_LABEL_PICTURE, "picture label");
    require(doc.tables_size() == 1, "table item");
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 2 && data.num_cols() == 2, "otsl grid shape");
    require(data.grid(0).cells(0).text() == "Name", "header cell text");
    require(data.grid(0).cells(0).column_header(), "ched marks a column header");
    require(data.grid(1).cells(1).text() == "12", "body cell text");
    require(!data.grid(1).cells(1).column_header(), "fcel is not a header");
}

void verify_unclosed_and_unknown_tags() {
    // SmolDocling does not always close elements; unknown tags keep their
    // text rather than dropping it.
    const std::string text =
        "<doctag><title><loc_50><loc_50><loc_400><loc_90>Doc Title"
        "<text><loc_50><loc_120><loc_400><loc_160>First paragraph."
        "<page_footer><loc_50><loc_950><loc_400><loc_980>1</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "unclosed elements still map: " + error);
    require(doc.texts_size() == 3, "title + text + footer");
    require(doc.texts(0).has_title(), "title item");
    require(doc.texts(0).title().base().label() == docv1::DOC_ITEM_LABEL_TITLE, "title label");
    require(doc.texts(2).text().base().label() == docv1::DOC_ITEM_LABEL_PAGE_FOOTER,
            "footer label");
}

void verify_mapping_failures() {
    docv1::Document doc;
    std::string error;
    require(!vlm::mapping::map_doctags("plain prose, no markup at all", page_context(), &doc,
                                       &error),
            "no markup is a mapping failure");
    require(!error.empty(), "failure carries a reason");
    error.clear();
    require(!vlm::mapping::map_doctags("<doctag></doctag>", page_context(), &doc, &error),
            "empty doctag wrapper is a mapping failure");
}

std::string base64_decode(const std::string& encoded) {
    auto value_of = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    int group = 0, bits = 0;
    for (char c : encoded) {
        int value = value_of(c);
        if (value < 0) {
            continue;
        }
        group = (group << 6) | value;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((group >> bits) & 0xFF));
        }
    }
    return out;
}

// A real 100x50 PNG (quadrant colors) so the crop path has something to
// decode; fake signature-only PNGs exercise the no-image fallback.
const std::string kRealPngBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAGQAAAAyCAIAAAAlV+npAAAApUlEQVR4nO2QwQnAMACEsv/S7QL53MsIigOI"
    "5zvnQfkCkXyBSL5AJF8gki8QyReI5AtE8gUi+QKRfIFIvkAkXyCSLxDJF4jkC0TyBSL5ApF8gUi+QCRfIJ"
    "IvEMkXiOQLRPIFIvmCexafcPFNmjXQrIFmDTRroFkDzRpo1kCzBpo10KyBZg00a6BZA80aaNZAswaaNdCs"
    "gWYNNGugWQPNGmjWQLMGfpOeMGgRmrqiAAAAAElFTkSuQmCC";

void verify_embedded_otsl_spans() {
    const std::string text =
        "<doctag><table><loc_50><loc_400><loc_450><loc_480>"
        "<otsl><fcel>A<lcel><nl><ucel><xcel><nl><fcel>B<fcel>C<nl></otsl>"
        "</table></doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "table with spans maps: " + error);
    const docv1::TableData& data = doc.tables(0).data();
    require(data.num_rows() == 3 && data.num_cols() == 2, "embedded grid shape");
    require(data.table_cells_size() == 3, "embedded fillers suppressed");
    require(data.table_cells(0).row_span() == 2 && data.table_cells(0).col_span() == 2,
            "embedded 2d span on the anchor");
    require(data.grid(1).cells(1).text() == "A", "embedded anchor covers the grid region");
}

void verify_code_language() {
    const std::string text =
        "<doctag>"
        "<code><loc_50><loc_100><loc_400><loc_150><_Python_>print('hi')</code>"
        "<code><loc_50><loc_200><loc_400><loc_250><_cobol-x_>MOVE A TO B</code>"
        "<code><loc_50><loc_300><loc_400><loc_350>int main() {}</code>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "code items map: " + error);
    require(doc.texts_size() == 3, "three code items");
    const docv1::CodeItem& python = doc.texts(0).code();
    require(python.code_language() == docv1::CODE_LANGUAGE_LABEL_PYTHON,
            "language token selects the enum");
    require(python.code_language_raw() == "Python", "raw language kept");
    require(python.text() == "print('hi')", "language token stripped from the text");
    // Docling matches the token case-sensitively; unknown tokens are
    // UNKNOWN with the raw string kept.
    const docv1::CodeItem& unknown = doc.texts(1).code();
    require(unknown.code_language() == docv1::CODE_LANGUAGE_LABEL_UNKNOWN,
            "unmatched token falls back to UNKNOWN");
    require(unknown.code_language_raw() == "cobol-x", "unmatched raw kept");
    const docv1::CodeItem& plain = doc.texts(2).code();
    require(plain.code_language() == docv1::CODE_LANGUAGE_LABEL_UNKNOWN,
            "no token is UNKNOWN");
    require(!plain.has_code_language_raw(), "no token, no raw");
}

void verify_charspan() {
    const std::string text =
        "<doctag>"
        "<section_header_level_1><loc_50><loc_100><loc_400><loc_150>Introduction"
        "</section_header_level_1>"
        "<picture><loc_100><loc_100><loc_300><loc_300></picture>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "charspan page maps: " + error);
    const docv1::ProvenanceItem& text_prov = doc.texts(0).section_header().base().prov(0);
    require(text_prov.has_charspan(), "text provenance carries charspan");
    require(text_prov.charspan().start() == 0 && text_prov.charspan().end() == 12,
            "charspan is [0, len(text))");
    const docv1::ProvenanceItem& pic_prov = doc.pictures(0).prov(0);
    require(pic_prov.has_charspan() && pic_prov.charspan().start() == 0 &&
                pic_prov.charspan().end() == 0,
            "floating items get charspan [0, 0]");
}

void verify_picture_classification() {
    const std::string text =
        "<doctag>"
        "<picture><loc_100><loc_100><loc_300><loc_300><logo></picture>"
        "<picture><loc_100><loc_300><loc_300><loc_450><line></picture>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "classified pictures map: " + error);
    require(doc.pictures_size() == 2, "two pictures");
    const docv1::PictureItem& logo = doc.pictures(0);
    require(logo.annotations_size() == 1 && logo.annotations(0).has_classification(),
            "classification annotation attached");
    const docv1::PictureClassificationData& data = logo.annotations(0).classification();
    require(data.provenance() == "load_from_doctags", "classification provenance");
    require(data.predicted_classes_size() == 1 &&
                data.predicted_classes(0).class_name() == "logo",
            "predicted class named");
    require(logo.meta().classification().predictions_size() == 1 &&
                logo.meta().classification().predictions(0).created_by() == "load_from_doctags",
            "meta prediction carries created_by");
    // The tag is a bare label: no probability is reported, so none is
    // claimed. Fabricated certainty survives into every downstream
    // ranking, which is why this assertion is the point of the test.
    require(!logo.meta().classification().predictions(0).has_confidence(),
            "a label with no probability behind it claims no confidence");
    require(data.predicted_classes(0).confidence() == 0.0,
            "the annotation class asserts no confidence either");
    // Legacy SmolDocling alias: <line> maps to line_chart.
    require(doc.pictures(1).annotations(0).classification().predicted_classes(0).class_name() ==
                "line_chart",
            "legacy alias maps to the v2 class");
}

// The model's own words inside a picture chunk are the only description
// of that region anyone will ever get; they used to be parsed and thrown
// away.
void verify_picture_description() {
    const std::string text =
        "<doctag>"
        "<picture><loc_100><loc_100><loc_300><loc_300><photograph>"
        "A red tractor parked in front of a barn."
        "<caption><loc_100><loc_310><loc_300><loc_330>Figure 2</caption>"
        "</picture>"
        "<picture><loc_10><loc_10><loc_60><loc_60><icon></picture>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "described picture maps: " + error);
    require(doc.pictures_size() == 2, "two pictures");
    const docv1::PictureItem& described = doc.pictures(0);
    require(described.meta().description().text() ==
                "A red tractor parked in front of a barn.",
            "the region text becomes the picture description");
    require(described.meta().description().created_by() == "load_from_doctags",
            "the description names its producer");
    require(!described.meta().description().has_confidence(),
            "a transcribed description claims no confidence");
    bool saw_description = false;
    for (const docv1::PictureAnnotation& annotation : described.annotations()) {
        if (annotation.has_description()) {
            saw_description = true;
            require(annotation.description().kind() == "description" &&
                        annotation.description().provenance() == "load_from_doctags",
                    "description annotation kind and provenance");
            require(annotation.description().text().contains("red tractor"),
                    "description annotation carries the same text");
        }
    }
    require(saw_description, "the description also lands in the annotations union");
    // The caption is a separate linked item, never folded into the
    // description.
    require(described.captions_size() == 1, "caption still links as its own item");
    require(!described.meta().description().text().contains("Figure 2"),
            "caption text stays out of the description");
    // A picture the model said nothing about claims no description.
    require(!doc.pictures(1).meta().has_description(),
            "a picture with no inner text gets no description");
}

void verify_chart() {
    const std::string text =
        "<doctag>"
        "<chart><loc_100><loc_100><loc_400><loc_400><bar_chart>"
        "<otsl><ched>Q<ched>Sales<nl><fcel>Q1<fcel>10<nl></otsl>"
        "</chart>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "chart maps: " + error);
    require(doc.pictures_size() == 1, "chart is a picture item");
    require(doc.tables_size() == 0, "chart data is not a table item");
    const docv1::PictureItem& chart = doc.pictures(0);
    require(chart.label() == docv1::DOC_ITEM_LABEL_PICTURE, "chart keeps the picture label");

    bool saw_classification = false;
    bool saw_tabular = false;
    for (const docv1::PictureAnnotation& annotation : chart.annotations()) {
        if (annotation.has_classification()) {
            saw_classification = true;
            require(annotation.classification().predicted_classes(0).class_name() == "bar_chart",
                    "chart classification");
        }
        if (annotation.has_tabular_chart()) {
            saw_tabular = true;
            require(annotation.tabular_chart().title() == "bar_chart",
                    "tabular chart title is the classification");
            const docv1::TableData& data = annotation.tabular_chart().chart_data();
            require(data.num_rows() == 2 && data.num_cols() == 2, "chart data grid shape");
            require(data.grid(0).cells(0).column_header(), "chart data keeps header flags");
            require(data.grid(1).cells(1).text() == "10", "chart data cell text");
        }
    }
    require(saw_classification && saw_tabular, "classification + tabular chart annotations");
    require(chart.meta().tabular_chart().chart_data().num_rows() == 2,
            "meta tabular chart carries the same data");
}

void verify_source_order() {
    const std::string text =
        "<doctag>"
        "<text><loc_50><loc_100><loc_400><loc_150>First.</text>"
        "<picture><loc_100><loc_200><loc_300><loc_400></picture>"
        "<text><loc_50><loc_450><loc_400><loc_500>Second.</text>"
        "<table><loc_50><loc_550><loc_450><loc_600><otsl><fcel>x<nl></otsl></table>"
        "<text><loc_50><loc_650><loc_400><loc_700>Third.</text>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "mixed page maps: " + error);
    require(doc.body().children_size() == 5, "all items are body children");
    const char* expected[] = {"#/texts/0", "#/pictures/0", "#/texts/1", "#/tables/0",
                              "#/texts/2"};
    for (int i = 0; i < 5; i++) {
        require(doc.body().children(i).ref() == expected[i],
                std::string("body children follow token order at ") + std::to_string(i) +
                    ": " + doc.body().children(i).ref());
    }
    require(doc.texts(1).text().base().self_ref() == "#/texts/1", "self refs match the order");
    require(doc.pictures(0).parent().ref() == "#/body", "parents stamped");
}

void verify_table_caption() {
    const std::string text =
        "<doctag>"
        "<table><loc_50><loc_400><loc_450><loc_480>"
        "<otsl><fcel>x<nl></otsl>"
        "<caption><loc_60><loc_485><loc_300><loc_495>Table 1: Parts</caption>"
        "</table>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "captioned table maps: " + error);
    require(doc.texts_size() == 1 && doc.tables_size() == 1, "caption item + table item");
    const docv1::TextItemBase& caption = doc.texts(0).text().base();
    require(caption.label() == docv1::DOC_ITEM_LABEL_CAPTION, "caption label");
    require(caption.text() == "Table 1: Parts", "caption text");
    require(caption.prov(0).charspan().end() == 14, "caption charspan");
    require(caption.prov(0).bbox().l() == 120 && caption.prov(0).bbox().t() == 970,
            "caption carries its own box");
    require(doc.tables(0).captions_size() == 1 &&
                doc.tables(0).captions(0).ref() == "#/texts/0",
            "table links the caption");
    // Docling adds the caption before the table it belongs to.
    require(doc.body().children(0).ref() == "#/texts/0" &&
                doc.body().children(1).ref() == "#/tables/0",
            "caption precedes its table in body order");
}

void verify_picture_crop() {
    vlm::mapping::PageContext page = page_context();
    page.png = base64_decode(kRealPngBase64);
    // Declared page 1000x1000, real raster 100x50: locs 0..250 scale to
    // 500 page px, then to 50x25 raster px (the red quadrant).
    const std::string text =
        "<doctag>"
        "<picture><loc_0><loc_0><loc_250><loc_250></picture>"
        "<picture></picture>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page, &doc, &error),
            "cropped pictures map: " + error);
    require(doc.pictures_size() == 2, "two pictures");
    const docv1::PictureItem& cropped = doc.pictures(0);
    require(cropped.has_image(), "region crop attached");
    require(cropped.image().mimetype() == "image/png", "crop mimetype");
    require(cropped.image().size().width() == 50 && cropped.image().size().height() == 25,
            "crop scales the box into raster pixels");
    require(cropped.image().uri().starts_with("data:image/png;base64,"),
            "crop is a PNG data URI");
    // No locs → no crop, but the PictureItem still lands.
    require(!doc.pictures(1).has_image(), "no box, no crop, picture still emitted");

    // Undecodable page bytes (the fake signature-only PNGs) also degrade
    // to a picture without an image instead of failing the page.
    vlm::mapping::PageContext fake = page_context();
    fake.png = make_png("NOT-A-REAL-PNG");
    docv1::Document doc2;
    require(vlm::mapping::map_doctags(
                "<doctag><picture><loc_0><loc_0><loc_250><loc_250></picture></doctag>", fake,
                &doc2, &error),
            "undecodable raster still maps: " + error);
    require(doc2.pictures_size() == 1 && !doc2.pictures(0).has_image(),
            "decode failure skips the image silently");
}

void verify_key_value_region() {
    const std::string text =
        "<doctag>"
        "<key_value_region><loc_50><loc_100><loc_450><loc_300>"
        "<key_0><loc_60><loc_110><loc_200><loc_140>Name<link_1></key_0>"
        "<value_1><loc_220><loc_110><loc_440><loc_140>Alice</value_1>"
        "<key_2><loc_60><loc_160><loc_200><loc_190>Age<link_9></key_2>"
        "<value_3><loc_220><loc_160><loc_440><loc_190>37</value_3>"
        "</key_value_region>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "key_value_region maps: " + error);
    require(doc.key_value_items_size() == 1, "one key_value item");
    const docv1::KeyValueItem& kv = doc.key_value_items(0);
    require(kv.label() == docv1::DOC_ITEM_LABEL_KEY_VALUE_REGION, "key value label");
    require(kv.content_layer() == docv1::CONTENT_LAYER_BODY, "key value is body content");
    require(kv.self_ref() == "#/key_value_items/0", "key value self ref");
    require(kv.parent().ref() == "#/body", "key value parented to the body");
    require(doc.body().children_size() == 1 &&
                doc.body().children(0).ref() == "#/key_value_items/0",
            "body child ref follows source order");
    // The region's own provenance: locs ahead of the first <key_N> cell,
    // charspan [0, 0] like docling's floating items.
    require(kv.prov_size() == 1, "region provenance from the leading locs");
    require(kv.prov(0).charspan().start() == 0 && kv.prov(0).charspan().end() == 0,
            "region charspan is [0, 0]");
    const docv1::BoundingBox& box = kv.prov(0).bbox();
    require(box.l() == 100 && box.t() == 200 && box.r() == 900 && box.b() == 600,
            "region box scales from the grid");

    const docv1::GraphData& graph = kv.graph();
    require(graph.cells_size() == 4, "four key/value cells");
    require(graph.cells(0).label() == docv1::GRAPH_CELL_LABEL_KEY &&
                graph.cells(0).cell_id() == 0 && graph.cells(0).text() == "Name",
            "key cell: label, id, cleaned text");
    require(graph.cells(1).label() == docv1::GRAPH_CELL_LABEL_VALUE &&
                graph.cells(1).cell_id() == 1 && graph.cells(1).text() == "Alice",
            "value cell");
    require(graph.cells(2).text() == "Age" && graph.cells(3).text() == "37",
            "second pair, link token stripped");
    require(graph.cells(0).orig() == "Name", "orig mirrors text like docling");
    require(graph.cells(0).has_prov(), "cells carry their own boxes");
    const docv1::BoundingBox& cell_box = graph.cells(0).prov().bbox();
    require(cell_box.l() == 120 && cell_box.t() == 220 && cell_box.r() == 400 &&
                cell_box.b() == 280,
            "cell box scales from the grid");
    require(graph.cells(0).prov().charspan().end() == 0, "cell charspan is [0, 0]");
    // <link_1> on key_0 links to value_1; <link_9> points at no cell and
    // is dropped like docling's validation.
    require(graph.links_size() == 1, "only the valid link survives");
    require(graph.links(0).label() == docv1::GRAPH_LINK_LABEL_TO_VALUE, "TO_VALUE link");
    require(graph.links(0).source_cell_id() == 0 && graph.links(0).target_cell_id() == 1,
            "link endpoints are cell ids");
}

void verify_list_groups() {
    const std::string text =
        "<doctag>"
        "<text><loc_50><loc_50><loc_400><loc_90>Intro</text>"
        "<ordered_list>"
        "<list_item><loc_50><loc_100><loc_400><loc_130>First</list_item>"
        "<list_item><loc_50><loc_140><loc_400><loc_170>Second</list_item>"
        "</ordered_list>"
        "<unordered_list>"
        "<list_item><loc_50><loc_200><loc_400><loc_230>Bullet</list_item>"
        "</unordered_list>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "list groups map: " + error);
    require(doc.groups_size() == 2, "one group per list chunk");
    require(doc.texts_size() == 4, "intro + three list items in source order");

    const docv1::GroupItem& ordered = doc.groups(0);
    require(ordered.self_ref() == "#/groups/0", "group self ref");
    require(ordered.name() == "list", "docling names list groups 'list'");
    require(ordered.label() == docv1::GROUP_LABEL_LIST,
            "docling folds ORDERED_LIST onto LIST");
    require(ordered.parent().ref() == "#/body", "group parented to the body");
    require(ordered.content_layer() == docv1::CONTENT_LAYER_BODY, "group is body content");
    require(ordered.children_size() == 2 && ordered.children(0).ref() == "#/texts/1" &&
                ordered.children(1).ref() == "#/texts/2",
            "ordered list children in source order");

    const docv1::ListItem& first = doc.texts(1).list_item();
    require(first.enumerated() && first.marker() == "1.", "ordered marker 1.");
    require(doc.texts(2).list_item().enumerated() &&
                doc.texts(2).list_item().marker() == "2.",
            "ordered marker 2.");
    require(!doc.texts(3).list_item().enumerated() &&
                !doc.texts(3).list_item().has_marker(),
            "unordered items carry no marker");
    require(first.base().parent().ref() == "#/groups/0", "items parented to the group");
    require(doc.texts(3).list_item().base().parent().ref() == "#/groups/1",
            "unordered item parented to its own group");
    // List items keep their own provenance and charspan.
    const docv1::ProvenanceItem& prov = first.base().prov(0);
    require(prov.charspan().start() == 0 && prov.charspan().end() == 5, "item charspan");
    require(prov.bbox().l() == 100 && prov.bbox().t() == 200 && prov.bbox().r() == 800 &&
                prov.bbox().b() == 260,
            "item keeps its own box");

    // Source-order emission: the group refs sit where the chunks appeared.
    require(doc.body().children_size() == 3, "body children");
    require(doc.body().children(0).ref() == "#/texts/0" &&
                doc.body().children(1).ref() == "#/groups/0" &&
                doc.body().children(2).ref() == "#/groups/1",
            "body children follow token order");
    require(doc.groups(1).children_size() == 1 &&
                doc.groups(1).children(0).ref() == "#/texts/3",
            "unordered group children");
}

void verify_inline_group() {
    const std::string text =
        "<doctag>"
        "<inline><loc_50><loc_100><loc_400><loc_200>"
        "<text>Alpha</text>"
        "<text>Beta</text>"
        "</inline>"
        "<text><loc_50><loc_300><loc_400><loc_340>Outside</text>"
        "</doctag>";
    docv1::Document doc;
    std::string error;
    require(vlm::mapping::map_doctags(text, page_context(), &doc, &error),
            "inline group maps: " + error);
    require(doc.groups_size() == 1, "one inline group");
    const docv1::GroupItem& group = doc.groups(0);
    require(group.label() == docv1::GROUP_LABEL_INLINE, "inline label");
    require(group.name() == "group", "docling's InlineGroup default name");
    require(group.self_ref() == "#/groups/0" && group.parent().ref() == "#/body",
            "group refs");
    require(group.children_size() == 2 && group.children(0).ref() == "#/texts/0" &&
                group.children(1).ref() == "#/texts/1",
            "inline children in source order");

    require(doc.texts_size() == 3, "two inline children + the outside text");
    // Every inline child shares the chunk's one box (docling's common
    // bbox), not a per-item one.
    for (int i = 0; i < 2; i++) {
        const docv1::TextItemBase& base = doc.texts(i).text().base();
        require(base.parent().ref() == "#/groups/0", "child parented to the inline group");
        const docv1::BoundingBox& box = base.prov(0).bbox();
        require(box.l() == 100 && box.t() == 200 && box.r() == 800 && box.b() == 400,
                "children share the inline chunk's box");
    }
    require(doc.texts(0).text().base().text() == "Alpha" &&
                doc.texts(1).text().base().text() == "Beta",
            "child texts");
    require(doc.body().children_size() == 2 && doc.body().children(0).ref() == "#/groups/0" &&
                doc.body().children(1).ref() == "#/texts/2",
            "body children follow token order");
}

}  // namespace

int main() {
    try {
        verify_generation_source_on_every_item();
        verify_heading_paragraph_boxes();
        verify_table_picture_furniture();
        verify_unclosed_and_unknown_tags();
        verify_mapping_failures();
        verify_embedded_otsl_spans();
        verify_code_language();
        verify_charspan();
        verify_picture_classification();
        verify_picture_description();
        verify_chart();
        verify_source_order();
        verify_table_caption();
        verify_picture_crop();
        verify_key_value_region();
        verify_list_groups();
        verify_inline_group();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("doctags-mapper-test passed");
    return 0;
}
