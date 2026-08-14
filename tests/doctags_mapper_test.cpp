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

}  // namespace

int main() {
    try {
        verify_heading_paragraph_boxes();
        verify_table_picture_furniture();
        verify_unclosed_and_unknown_tags();
        verify_mapping_failures();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "doctags-mapper-test passed\n";
    return 0;
}
