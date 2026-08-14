cmake_minimum_required(VERSION 3.20)
project(grpc-vlm-convert VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# gRPC's CMake flips BUILD_SHARED_LIBS-adjacent defaults between releases;
# pin static explicitly so a green build cannot produce an unbootable image.
set(BUILD_SHARED_LIBS OFF)

option(GRPC_VLM_WERROR "Treat warnings as errors for first-party targets" OFF)
set(GRPC_VLM_SANITIZE "" CACHE STRING "Comma list for -fsanitize on first-party targets")

# Warnings apply to first-party targets only; fetched dependencies use -I
# (not -isystem) includes and would poison a global -Werror.
add_library(vlm_flags INTERFACE)
target_compile_options(vlm_flags INTERFACE -Wall -Wextra -Wnon-virtual-dtor)
if(GRPC_VLM_WERROR)
  target_compile_options(vlm_flags INTERFACE -Werror -Wno-error=deprecated-declarations)
endif()
if(GRPC_VLM_SANITIZE)
  target_compile_options(vlm_flags INTERFACE -fsanitize=${GRPC_VLM_SANITIZE} -fno-omit-frame-pointer -g)
  target_link_options(vlm_flags INTERFACE -fsanitize=${GRPC_VLM_SANITIZE})
endif()

find_package(Threads REQUIRED)

include(FetchContent)

set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG v1.83.0
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE)
FetchContent_MakeAvailable(grpc)

# The VLM endpoint is an OpenAI-compatible HTTP server; cpp-httplib is the
# client here and doubles as the fake VLM server in tests. Plaintext HTTP
# only — VLM servers sit on the pod network, no TLS in v1.
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.20.0
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(httplib)

# JSON is the VLM endpoint's dialect (request build, response parse). It
# never crosses the gRPC wire — typed protos only there.
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# Generated proto code. protoc and grpc_cpp_plugin are the fetched targets,
# so no host protoc is required (buf handles lint only).
# ---------------------------------------------------------------------------
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(VLM_PROTOS
  "${PROTO_DIR}/ai/pipestream/document/v1/document.proto"
  "${PROTO_DIR}/ai/pipestream/vlm/v1/vlm_convert.proto")
set(VLM_GENERATED
  "${GENERATED_DIR}/ai/pipestream/document/v1/document.pb.cc"
  "${GENERATED_DIR}/ai/pipestream/document/v1/document.pb.h"
  "${GENERATED_DIR}/ai/pipestream/vlm/v1/vlm_convert.pb.cc"
  "${GENERATED_DIR}/ai/pipestream/vlm/v1/vlm_convert.pb.h"
  "${GENERATED_DIR}/ai/pipestream/vlm/v1/vlm_convert.grpc.pb.cc"
  "${GENERATED_DIR}/ai/pipestream/vlm/v1/vlm_convert.grpc.pb.h")

add_custom_command(
  OUTPUT ${VLM_GENERATED}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
  COMMAND $<TARGET_FILE:protoc>
  ARGS --proto_path=${PROTO_DIR}
       --proto_path=${protobuf_SOURCE_DIR}/src
       --cpp_out=${GENERATED_DIR}
       --grpc_out=${GENERATED_DIR}
       --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc_cpp_plugin>
       ${VLM_PROTOS}
  DEPENDS ${VLM_PROTOS} protoc grpc_cpp_plugin
  VERBATIM)

add_library(vlm_proto STATIC
  "${GENERATED_DIR}/ai/pipestream/document/v1/document.pb.cc"
  "${GENERATED_DIR}/ai/pipestream/vlm/v1/vlm_convert.pb.cc"
  "${GENERATED_DIR}/ai/pipestream/vlm/v1/vlm_convert.grpc.pb.cc")
target_include_directories(vlm_proto PUBLIC "${GENERATED_DIR}")
target_link_libraries(vlm_proto PUBLIC grpc++)

# ---------------------------------------------------------------------------
# First-party code
# ---------------------------------------------------------------------------
add_library(vlm_core STATIC
  src/config.cpp
  src/presets.cpp
  src/vlm_client.cpp
  src/mapping/mapper.cpp
  src/mapping/doctags.cpp
  src/mapping/markdown.cpp
  src/mapping/html.cpp
  src/mapping/otsl.cpp
  src/service/vlm_convert_service.cpp)
target_include_directories(vlm_core PUBLIC src)
target_compile_definitions(vlm_core PRIVATE GRPC_VLM_VERSION="${PROJECT_VERSION}")
target_link_libraries(vlm_core
  PUBLIC vlm_proto httplib::httplib nlohmann_json::nlohmann_json Threads::Threads
  PRIVATE vlm_flags)

add_executable(grpc-vlm-convert-server src/main.cpp)
target_link_libraries(grpc-vlm-convert-server PRIVATE vlm_core vlm_flags grpc++_reflection)

install(TARGETS grpc-vlm-convert-server RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Tests: plain int main() executables under CTest, fleet style. The golden
# test needs a live VLM endpoint and skips with 77 when none is configured,
# so the suite stays green on a bare checkout.
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  set(GRPC_VLM_TESTS
    doctags-mapper-test
    markdown-mapper-test
    vlm-convert-service-test
    vlm-golden-test)
  foreach(test_name IN LISTS GRPC_VLM_TESTS)
    string(REPLACE "-" "_" test_source "${test_name}")
    add_executable(${test_name} "tests/${test_source}.cpp")
    target_link_libraries(${test_name} PRIVATE vlm_core vlm_flags)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  add_custom_target(grpc-vlm-convert-tests DEPENDS ${GRPC_VLM_TESTS})
  set_tests_properties(${GRPC_VLM_TESTS} PROPERTIES TIMEOUT 300 LABELS vlm)
  set_tests_properties(vlm-golden-test PROPERTIES SKIP_RETURN_CODE 77)
endif()
