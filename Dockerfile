# syntax=docker/dockerfile:1.7
# grpc-vlm-convert — CPU-only image. The VLM itself is a separate server;
# this process is a mapper + HTTP client, so the runtime needs no GPU and
# no model mount.
#
# The build stage compiles the server and runs the test suite; the tests
# gate the image. The golden test needs a live VLM endpoint and skips
# cleanly (exit 77) in the image build.

FROM ubuntu:24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates cmake g++ git make ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# The cache id encodes every ABI-sensitive dependency; bump it when gRPC,
# cpp-httplib, nlohmann/json, or the toolchain moves.
RUN --mount=type=cache,id=grpc-vlm-convert-ubuntu24-grpc1.83.0-httplib0.20.0,target=/build \
    cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
        -DGRPC_VLM_WERROR=ON \
    && cmake --build /build --target grpc-vlm-convert-server grpc-vlm-convert-tests --parallel \
    && ctest --test-dir /build -L vlm --output-on-failure \
    && mkdir -p /out && cp /build/grpc-vlm-convert-server /out/

FROM ubuntu:24.04

COPY --from=build /out/grpc-vlm-convert-server /usr/local/bin/grpc-vlm-convert-server

ENV GRPC_VLM_LISTEN_ADDRESS=0.0.0.0:50058

# Diskless contract: pages live in memory, nothing is written. Run with
# --read-only; the VLM endpoint is another container:
#   docker run --rm --read-only -e GRPC_VLM_ENDPOINT=http://vlm:8080 \
#     -p 50058:50058 grpc-vlm-convert
USER 65532:65532
EXPOSE 50058 50059
ENTRYPOINT ["/usr/local/bin/grpc-vlm-convert-server"]
