# syntax=docker/dockerfile:1.26
# grpc-vlm-convert — CPU-only image. The VLM itself is a separate server;
# this process is a mapper + HTTP client, so the runtime needs no GPU and
# no model mount.
#
# The build stage compiles the server and runs the test suite; the tests
# gate the image. The golden test needs a live VLM endpoint and skips
# cleanly (exit 77) in the image build.

FROM ubuntu:26.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates cmake g++ git make ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# The cache id encodes every ABI-sensitive dependency; bump it when gRPC,
# cpp-httplib, nlohmann/json, the C++ standard, or the toolchain moves.
# TARGETARCH keys the cache per platform: multi-arch legs otherwise share
# one mount id and poison each other's object files.
ARG TARGETARCH
RUN --mount=type=cache,id=grpc-vlm-convert-ubuntu26-cxx23-grpc1.83.0-httplib0.53.1-${TARGETARCH},target=/build \
    cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
        -DGRPC_VLM_WERROR=ON \
    && cmake --build /build --target grpc-vlm-convert-server grpc-vlm-convert-tests --parallel \
    && ctest --test-dir /build -L vlm --output-on-failure \
    && mkdir -p /out && cp /build/grpc-vlm-convert-server /out/

FROM ubuntu:26.04

COPY --from=build /out/grpc-vlm-convert-server /usr/local/bin/grpc-vlm-convert-server

ENV GRPC_VLM_LISTEN_ADDRESS=0.0.0.0:50058

# Diskless contract: pages live in memory, nothing is written. Run with
# --read-only; the VLM endpoint is another container:
#   docker run --rm --read-only -e GRPC_VLM_ENDPOINT=http://vlm:8080 \
#     -p 50058:50058 grpc-vlm-convert
# Only gRPC is exposed: the HTTP front end (50059) binds loopback inside
# the container by default, so nothing outside it can reach that port. To
# serve HTTP, set GRPC_VLM_HTTP_HOST=0.0.0.0 with GRPC_VLM_HTTP_TOKEN (the
# server refuses to start without one) and publish it:
#   docker run --rm --read-only -e GRPC_VLM_ENDPOINT=http://vlm:8080 \
#     -e GRPC_VLM_HTTP_HOST=0.0.0.0 -e GRPC_VLM_HTTP_TOKEN=... \
#     -p 50058:50058 -p 50059:50059 grpc-vlm-convert
USER 65532:65532
EXPOSE 50058
ENTRYPOINT ["/usr/local/bin/grpc-vlm-convert-server"]
