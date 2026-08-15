#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "config.h"
#include "http_gateway.h"
#include "service/vlm_convert_service.h"

namespace {

// The handler only writes one byte to a pipe: grpc::Server::Shutdown
// allocates and takes locks, none of which is async-signal-safe. A
// dedicated thread does the real shutdown.
int g_shutdown_pipe[2] = {-1, -1};

void handle_shutdown(int /*signum*/) {
    char one = 1;
    [[maybe_unused]] ssize_t n = ::write(g_shutdown_pipe[1], &one, 1);
}

void install_shutdown_pipe() {
    if (::pipe2(g_shutdown_pipe, O_CLOEXEC) != 0) {
        throw std::runtime_error("shutdown pipe creation failed");
    }
    struct sigaction action = {};
    action.sa_handler = handle_shutdown;
    ::sigaction(SIGINT, &action, nullptr);
    ::sigaction(SIGTERM, &action, nullptr);
    ::signal(SIGPIPE, SIG_IGN);
}

}  // namespace

int main() {
    try {
        install_shutdown_pipe();

        const vlm::Config config = vlm::load_config_from_env();
        vlm::VlmConvertServiceImpl service(config);

        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        grpc::ServerBuilder builder;
        builder.AddListeningPort(config.listen_address, grpc::InsecureServerCredentials());
        builder.SetMaxReceiveMessageSize(64 * 1024 * 1024);
        builder.RegisterService(&service);
        std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
        if (server == nullptr) {
            std::cerr << "Unable to listen on " << config.listen_address << '\n';
            return 1;
        }

        // The HTTP/JSON front end shares the service instance (and its
        // metrics) with gRPC; GRPC_VLM_HTTP_PORT 0/empty leaves it off.
        vlm::HttpGateway http_gateway(config, service);
        if (config.http_port != 0 &&
            !http_gateway.start("0.0.0.0", static_cast<int>(config.http_port))) {
            std::cerr << "Unable to listen on 0.0.0.0:" << config.http_port << " (HTTP)\n";
            return 1;
        }

        std::thread shutdown_thread([&server] {
            char byte = 0;
            [[maybe_unused]] ssize_t n = ::read(g_shutdown_pipe[0], &byte, 1);
            server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(10));
        });

        // Interval metrics line; a condition variable instead of a sleep
        // loop so shutdown never waits out the interval.
        std::mutex metrics_mutex;
        std::condition_variable metrics_stop;
        bool stopping = false;
        std::thread metrics_thread;
        if (config.metrics_interval_seconds > 0) {
            metrics_thread = std::thread([&] {
                std::unique_lock<std::mutex> lock(metrics_mutex);
                while (!metrics_stop.wait_for(
                    lock, std::chrono::seconds(config.metrics_interval_seconds),
                    [&] { return stopping; })) {
                    std::cout << "grpc-vlm-convert metrics: streams{converted="
                              << service.converted.load() << ",rejected=" << service.rejected.load()
                              << ",failed=" << service.failed.load() << "} pages{ok="
                              << service.pages_ok.load() << ",failed=" << service.pages_failed.load()
                              << "}" << std::endl;
                }
            });
        }

        std::cout << "grpc-vlm-convert listening on " << config.listen_address
                  << (config.http_port != 0
                          ? " (HTTP on 0.0.0.0:" + std::to_string(config.http_port) + ")"
                          : "")
                  << " (endpoint "
                  << (config.endpoint.empty() ? "<none — per-request override required>"
                                              : config.endpoint)
                  << ")" << std::endl;
        server->Wait();

        http_gateway.stop();

        // Wake the shutdown thread if Wait() returned for another reason.
        handle_shutdown(0);
        shutdown_thread.join();
        {
            std::lock_guard<std::mutex> lock(metrics_mutex);
            stopping = true;
        }
        metrics_stop.notify_all();
        if (metrics_thread.joinable()) {
            metrics_thread.join();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Startup failed: " << error.what() << '\n';
        return 1;
    }
}
