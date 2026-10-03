#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

namespace vlm {

// A credential read from the environment: the VLM API key, the HTTP front
// end's token. It has no formatter, no stream operator and no implicit
// conversion, so it cannot reach a log line or an error message by
// accident; reveal() is the one way out, at the line that sends it.
class Secret {
  public:
    Secret() = default;
    explicit Secret(std::string value) : value_(std::move(value)) {}

    bool empty() const { return value_.empty(); }
    const std::string& reveal() const { return value_; }

    // True when `candidate` is this secret. Every byte of the candidate is
    // compared even after a mismatch, so the time a check takes does not
    // say how much of a guess was right. An empty secret matches nothing.
    bool matches(std::string_view candidate) const {
        if (value_.empty()) {
            return false;
        }
        unsigned char difference = candidate.size() == value_.size() ? 0 : 1;
        for (size_t i = 0; i < candidate.size(); i++) {
            const auto expected =
                i < value_.size() ? static_cast<unsigned char>(value_[i]) : '\0';
            difference |= static_cast<unsigned char>(expected ^
                                                     static_cast<unsigned char>(candidate[i]));
        }
        return difference == 0;
    }

  private:
    std::string value_;
};

}  // namespace vlm
