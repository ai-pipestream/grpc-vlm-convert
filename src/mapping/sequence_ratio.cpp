#include "sequence_ratio.h"

#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vlm::mapping {
namespace {

// Code points of a UTF-8 string. An invalid or truncated sequence yields
// its lead byte as its own value above the Unicode range, so it still
// compares equal only to the same byte.
std::vector<char32_t> code_points(std::string_view text) {
    std::vector<char32_t> out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        size_t length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3
                                       : (lead >> 3) == 0x1E ? 4
                                                             : 0;
        bool valid = length != 0 && i + length <= text.size();
        char32_t value = length == 1 ? lead : lead & (0x7F >> length);
        for (size_t k = 1; valid && k < length; k++) {
            const auto next = static_cast<unsigned char>(text[i + k]);
            valid = (next & 0xC0) == 0x80;
            value = (value << 6) | (next & 0x3F);
        }
        if (!valid) {
            out.push_back(0x110000 + lead);
            i++;
            continue;
        }
        out.push_back(value);
        i += length;
    }
    return out;
}

struct Match {
    size_t a, b, size;
};

class Matcher {
  public:
    Matcher(std::vector<char32_t> a, std::vector<char32_t> b) : a_(std::move(a)), b_(std::move(b)) {
        for (size_t j = 0; j < b_.size(); j++) {
            b2j_[b_[j]].push_back(j);
        }
    }

    // find_longest_match with no junk: the longest a[i:i+k] == b[j:j+k]
    // in the ranges, earliest in a, then earliest in b.
    Match longest(size_t alo, size_t ahi, size_t blo, size_t bhi) const {
        Match best{alo, blo, 0};
        std::unordered_map<size_t, size_t> j2len;
        for (size_t i = alo; i < ahi; i++) {
            std::unordered_map<size_t, size_t> next;
            const auto found = b2j_.find(a_[i]);
            if (found != b2j_.end()) {
                for (const size_t j : found->second) {
                    if (j < blo) {
                        continue;
                    }
                    if (j >= bhi) {
                        break;
                    }
                    const auto previous = j > 0 ? j2len.find(j - 1) : j2len.end();
                    const size_t k = (previous == j2len.end() ? 0 : previous->second) + 1;
                    next[j] = k;
                    if (k > best.size) {
                        best = {i + 1 - k, j + 1 - k, k};
                    }
                }
            }
            j2len = std::move(next);
        }
        return best;
    }

    // Sum of get_matching_blocks sizes.
    size_t matches() const {
        size_t total = 0;
        std::vector<std::tuple<size_t, size_t, size_t, size_t>> queue{{0, a_.size(), 0, b_.size()}};
        while (!queue.empty()) {
            const auto [alo, ahi, blo, bhi] = queue.back();
            queue.pop_back();
            const Match m = longest(alo, ahi, blo, bhi);
            if (m.size == 0) {
                continue;
            }
            total += m.size;
            if (alo < m.a && blo < m.b) {
                queue.emplace_back(alo, m.a, blo, m.b);
            }
            if (m.a + m.size < ahi && m.b + m.size < bhi) {
                queue.emplace_back(m.a + m.size, ahi, m.b + m.size, bhi);
            }
        }
        return total;
    }

  private:
    std::vector<char32_t> a_;
    std::vector<char32_t> b_;
    std::unordered_map<char32_t, std::vector<size_t>> b2j_;
};

}  // namespace

double sequence_ratio(std::string_view a, std::string_view b) {
    std::vector<char32_t> left = code_points(a);
    std::vector<char32_t> right = code_points(b);
    const size_t length = left.size() + right.size();
    if (length == 0) {
        return 1.0;
    }
    const Matcher matcher(std::move(left), std::move(right));
    return 2.0 * static_cast<double>(matcher.matches()) / static_cast<double>(length);
}

}  // namespace vlm::mapping
