#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <list>
#include <mutex>
#include <utility>

namespace vlm {

class Lease;

// A counting budget with first-come, first-served blocking acquisition:
// bytes of page images for the buffer caps, call slots for the in-flight
// cap. Waiters queue in arrival order, so a large page is not starved by a
// run of small ones, and a waiter whose stream stops leaves the line
// instead of parking an RPC thread nobody is waiting on.
class Budget {
  public:
    explicit Budget(size_t capacity) : capacity_(std::max<size_t>(capacity, 1)) {}
    Budget(const Budget&) = delete;
    Budget& operator=(const Budget&) = delete;

    // Blocks until `amount` fits, in arrival order, and takes it. An amount
    // above the whole budget is clamped to it (the request then waits for
    // the budget to drain and runs alone, rather than never), and zero
    // counts as one. `halted` is polled, since nothing signals it; when it
    // turns true first, the returned lease is empty.
    Lease acquire(size_t amount, const std::function<bool()>& halted);

    size_t capacity() const { return capacity_; }

    // What is taken right now (tests and diagnostics).
    size_t in_use() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return used_;
    }

  private:
    friend class Lease;

    void release(size_t amount) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            used_ -= amount;
        }
        changed_.notify_all();
    }

    static constexpr auto kPollInterval = std::chrono::milliseconds(50);

    const size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    size_t used_ = 0;
    // Waiters in arrival order; only the front may take.
    std::list<size_t> line_;
};

// One acquisition from a Budget, given back when the lease is destroyed or
// reset. Move-only; an empty lease holds nothing.
class Lease {
  public:
    Lease() = default;
    Lease(Lease&& other) noexcept
        : budget_(std::exchange(other.budget_, nullptr)), amount_(other.amount_) {}
    Lease& operator=(Lease&& other) noexcept {
        if (this != &other) {
            reset();
            budget_ = std::exchange(other.budget_, nullptr);
            amount_ = other.amount_;
        }
        return *this;
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() { reset(); }

    explicit operator bool() const { return budget_ != nullptr; }

    void reset() {
        if (budget_ != nullptr) {
            budget_->release(amount_);
            budget_ = nullptr;
        }
    }

  private:
    friend class Budget;
    Lease(Budget* budget, size_t amount) : budget_(budget), amount_(amount) {}

    Budget* budget_ = nullptr;
    size_t amount_ = 0;
};

inline Lease Budget::acquire(size_t amount, const std::function<bool()>& halted) {
    amount = std::clamp<size_t>(amount, 1, capacity_);
    std::unique_lock<std::mutex> lock(mutex_);
    const auto me = line_.insert(line_.end(), amount);
    for (;;) {
        if (line_.begin() == me && used_ + amount <= capacity_) {
            used_ += amount;
            line_.erase(me);
            lock.unlock();
            changed_.notify_all();  // the next in line may fit too
            return Lease(this, amount);
        }
        if (halted && halted()) {
            line_.erase(me);
            lock.unlock();
            changed_.notify_all();  // whoever was behind is now in front
            return Lease();
        }
        changed_.wait_for(lock, kPollInterval);
    }
}

}  // namespace vlm
