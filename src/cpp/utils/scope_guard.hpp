#pragma once

#include <utility>

template <class Fn>
class ScopeGuard {
  public:
    ScopeGuard(Fn&& func) : func_(std::forward<Fn>(func)) {}
    ~ScopeGuard() {
        if (armed_)
            func_();
    }

    ScopeGuard(ScopeGuard&& other) : func_(std::move(other.func_)), armed_(other.armed_) {
        other.armed_ = false;
    }

    void disable() { armed_ = false; }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard& operator=(ScopeGuard&&) = delete;

  private:
    Fn func_;
    bool armed_ = true;
};

template <class Fn>
ScopeGuard<Fn> make_scope_guard(Fn&& rollback_fn) {
    return ScopeGuard<Fn>(std::forward<Fn>(rollback_fn));
}
