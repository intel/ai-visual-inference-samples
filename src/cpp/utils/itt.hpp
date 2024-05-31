#pragma once
#include <unordered_map>
#include <iostream>

#ifdef ENABLE_PROFILING_ITT
#include <ittnotify.h>

class IttTask {
    __itt_domain* domain_;
    static __itt_domain* get_domain() {
        static __itt_domain* domain = __itt_domain_create("visual_ai");
        return domain;
    }

  public:
    IttTask() noexcept : domain_(get_domain()) {}
    IttTask(std::string_view name) noexcept : domain_(__itt_domain_create(name.data())) {}
    void start(std::string_view name) noexcept {
        __itt_task_begin(domain_, __itt_null, __itt_null, __itt_string_handle_create(name.data()));
    }
    void end() noexcept { __itt_task_end(domain_); }
};

#else

class IttTask {
  public:
    IttTask() noexcept {}
    IttTask(std::string_view /*name*/) noexcept {}
    void start(std::string_view /*name*/) noexcept {}
    void end() noexcept {}
};

#endif // ENABLE_PROFILING_ITT
