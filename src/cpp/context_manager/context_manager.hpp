#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <cstdint>
#include "l0_context.hpp"
#include "vaapi_context.hpp"

class ContextManager {
  public:
    static ContextManager& get_instance();

    std::shared_ptr<L0Context> get_l0_context(const uint32_t device);
    VaDpyWrapper get_va_display(const std::string& device);

    static std::string get_device_path_from_device_name(const std::string& device_name);
    static uint32_t get_ze_device_id(const std::string& device_name);

    static VaDpyWrapper get_vaapi_diplay_by_device_name(const std::string& device_name) {
        const auto dev_path = get_device_path_from_device_name(device_name);
        return get_instance().get_va_display(dev_path);
    }

    static std::shared_ptr<L0Context> get_l0_ctx_by_device_name(const std::string& device_name) {
        const auto l0_dev_id = get_ze_device_id(device_name);
        return get_instance().get_l0_context(l0_dev_id);
    }

  private:
    ContextManager() = default;
    std::unordered_map<uint32_t, std::shared_ptr<L0Context>> l0_contexts_;
    std::unordered_map<std::string, VaDpyWrapper> vaapi_display_wrappers_;

    // Delete copy constructor and assignment operator
    ContextManager(const ContextManager&) = delete;
    ContextManager& operator=(const ContextManager&) = delete;
};
