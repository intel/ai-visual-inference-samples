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
    std::shared_ptr<VaApiContext> get_va_context(const std::string& device);

    static std::string get_device_path_from_device_name(const std::string& device_name);
    static uint32_t get_ze_device_id(const std::string& device_name);

  private:
    ContextManager() = default;
    std::unordered_map<uint32_t, std::shared_ptr<L0Context>> l0_contexts_;
    std::unordered_map<std::string, std::shared_ptr<VaApiContext>> vaapi_contexts_;

    // Delete copy constructor and assignment operator
    ContextManager(const ContextManager&) = delete;
    ContextManager& operator=(const ContextManager&) = delete;
};
