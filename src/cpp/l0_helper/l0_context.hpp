#pragma once

#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <typeinfo>
#include <level_zero/ze_api.h>

class L0Context final {
  public:
    L0Context(uint32_t ze_device_id);

    L0Context(const L0Context&) = delete;
    L0Context& operator=(const L0Context&) = delete;

    ze_context_handle_t get_ze_context() const;
    ze_device_handle_t get_ze_device() const;

  private:
    ze_driver_handle_t ze_driver_ = nullptr;
    ze_context_handle_t ze_context_ = nullptr;
    ze_device_handle_t ze_device_ = nullptr;

    uint32_t ze_device_id_ = 0;
};

using L0ContextPtr = std::shared_ptr<L0Context>;