#pragma once

#include <functional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <memory>
#include "unistd.h"
#include <fcntl.h>
#include <iostream>

extern "C" {
#include <va/va.h>
#include <va/va_backend.h>
#include <va/va_drm.h>
}

/**
 * Wrapper around VaDisplay.
 * Needed for more convenient usage of VADisplay and its fields.
 */
class VaDpyWrapper final {
  private:
    struct Storage {
        int drm_fd = -1;
        VADisplay display = nullptr;
        Storage(std::string_view device);
        ~Storage();

        // No COPY
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
    };

    std::shared_ptr<Storage> ptr_;

  public:
    explicit VaDpyWrapper() noexcept = default;
    explicit VaDpyWrapper(std::string_view device) : ptr_(std::make_shared<Storage>(device)) {}

    // Default copy
    VaDpyWrapper(const VaDpyWrapper&) = default;
    VaDpyWrapper& operator=(const VaDpyWrapper&) = default;

    // Default move
    VaDpyWrapper(VaDpyWrapper&&) = default;
    VaDpyWrapper& operator=(VaDpyWrapper&&) = default;

    // Returns native handle
    VADisplay native() const noexcept {
        if (ptr_)
            return ptr_->display;
        return nullptr;
    }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }
};

class VaApiContext final {
  public:
    explicit VaApiContext(VaDpyWrapper va_display_wrapper);

    ~VaApiContext();

    // No Copy
    VaApiContext(const VaApiContext&) = delete;
    VaApiContext& operator=(const VaApiContext&) = delete;

    /* getters */
    VADisplay display_native() const;
    const VaDpyWrapper& display() const;
    VAContextID id_native() const;
    bool is_pixel_format_supported(int format) const;

  private:
    VaDpyWrapper _display;
    VAConfigID _va_config_id = VA_INVALID_ID;
    VAContextID _va_context_id = VA_INVALID_ID;
    int _dri_file_descriptor = 0;
    int _rt_format = VA_RT_FORMAT_YUV420;
    std::set<int> _supported_pixel_formats;

    /* private helper methods */
    void create_config_and_contexts();
    void create_supported_pixel_formats();
};

using VaApiContextPtr = std::shared_ptr<VaApiContext>;
