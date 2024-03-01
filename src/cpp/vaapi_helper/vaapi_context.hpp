#pragma once

#include <functional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <memory>

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
  public:
    explicit VaDpyWrapper() = default;
    explicit VaDpyWrapper(VADisplay d) : _dpy(d) {
        if (!isDisplayValid(_dpy))
            throw std::invalid_argument("VADisplay is invalid.");
    }

    static VaDpyWrapper fromHandle(VADisplay d) { return VaDpyWrapper(d); }

    static bool isDisplayValid(VADisplay d) noexcept {
        auto pDisplayContext = reinterpret_cast<VADisplayContextP>(d);
        return d && pDisplayContext && (pDisplayContext->vadpy_magic == VA_DISPLAY_MAGIC) &&
               pDisplayContext->pDriverContext;
    }

    VADisplay raw() const noexcept { return _dpy; }

    explicit operator bool() const noexcept { return isDisplayValid(_dpy); }

    VADisplayContextP dpyCtx() const noexcept { return reinterpret_cast<VADisplayContextP>(_dpy); }

    VADriverContextP drvCtx() const noexcept { return dpyCtx()->pDriverContext; }

    const VADriverVTable& drvVtable() const noexcept { return *drvCtx()->vtable; }

  private:
    VADisplay _dpy = nullptr;
};

class VaApiContext final {
  public:
    explicit VaApiContext(VADisplay va_display);
    explicit VaApiContext(std::string_view device);

    ~VaApiContext();

    // No Copy
    VaApiContext(const VaApiContext&) = delete;
    VaApiContext& operator=(const VaApiContext&) = delete;

    /* getters */
    VADisplay DisplayRaw() const;
    VaDpyWrapper Display() const;
    VAContextID Id() const;
    int RTFormat() const;
    bool IsPixelFormatSupported(int format) const;

  private:
    int drm_fd_ = -1;
    VaDpyWrapper _display;
    VAConfigID _va_config_id = VA_INVALID_ID;
    VAContextID _va_context_id = VA_INVALID_ID;
    int _dri_file_descriptor = 0;
    int _rt_format = VA_RT_FORMAT_YUV420;
    std::set<int> _supported_pixel_formats;

    /* private helper methods */
    void create_config_and_contexts();
    void create_supported_pixel_formats();
    void init_vaapi_objects(std::string_view device);
};

using VaApiContextPtr = std::shared_ptr<VaApiContext>;
