#pragma once

#include <future>

struct FrameDescription {
    uint32_t format = 0; // FourCC
    uint16_t width = 0;
    uint16_t height = 0;
    uint32_t size = 0;

    // TODO: move
    uint32_t va_surface_id = 0xffffffff;
    void* va_display = nullptr;
};

struct VaApiFrame {
    FrameDescription desc;
    bool completed = true;
    std::future<void> sync;
    bool owns_surface = false;

    VaApiFrame() = default;
    VaApiFrame(void* va_display, uint32_t va_surface_id, uint32_t width, uint32_t height,
               int format, bool owns = false);
    VaApiFrame(void* va_display, uint32_t va_surface_id, bool owns = false);
    VaApiFrame(void* va_display, uint32_t width, uint32_t height, int format);
    virtual ~VaApiFrame();

    // Implement deep copy of VaSurface only
    static std::unique_ptr<VaApiFrame> copy_from(const VaApiFrame& other);
    // No Copy
    VaApiFrame(const VaApiFrame&) = delete;
    VaApiFrame& operator=(const VaApiFrame&) = delete;
};
