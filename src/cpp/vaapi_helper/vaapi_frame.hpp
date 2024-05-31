#pragma once

#include <future>
#include <assert.h>

#include "visual_ai/frame.hpp"
#include "visual_ai/system_frame.hpp"

struct FrameDescription {
    uint32_t format = 0; // FourCC
    uint16_t width = 0;
    uint16_t height = 0;
    uint32_t size = 0;

    // TODO: move
    uint32_t va_surface_id = 0xffffffff;
    void* va_display = nullptr;
};

struct VaApiFrame : public Frame {
    FrameDescription desc;
    bool completed = true;
    std::future<void> completed_sync; // TODO: consider removing
    bool owns_surface = false;

    VaApiFrame() = default;
    VaApiFrame(void* va_display, uint32_t va_surface_id, uint32_t width, uint32_t height,
               int format, bool owns = false);
    VaApiFrame(void* va_display, uint32_t va_surface_id, bool owns = false);
    VaApiFrame(void* va_display, uint32_t width, uint32_t height, int format);
    ~VaApiFrame();

    // Implement deep copy of VaSurface only
    static std::unique_ptr<VaApiFrame> copy_from(const VaApiFrame& other);
    std::unique_ptr<SystemFrame> copy_to_system();

    // Checks if surface status is "ready"
    bool is_ready() const;

    // Performs surface synchronization
    void sync() const override;
};

// Thin wrapper around raw VaApiFrame pointer.
// Enables conversion to VaApiFrame and custom deletion
//
// Main purpose is wrapping raw pointers from pool to enable custom deletion
// TODO: Integrate into frame pool.
struct VaApiFrameWrap final : public VaApiFrame {
    using FrameDelFn = std::function<void(VaApiFrame*)>;

    VaApiFrame* frame = nullptr;
    FrameDelFn frame_deleter;

    VaApiFrameWrap(VaApiFrame* f, FrameDelFn del_fn)
        : VaApiFrame(f->desc.va_display, f->desc.va_surface_id, f->desc.width, f->desc.height,
                     f->desc.format, false),
          frame(f), frame_deleter(del_fn) {}

    ~VaApiFrameWrap() {
        if (frame) {
            assert(frame_deleter);
            frame_deleter(frame);
        }
    }
};
