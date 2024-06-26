#include "vaapi_frame_pool.hpp"

VaApiFramePool::VaApiFramePool(VaApiContextPtr context, uint32_t pool_size, FrameInfo info)
    : context_(std::move(context)) {
    if (!context_)
        throw std::invalid_argument("VaApiContext is nullptr");
    if (pool_size == 0)
        throw std::invalid_argument("pool_size can't be zero");

    if (info.height == 0 || info.width == 0) {
        throw std::invalid_argument(
            "Frame dimensions can`t be zero. Width: " + std::to_string(info.width) +
            " Height: " + std::to_string(info.width));
    }

    if (!context_->is_pixel_format_supported(info.format)) {
        throw std::invalid_argument("Unsupported requested pixel format " +
                                    std::to_string(info.format));
    }

    frames_.reserve(pool_size);
    for (size_t i = 0; i < pool_size; i++) {
        frames_.push_back(std::make_shared<VaApiFrame>(context_->display_native(), info.width,
                                                       info.height, info.format));
    }
}

VaApiFrame* VaApiFramePool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        auto frame = acquire_internal_locked();
        if (frame)
            return frame;
        free_frame_cond_variable_.wait(lock);
    }
}

VaApiFrame* VaApiFramePool::acquire_nowait() {
    std::unique_lock<std::mutex> lock(mutex_);
    return acquire_internal_locked();
}

void VaApiFramePool::release(VaApiFrame* frame) {
    frame->completed = true;
    free_frame_cond_variable_.notify_one();
}

void VaApiFramePool::flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto& frame : frames_) {
        if (!frame->completed)
            frame->completed_sync.wait();
    }
}

VaApiFrame* VaApiFramePool::acquire_internal_locked() {
    for (auto& frame : frames_) {
        if (frame->completed) {
            frame->completed = false;
            return frame.get();
        }
    }
    return nullptr;
}
