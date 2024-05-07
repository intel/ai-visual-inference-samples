#pragma once

#include "vaapi_context.hpp"
#include "vaapi_frame.hpp"
#include <future>
#include <vector>

class VaApiFramePool {
    std::vector<std::shared_ptr<VaApiFrame>> frames_;
    std::condition_variable free_frame_cond_variable_;
    std::mutex mutex_;
    VaApiContextPtr context_;

  public:
    struct FrameInfo {
        uint32_t width;
        uint32_t height;
        // uint32_t batch;
        uint32_t format;
    };
    VaApiFramePool(VaApiContextPtr context, uint32_t pool_size, FrameInfo info);

    // Acquire frame from pool. If there is no free frame, it waits for a free frame.
    VaApiFrame* acquire();
    // Acquire frame from pool. If there is no free frame, it returns nullptr immedeatly.
    VaApiFrame* acquire_nowait();
    void release(VaApiFrame* frame);
    void flush();

  private:
    VaApiFrame* acquire_internal_locked();
};