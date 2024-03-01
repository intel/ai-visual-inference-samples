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

    VaApiFrame* acquire();
    void release(VaApiFrame* frame);
    void flush();
};