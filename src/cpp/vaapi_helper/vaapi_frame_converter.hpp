#pragma once

#include "vaapi_context.hpp"
#include "vaapi_frame_pool.hpp"

class VaApiFrameConverter {
    VaApiContextPtr context_;
    std::unique_ptr<VaApiFramePool> frame_pool;
    bool initalized_ = false;
    struct {
        uint32_t width = 0;
        uint32_t height = 0;
    } output_resolution_;
    uint32_t output_color_format_ = 0;
    uint32_t pool_size_ = 5;
    void init_frame_converter();

  public:
    VaApiFrameConverter(VaApiContextPtr context);
    VaApiFrameConverter(VADisplay va_display, uint32_t out_width, uint32_t out_height,
                        uint32_t out_color_format);
    VaApiFrame* convert(const VaApiFrame& src_frame);
    void release_frame(VaApiFrame* frame); // TODO can we do it better?
    void set_output_color_format(uint32_t format);
    void set_ouput_resolution(int width, int height);
    void set_pool_size(uint32_t pool_size);
};