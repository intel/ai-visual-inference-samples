#include "vaapi_frame_converter.hpp"

#include <unistd.h>
#include <cstring>

extern "C" {
#include <drm/drm_fourcc.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
}

void VaApiFrameConverter::init_frame_converter() {
    VaApiFramePool::FrameInfo frame_info;
    frame_info.width = output_resolution_.width;
    frame_info.height = output_resolution_.height;
    frame_info.format = output_color_format_;
    frame_pool = std::make_unique<VaApiFramePool>(context_, pool_size_, frame_info);
    initalized_ = true;
}

VaApiFrameConverter::VaApiFrameConverter(VaApiContextPtr context) : context_(std::move(context)) {
}

VaApiFrameConverter::VaApiFrameConverter(VaDpyWrapper va_display, uint32_t out_width,
                                         uint32_t out_height, uint32_t out_color_format)
    : output_resolution_{out_width, out_height}, output_color_format_(out_color_format),
      context_(std::make_shared<VaApiContext>(std::move(va_display))) {
}

void VaApiFrameConverter::set_output_color_format(uint32_t format) {
    output_color_format_ = format;
}
void VaApiFrameConverter::set_ouput_resolution(int width, int height) {
    output_resolution_.width = width;
    output_resolution_.height = height;
}

void VaApiFrameConverter::set_pool_size(uint32_t pool_size) {
    pool_size_ = pool_size;
}

void VaApiFrameConverter::release_frame(VaApiFrame* frame) {
    frame_pool->release(frame);
}

VaApiFrame* VaApiFrameConverter::convert(const VaApiFrame& src_frame) {
    // Full frame
    VARectangle src_region = {.x = static_cast<int16_t>(0),
                              .y = static_cast<int16_t>(0),
                              .width = src_frame.desc.width,
                              .height = src_frame.desc.height};

    return convert(src_frame, src_region);
}

VaApiFrame* VaApiFrameConverter::convert(const VaApiFrame& src_frame, VARectangle src_region) {
    if (!initalized_)
        init_frame_converter();
    auto dst_frame = frame_pool->acquire();
    convert_internal(src_frame, dst_frame, src_region);
    return dst_frame;
}

VaApiFrameConverter::ConvertResult VaApiFrameConverter::convert_ex(const VaApiFrame& src_frame,
                                                                   VARectangle src_region,
                                                                   bool allow_dyn_allocation) {
    if (!initalized_)
        init_frame_converter();

    std::unique_ptr<VaApiFrame> res_frame;

    VaApiFrame* pool_frame = get_frame_from_pool(allow_dyn_allocation);
    if (pool_frame) {
        // Create wrapped smart pointer that will return frame into pool
        res_frame = std::make_unique<VaApiFrameWrap>(
            pool_frame,
            [converter = this->shared_from_this()](VaApiFrame* f) { converter->release_frame(f); });
    } else {
        // TODO: context should stay alive
        res_frame =
            std::make_unique<VaApiFrame>(context_->display_native(), output_resolution_.width,
                                         output_resolution_.height, output_color_format_);
    }
    convert_internal(src_frame, res_frame.get(), src_region);

    return {std::move(res_frame), pool_frame == nullptr};
}

VaApiFrame* VaApiFrameConverter::get_frame_from_pool(bool nowait) {
    if (nowait)
        return frame_pool->acquire_nowait();
    return frame_pool->acquire();
}

void VaApiFrameConverter::convert_internal(const VaApiFrame& src_frame, VaApiFrame* dst_frame,
                                           VARectangle src_region) {
    if (src_frame.desc.va_display != dst_frame->desc.va_display)
        throw std::runtime_error("Cannot convert frames from different VADisplays.");
    uint64_t fd = 0;
    VASurfaceID src_surface = src_frame.desc.va_surface_id;

    VAProcPipelineParameterBuffer pipeline_param;
    std::memset(&pipeline_param, 0, sizeof(pipeline_param));

    pipeline_param.surface = src_surface;

    // Scale and csc mode
    pipeline_param.filter_flags = VA_FILTER_SCALING_FAST;

    // Crop ROI
    if (src_region.width > 0 && src_region.height > 0)
        pipeline_param.surface_region = &src_region;

    // Resize to this Rect
    VARectangle dst_surface_region = {.x = static_cast<int16_t>(0),
                                      .y = static_cast<int16_t>(0),
                                      .width = dst_frame->desc.width,
                                      .height = dst_frame->desc.height};
    pipeline_param.output_region = &dst_surface_region;

    VABufferID pipeline_param_buf_id = VA_INVALID_ID;

    auto status = vaCreateBuffer(context_->display_native(), context_->id_native(),
                                 VAProcPipelineParameterBufferType, sizeof(pipeline_param), 1,
                                 &pipeline_param, &pipeline_param_buf_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateBuffer failed, " + std::to_string(status));

    {
        // These operations can't be called asynchronously from different threads
        status = vaBeginPicture(context_->display_native(), context_->id_native(),
                                dst_frame->desc.va_surface_id);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaBeginPicture failed, " + std::to_string(status));

        status = vaRenderPicture(context_->display_native(), context_->id_native(),
                                 &pipeline_param_buf_id, 1);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaRenderPicture failed, " + std::to_string(status));

        status = vaEndPicture(context_->display_native(), context_->id_native());
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaEndPicture failed, " + std::to_string(status));
    }

    status = vaDestroyBuffer(context_->display_native(), pipeline_param_buf_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaDestroyBuffer failed, " + std::to_string(status));
}
