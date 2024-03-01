#include "vaapi_frame_converter.hpp"

#include <unistd.h>
#include <cstring>

extern "C" {
#include <drm/drm_fourcc.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
}

VASurfaceID ConvertVASurfaceFromDifferentDriverContext(VaDpyWrapper src_display,
                                                       VASurfaceID src_surface,
                                                       VaDpyWrapper dst_display, int rt_format,
                                                       uint64_t& drm_fd_out) {

    VADRMPRIMESurfaceDescriptor drm_descriptor = VADRMPRIMESurfaceDescriptor();
    auto status = src_display.drvVtable().vaSyncSurface(src_display.drvCtx(), src_surface);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaSyncSurface failed, " + std::to_string(status));

    status = src_display.drvVtable().vaExportSurfaceHandle(
        src_display.drvCtx(), src_surface, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
        VA_EXPORT_SURFACE_READ_ONLY, &drm_descriptor);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaExportSurfaceHandle failed, " + std::to_string(status));

    VASurfaceAttribExternalBuffers external = VASurfaceAttribExternalBuffers();
    external.width = drm_descriptor.width;
    external.height = drm_descriptor.height;
    external.pixel_format = drm_descriptor.fourcc;

    if (drm_descriptor.num_objects != 1)
        throw std::invalid_argument("Unexpected objects number");
    auto object = drm_descriptor.objects[0];
    external.num_buffers = 1;
    uint64_t drm_fd = object.fd;
    drm_fd_out = drm_fd;
    external.buffers = &drm_fd;
    external.data_size = object.size;
    external.flags = object.drm_format_modifier == DRM_FORMAT_MOD_LINEAR
                         ? 0
                         : VA_SURFACE_EXTBUF_DESC_ENABLE_TILING;

    uint32_t k = 0;
    for (uint32_t i = 0; i < drm_descriptor.num_layers; i++) {
        for (uint32_t j = 0; j < drm_descriptor.layers[i].num_planes; j++) {
            external.pitches[k] = drm_descriptor.layers[i].pitch[j];
            external.offsets[k] = drm_descriptor.layers[i].offset[j];
            ++k;
        }
    }
    external.num_planes = k;

    VASurfaceAttrib attribs[2] = {};
    attribs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
    attribs[0].type = VASurfaceAttribMemoryType;
    attribs[0].value.type = VAGenericValueTypeInteger;
    attribs[0].value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME;

    attribs[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
    attribs[1].type = VASurfaceAttribExternalBufferDescriptor;
    attribs[1].value.type = VAGenericValueTypePointer;
    attribs[1].value.value.p = &external;

    VASurfaceID dst_surface = VA_INVALID_SURFACE;
    status = dst_display.drvVtable().vaCreateSurfaces2(dst_display.drvCtx(), rt_format,
                                                       drm_descriptor.width, drm_descriptor.height,
                                                       &dst_surface, 1, attribs, 2);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateSurfaces2 failed, " + std::to_string(status));
    return dst_surface;
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

VaApiFrameConverter::VaApiFrameConverter(VADisplay va_display, uint32_t out_width,
                                         uint32_t out_height, uint32_t out_color_format)
    : output_resolution_{out_width, out_height}, output_color_format_(out_color_format),
      context_(std::make_shared<VaApiContext>(va_display)) {
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
    if (!initalized_)
        init_frame_converter();
    auto dst_frame = frame_pool->acquire();
    bool owns_src_surface = false;
    uint64_t fd = 0;
    VASurfaceID src_surface = src_frame.desc.va_surface_id;
    if (src_frame.desc.va_display != dst_frame->desc.va_display) {
        src_surface = ConvertVASurfaceFromDifferentDriverContext(
            VaDpyWrapper::fromHandle(src_frame.desc.va_display), src_surface, context_->Display(),
            context_->RTFormat(), fd);
        owns_src_surface = true;
    }

    VAProcPipelineParameterBuffer pipeline_param;
    std::memset(&pipeline_param, 0, sizeof(pipeline_param));

    pipeline_param.surface = src_surface;

    // Scale and csc mode
    pipeline_param.filter_flags = VA_FILTER_SCALING_FAST;

    // Crop ROI
    VARectangle src_surface_region = {.x = static_cast<int16_t>(0),
                                      .y = static_cast<int16_t>(0),
                                      .width = src_frame.desc.width,
                                      .height = src_frame.desc.height};
    if (src_surface_region.width > 0 && src_surface_region.height > 0)
        pipeline_param.surface_region = &src_surface_region;

    // Resize to this Rect
    VARectangle dst_surface_region = {.x = static_cast<int16_t>(0),
                                      .y = static_cast<int16_t>(0),
                                      .width = dst_frame->desc.width,
                                      .height = dst_frame->desc.height};
    pipeline_param.output_region = &dst_surface_region;

    auto va_drv_context = context_->Display().drvCtx();
    const auto& vtable = context_->Display().drvVtable();
    VABufferID pipeline_param_buf_id = VA_INVALID_ID;

    auto status =
        vtable.vaCreateBuffer(va_drv_context, context_->Id(), VAProcPipelineParameterBufferType,
                              sizeof(pipeline_param), 1, &pipeline_param, &pipeline_param_buf_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateBuffer failed, " + std::to_string(status));

    {
        status =
            vtable.vaBeginPicture(va_drv_context, context_->Id(), dst_frame->desc.va_surface_id);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaBeginPicture failed, " + std::to_string(status));

        status = vtable.vaRenderPicture(va_drv_context, context_->Id(), &pipeline_param_buf_id, 1);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaRenderPicture failed, " + std::to_string(status));

        status = vtable.vaEndPicture(va_drv_context, context_->Id());
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaEndPicture failed, " + std::to_string(status));
    }

    status = vtable.vaDestroyBuffer(va_drv_context, pipeline_param_buf_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaDestroyBuffer failed, " + std::to_string(status));

    if (owns_src_surface) {
        status = vtable.vaDestroySurfaces(va_drv_context, &src_surface, 1);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaDestroySurfaces failed, " + std::to_string(status));

        if (close(fd) == -1)
            throw std::runtime_error("VaApiConverter::Convert: close fd failed.");
    }
    return dst_frame;
}
