#include "vaapi_context.hpp"
#include "vaapi_frame.hpp"
#include <iostream>

VASurfaceID create_va_surface(VaDpyWrapper display, uint32_t width, uint32_t height,
                              int pixel_format, int rt_format) {
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeInteger;
    surface_attrib.value.value.i = pixel_format;

    VASurfaceID va_surface_id;
    auto status = display.drvVtable().vaCreateSurfaces2(display.drvCtx(), rt_format, width, height,
                                                        &va_surface_id, 1, &surface_attrib, 1);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateSurfaces2 failed, " + std::to_string(status));

    return va_surface_id;
}

int get_va_rt_format_for_fourcc(int _pixel_format) {
    switch (_pixel_format) {
    case VA_FOURCC_RGBA:
    case VA_FOURCC_BGRA:
        return VA_RT_FORMAT_RGB32;

    case VA_FOURCC_RGBP:
        return VA_RT_FORMAT_RGBP;

    case VA_FOURCC_NV12:
    default: // FIXME
        return VA_RT_FORMAT_YUV420;
    }
}

VaApiFrame::VaApiFrame(void* va_display, uint32_t va_surface_id, uint32_t width, uint32_t height,
                       int format, bool owns /*= false*/)
    : owns_surface(owns) {
    // Test that display is valid
    auto dpy = VaDpyWrapper::fromHandle(va_display);

    desc.va_display = va_display;
    desc.va_surface_id = va_surface_id;
    desc.format = format;
    desc.width = width;
    desc.height = height;
}

VaApiFrame::VaApiFrame(void* va_display, uint32_t va_surface_id, bool owns /*= false*/)
    : owns_surface(owns) {
    // Test that display is valid
    auto dpy = VaDpyWrapper::fromHandle(va_display);

    desc.va_display = va_display;
    desc.va_surface_id = va_surface_id;

    VAImage vaimage;
    auto status = dpy.drvVtable().vaDeriveImage(dpy.drvCtx(), va_surface_id, &vaimage);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaDeriveImage() failed: " + std::to_string(status));
    if (vaimage.width == 0 || vaimage.height == 0) {
        throw std::runtime_error("unexpected image size: width: " + std::to_string(vaimage.width) +
                                 "height: " + std::to_string(vaimage.height));
    }

    desc.format = vaimage.format.fourcc;
    desc.width = vaimage.width;
    desc.height = vaimage.height;
    desc.size = vaimage.data_size;

    status = dpy.drvVtable().vaDestroyImage(dpy.drvCtx(), vaimage.image_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaDestroyImage() failed: " + std::to_string(status));
}

VaApiFrame::VaApiFrame(void* va_display, uint32_t width, uint32_t height, int format)
    : owns_surface(true) {
    auto dpy = VaDpyWrapper::fromHandle(va_display);
    desc.width = width;
    desc.height = height;
    desc.format = format;
    desc.va_display = va_display;
    const int rt_format = get_va_rt_format_for_fourcc(desc.format);
    desc.va_surface_id = create_va_surface(dpy, desc.width, desc.height, desc.format, rt_format);
}

std::unique_ptr<VaApiFrame> VaApiFrame::copy_from(const VaApiFrame& other) {
    auto dpy = VaDpyWrapper::fromHandle(other.desc.va_display);
    auto nv12_copy = std::make_unique<VaApiFrame>();
    nv12_copy->desc = other.desc;
    const int rt_format = get_va_rt_format_for_fourcc(nv12_copy->desc.format);

    nv12_copy->desc.va_surface_id = create_va_surface(
        dpy, nv12_copy->desc.width, nv12_copy->desc.height, nv12_copy->desc.format, rt_format);

    VACopyObject dest_obj = {};
    dest_obj.obj_type = VACopyObjectSurface;
    dest_obj.object.surface_id = nv12_copy->desc.va_surface_id;

    VACopyObject src_obj = {};
    src_obj.obj_type = VACopyObjectSurface;
    src_obj.object.surface_id = other.desc.va_surface_id;

    if (src_obj.object.surface_id == VA_INVALID_SURFACE)
        throw std::runtime_error("Source surface ID is invalid.");

    VACopyOption va_copy_option = {};
    va_copy_option.bits.va_copy_sync = VA_EXEC_ASYNC;
    // VA_EXEC_MODE_POWER_SAVING to use "Copy Engine" instead of "Compute Engine" as it gives perf
    // boost in Media + Compute workloads.
    va_copy_option.bits.va_copy_mode = VA_EXEC_MODE_POWER_SAVING;

    VAStatus va_status = dpy.drvVtable().vaCopy(dpy.drvCtx(), &dest_obj, &src_obj, va_copy_option);
    if (va_status != VA_STATUS_SUCCESS)
        throw std::runtime_error("Failed to copy VaApiFrame frame with status: " +
                                 std::to_string(va_status));

    nv12_copy->owns_surface = true;
    return nv12_copy;
}

VaApiFrame::~VaApiFrame() {
    if (!owns_surface || desc.va_surface_id == VA_INVALID_ID)
        return;

    try {
        auto dpy = VaDpyWrapper::fromHandle(desc.va_display);
        auto status = dpy.drvVtable().vaDestroySurfaces(dpy.drvCtx(), &desc.va_surface_id, 1);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaDestroySurfaces failed, " + std::to_string(status));

    } catch (const std::exception& e) {
        // log_error("Failed to destroy surface: {}. Error: {}", desc.va_surface_id, e.what());
    }
}
