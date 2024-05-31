#include "vaapi_context.hpp"
#include "vaapi_frame.hpp"
#include "vaapi_utils.hpp"
#include <iostream>

VASurfaceID create_va_surface(VADisplay display, uint32_t width, uint32_t height, int pixel_format,
                              int rt_format) {
    VASurfaceAttrib surface_attrib;
    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeInteger;
    surface_attrib.value.value.i = pixel_format;
    VASurfaceID va_surface_id;
    auto status =
        vaCreateSurfaces(display, rt_format, width, height, &va_surface_id, 1, &surface_attrib, 1);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateSurfaces failed, " + std::to_string(status));

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

uint32_t get_supported_va_copy_mode(VADisplay display) {
    // VA_EXEC_MODE_POWER_SAVING to use "Copy Engine" instead of "Compute Engine" as it gives perf
    // boost in Media + Compute workloads. Note not every platform supports this copy mode so we
    // need to query its capabilities
    VADisplayAttribute attr = {.type = VADisplayAttribCopy};
    VAStatus va_status = vaGetDisplayAttributes(display, &attr, 1);
    if (va_status != VA_STATUS_SUCCESS)
        throw std::runtime_error("Failed to get device VADisplayAttribCopy attribute: " +
                                 std::to_string(va_status));
    if (attr.value == 0)
        throw std::runtime_error("vaCopy is not supported on current platform");
    return (attr.value & (1 << VA_EXEC_MODE_POWER_SAVING)) ? VA_EXEC_MODE_POWER_SAVING
                                                           : VA_EXEC_MODE_DEFAULT;
}

void copy(const VaApiFrame& src, const VaApiFrame& dst) {
    VACopyObject dest_obj = {};
    dest_obj.obj_type = VACopyObjectSurface;
    dest_obj.object.surface_id = dst.desc.va_surface_id;
    if (dest_obj.object.surface_id == VA_INVALID_SURFACE)
        throw std::runtime_error("Destination surface ID is invalid.");

    VACopyObject src_obj = {};
    src_obj.obj_type = VACopyObjectSurface;
    src_obj.object.surface_id = src.desc.va_surface_id;

    if (src_obj.object.surface_id == VA_INVALID_SURFACE)
        throw std::runtime_error("Source surface ID is invalid.");

    VACopyOption va_copy_option = {};
    va_copy_option.bits.va_copy_sync = VA_EXEC_ASYNC;
    static uint32_t supported_va_copy_mode = get_supported_va_copy_mode(src.desc.va_display);
    va_copy_option.bits.va_copy_mode = supported_va_copy_mode;
    auto va_status = vaCopy(src.desc.va_display, &dest_obj, &src_obj, va_copy_option);
    if (va_status != VA_STATUS_SUCCESS)
        throw std::runtime_error("Failed to copy VaApiFrame frame with status: " +
                                 std::to_string(va_status));
}

VaApiFrame::VaApiFrame(VADisplay va_display, uint32_t va_surface_id, uint32_t width,
                       uint32_t height, int format, bool owns /*= false*/)
    : owns_surface(owns) {
    // Test that display is valid
    desc.va_display = va_display;
    desc.va_surface_id = va_surface_id;
    desc.format = format;
    desc.width = width;
    desc.height = height;
}

VaApiFrame::VaApiFrame(VADisplay va_display, uint32_t va_surface_id, bool owns /*= false*/)
    : owns_surface(owns) {
    // Test that display is valid
    desc.va_display = va_display;
    desc.va_surface_id = va_surface_id;

    VAImage vaimage;
    auto status = vaDeriveImage(va_display, va_surface_id, &vaimage);
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

    status = vaDestroyImage(va_display, vaimage.image_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaDestroyImage() failed: " + std::to_string(status));
}

VaApiFrame::VaApiFrame(VADisplay va_display, uint32_t width, uint32_t height, int format)
    : owns_surface(true) {
    desc.width = width;
    desc.height = height;
    desc.format = format;
    desc.va_display = va_display;
    const int rt_format = get_va_rt_format_for_fourcc(desc.format);
    desc.va_surface_id =
        create_va_surface(va_display, desc.width, desc.height, desc.format, rt_format);
}

std::unique_ptr<VaApiFrame> VaApiFrame::copy_from(const VaApiFrame& other) {
    auto nv12_copy = std::make_unique<VaApiFrame>();
    nv12_copy->desc = other.desc;
    const int rt_format = get_va_rt_format_for_fourcc(nv12_copy->desc.format);

    nv12_copy->desc.va_surface_id =
        create_va_surface(other.desc.va_display, nv12_copy->desc.width, nv12_copy->desc.height,
                          nv12_copy->desc.format, rt_format);

    copy(other, *nv12_copy);
    nv12_copy->owns_surface = true;
    return nv12_copy;
}

std::unique_ptr<SystemFrame> VaApiFrame::copy_to_system() {
    if (desc.format != VA_FOURCC_RGBP)
        throw std::runtime_error("Unsupported pixel format: " + std::to_string(desc.format));

    const unsigned int rt_format = get_va_rt_format_for_fourcc(desc.format);

    VASurfaceAttrib surface_attrib[3] = {};
    surface_attrib[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib[0].type = VASurfaceAttribPixelFormat;
    surface_attrib[0].value.type = VAGenericValueTypeInteger;
    surface_attrib[0].value.value.i = desc.format;

    surface_attrib[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib[1].type = VASurfaceAttribMemoryType;
    surface_attrib[1].value.type = VAGenericValueTypeInteger;
    surface_attrib[1].value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_USER_PTR;

    surface_attrib[2].flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib[2].type = VASurfaceAttribExternalBufferDescriptor;
    surface_attrib[2].value.type = VAGenericValueTypePointer;
    VASurfaceAttribExternalBuffers ext_buffer{};
    surface_attrib[2].value.value.p = &ext_buffer;

    // Set up external buffer pitches and offsets
    ext_buffer.pitches[0] = desc.width;
    ext_buffer.offsets[0] = 0;
    ext_buffer.offsets[1] = ext_buffer.pitches[0] * desc.height;
    ext_buffer.pitches[1] = ext_buffer.pitches[0];
    ext_buffer.offsets[2] = ext_buffer.pitches[0] * desc.height * 2;
    ext_buffer.pitches[2] = ext_buffer.pitches[0];
    ext_buffer.num_planes = 3;

    // Allocate and align memory
    const uint32_t base_addr_align = 0x1000;
    uint32_t size = (ext_buffer.pitches[0] * desc.height) * 3; // frame size align with pitch.
    size = (size + base_addr_align - 1) & ~(base_addr_align - 1);
    size_t space = size + base_addr_align;
    std::vector<uint8_t> system_memory(space);
    void* system_memory_ptr = system_memory.data();
    if (!std::align(base_addr_align, size, system_memory_ptr, space))
        throw std::runtime_error("Failed to align memory.");

    // Set up external buffer memory and attributes
    ext_buffer.pixel_format = desc.format;
    ext_buffer.width = desc.width;
    ext_buffer.height = desc.height;
    ext_buffer.data_size = size;
    ext_buffer.num_buffers = 1;
    ext_buffer.buffers = reinterpret_cast<uintptr_t*>(&system_memory_ptr);
    ext_buffer.flags = VA_SURFACE_ATTRIB_MEM_TYPE_USER_PTR;

    // Create VA surface
    VASurfaceID va_surface_id;
    auto va_status = vaCreateSurfaces(desc.va_display, rt_format, desc.width, desc.height,
                                      &va_surface_id, 1, surface_attrib, 3);
    if (va_status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateSurfaces() failed: " + std::to_string(va_status));

    auto va_frame = std::make_unique<VaApiFrame>(desc.va_display, va_surface_id, desc.width,
                                                 desc.height, desc.format, true);

    copy(*this, *va_frame);

    std::vector<int64_t> shape = {3, desc.height, desc.width};
    std::vector<int64_t> strides = {ext_buffer.pitches[0] * desc.height, ext_buffer.pitches[0], 1};
    uint64_t offset = static_cast<uint8_t*>(system_memory_ptr) - system_memory.data();
    return std::make_unique<SystemFrame>(std::move(system_memory), std::move(shape),
                                         std::move(strides), offset, std::move(va_frame));
}

VaApiFrame::~VaApiFrame() {
    if (!owns_surface || desc.va_surface_id == VA_INVALID_ID)
        return;

    try {
        auto status = vaDestroySurfaces(desc.va_display, &desc.va_surface_id, 1);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error("vaDestroySurfaces failed, " + std::to_string(status));

    } catch (const std::exception& e) {
        // log_error("Failed to destroy surface: {}. Error: {}", desc.va_surface_id, e.what());
    }
}

bool VaApiFrame::is_ready() const {
    VASurfaceStatus surface_status = VASurfaceRendering;
    VAStatus sts = vaQuerySurfaceStatus(desc.va_display, desc.va_surface_id, &surface_status);
    if (sts != VA_STATUS_SUCCESS)
        throw std::runtime_error(std::string("vaQuerySurfaceStatus") +
                                 " failed: " + std::to_string(sts));

    return surface_status == VASurfaceReady;
}

void VaApiFrame::sync() const {
    if (is_ready())
        return;

    // Warning: Here we use polling status loop because of VaSyncSurface on batch 512 issue.
    // TODO: Change it to vaSyncSurface when issue fixed
    while (!is_ready()) {
    }
}
