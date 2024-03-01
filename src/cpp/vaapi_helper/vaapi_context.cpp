#include "vaapi_context.hpp"
#include "unistd.h"
#include <cassert>
#include <fcntl.h>

VaApiContext::VaApiContext(VADisplay va_display) : _display(va_display) {
    create_config_and_contexts();
    create_supported_pixel_formats();

    assert(_va_config_id != VA_INVALID_ID &&
           "Failed to initalize VaApiContext. Expected valid VAConfigID.");
    assert(_va_context_id != VA_INVALID_ID &&
           "Failed to initalize VaApiContext. Expected valid VAContextID.");
}

VaApiContext::VaApiContext(std::string_view device) {
    init_vaapi_objects(device);
    create_config_and_contexts();
    create_supported_pixel_formats();

    assert(_va_config_id != VA_INVALID_ID &&
           "Failed to initalize VaApiContext. Expected valid VAConfigID.");
    assert(_va_context_id != VA_INVALID_ID &&
           "Failed to initalize VaApiContext. Expected valid VAContextID.");
}

void VaApiContext::init_vaapi_objects(std::string_view device) {
    // Silent vainfo
    setenv("LIBVA_MESSAGING_LEVEL", "1", 1);

    // Init VADisplay
    drm_fd_ = open(device.data(), O_RDWR);
    if (drm_fd_ < 0)
        throw std::runtime_error("Failed to open device: " + std::string(device));
    auto va_display_ = vaGetDisplayDRM(drm_fd_);
    int major, minor;
    auto status = vaInitialize(va_display_, &major, &minor);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaInitialize failed, " + std::to_string(status));
    _display = VaDpyWrapper(va_display_);
}

VaApiContext::~VaApiContext() {
    auto vtable = _display.drvVtable();
    auto ctx = _display.drvCtx();

    if (_va_context_id != VA_INVALID_ID) {
        vtable.vaDestroyContext(ctx, _va_context_id);
    }

    if (_va_config_id != VA_INVALID_ID) {
        vtable.vaDestroyConfig(ctx, _va_config_id);
    }
    if (drm_fd_ >= 0) {
        close(drm_fd_);
    }
}

VAContextID VaApiContext::Id() const {
    return _va_context_id;
}

VaDpyWrapper VaApiContext::Display() const {
    return _display;
}

VADisplay VaApiContext::DisplayRaw() const {
    return _display.raw();
}

int VaApiContext::RTFormat() const {
    return _rt_format;
}

bool VaApiContext::IsPixelFormatSupported(int format) const {
    return _supported_pixel_formats.count(format);
}

/**
 * Creates config, va context, and sets the driver context using the internal VADisplay.
 * Setting the VADriverContextP, VAConfigID, and VAContextID to the corresponding variables.
 *
 * @pre _display must be set and initialized.
 * @post _va_config_id is set.
 * @post _va_context_id is set.
 *
 * @throw std::invalid_argument if the VaDpyWrapper is not created, runtime format not supported,
 * unable to get config attributes, unable to create config, or unable to create context.
 */
void VaApiContext::create_config_and_contexts() {
    assert(_display);

    auto ctx = _display.drvCtx();
    auto vtable = _display.drvVtable();

    VAConfigAttrib format_attrib;
    format_attrib.type = VAConfigAttribRTFormat;
    auto status =
        vtable.vaGetConfigAttributes(ctx, VAProfileNone, VAEntrypointVideoProc, &format_attrib, 1);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaGetConfigAttributes failed, " + std::to_string(status));

    if (not(format_attrib.value & _rt_format))
        throw std::invalid_argument("Could not create context. Runtime format is not supported.");

    VAConfigAttrib attrib;
    attrib.type = VAConfigAttribRTFormat;
    attrib.value = _rt_format;

    status = vtable.vaCreateConfig(ctx, VAProfileNone, VAEntrypointVideoProc, &attrib, 1,
                                   &_va_config_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateConfig failed, " + std::to_string(status));

    if (_va_config_id == 0) {
        throw std::invalid_argument(
            "Could not create VA config. Cannot initialize VaApiContext without VA config.");
    }

    status = vtable.vaCreateContext(ctx, _va_config_id, 1, 1, VA_PROGRESSIVE, nullptr, 0,
                                    &_va_context_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateContext failed, " + std::to_string(status));

    if (_va_context_id == 0) {
        throw std::invalid_argument(
            "Could not create VA context. Cannot initialize VaApiContext without VA context.");
    }
}

/**
 * Creates a set of formats supported by image.
 *
 * @pre _display must be set and initialized.
 * @post _supported_pixel_formats is set.
 *
 * @throw std::runtime_error if vaQueryImageFormats return non success code
 */
void VaApiContext::create_supported_pixel_formats() {
    assert(_display);

    auto ctx = _display.drvCtx();
    auto vtable = _display.drvVtable();

    std::vector<VAImageFormat> image_formats(ctx->max_image_formats);
    int size = 0;
    auto status = vtable.vaQueryImageFormats(ctx, image_formats.data(), &size);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaQueryImageFormats failed, " + std::to_string(status));

    for (int i = 0; i < size; i++)
        _supported_pixel_formats.insert(image_formats[i].fourcc);
}
