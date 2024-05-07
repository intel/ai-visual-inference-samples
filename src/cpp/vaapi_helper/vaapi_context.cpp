#include "vaapi_context.hpp"
#include "unistd.h"
#include <cassert>
#include <fcntl.h>
#include <iostream>

VaDpyWrapper::Storage::Storage(std::string_view device) {
    // Disabling libva info messages
    setenv("LIBVA_MESSAGING_LEVEL", "1", 1);

    drm_fd = open(device.data(), O_RDWR);
    if (drm_fd < 0)
        throw std::runtime_error("Failed to open device: " + std::string(device));
    display = vaGetDisplayDRM(drm_fd);
    int major, minor;
    auto status = vaInitialize(display, &major, &minor);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaInitialize failed, " + std::to_string(status));
}

VaDpyWrapper::Storage::~Storage() {
    if (display) {
        auto status = vaTerminate(display);
        if (status != VA_STATUS_SUCCESS)
            std::cout << "vaTerminate failed. Status: " << std::to_string(status) << "\n";
    }
    if (drm_fd >= 0)
        close(drm_fd);
}

VaApiContext::VaApiContext(VaDpyWrapper va_display) : _display(va_display) {
    create_config_and_contexts();
    create_supported_pixel_formats();

    assert(_va_config_id != VA_INVALID_ID &&
           "Failed to initalize VaApiContext. Expected valid VAConfigID.");
    assert(_va_context_id != VA_INVALID_ID &&
           "Failed to initalize VaApiContext. Expected valid VAContextID.");
}

VaApiContext::~VaApiContext() {
    if (_va_context_id != VA_INVALID_ID) {
        vaDestroyContext(_display.native(), _va_context_id);
    }

    if (_va_config_id != VA_INVALID_ID) {
        vaDestroyConfig(_display.native(), _va_config_id);
    }
}

VAContextID VaApiContext::id_native() const {
    return _va_context_id;
}

const VaDpyWrapper& VaApiContext::display() const {
    return _display;
}

VADisplay VaApiContext::display_native() const {
    return _display.native();
}

bool VaApiContext::is_pixel_format_supported(int format) const {
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

    VAConfigAttrib format_attrib;
    format_attrib.type = VAConfigAttribRTFormat;
    auto status = vaGetConfigAttributes(_display.native(), VAProfileNone, VAEntrypointVideoProc,
                                        &format_attrib, 1);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaGetConfigAttributes failed, " + std::to_string(status));

    if (not(format_attrib.value & _rt_format))
        throw std::invalid_argument("Could not create context. Runtime format is not supported.");

    VAConfigAttrib attrib;
    attrib.type = VAConfigAttribRTFormat;
    attrib.value = _rt_format;

    status = vaCreateConfig(_display.native(), VAProfileNone, VAEntrypointVideoProc, &attrib, 1,
                            &_va_config_id);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaCreateConfig failed, " + std::to_string(status));

    if (_va_config_id == 0) {
        throw std::invalid_argument(
            "Could not create VA config. Cannot initialize VaApiContext without VA config.");
    }

    // We use width=1 and height=1 as defaults, because width=0 and height=0 causes permanent device
    // busy on WSL. In general these parameters should not affect anything for Video Processing
    // Context
    status = vaCreateContext(_display.native(), _va_config_id, 1, 1, VA_PROGRESSIVE, nullptr, 0,
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

    std::vector<VAImageFormat> image_formats(vaMaxNumImageFormats(_display.native()));
    int size = 0;
    auto status = vaQueryImageFormats(_display.native(), image_formats.data(), &size);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error("vaQueryImageFormats failed, " + std::to_string(status));

    for (int i = 0; i < size; i++)
        _supported_pixel_formats.insert(image_formats[i].fourcc);
}
