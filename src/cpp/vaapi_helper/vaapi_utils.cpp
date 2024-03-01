
#include "vaapi_utils.hpp"
#include <iostream>
#include <scope_guard.hpp>

// use command below to convert nv12 surface to bmp image
// ffmpeg -s <width>x<height> -pix_fmt nv12 -f rawvideo -i dump.nv12 out.bmp
bool dump_va_surface(VADisplay display, VASurfaceID surface, const std::string& filename) {
    // log_info("dump_va_surface: display={}, surface={}, file={}", display, surface, filename);

    FILE* fp = fopen(filename.data(), "wb+");
    if (!fp) {
        std::cerr << "dump_va_surface: fopen failed\n";
        return false;
    }
    auto fp_guard = make_scope_guard([&] { fclose(fp); });

    VAImage vaimage;
    auto status = vaDeriveImage(display, surface, &vaimage);
    if (status != VA_STATUS_SUCCESS) {
        std::cerr << "dump_va_surface: vaDeriveImage failed, code=" << status << std::endl;
        return false;
    }
    auto image_guard = make_scope_guard([&] {
        status = vaDestroyImage(display, vaimage.image_id);
        if (status != VA_STATUS_SUCCESS)
            std::cerr << "dump_va_surface: vaDestroyImage failed, code=" << status << std::endl;
    });

    if (vaimage.format.fourcc != VA_FOURCC_NV12) {
        std::cerr << "dump_va_surface: only NV12 surfaces are supported, surface format="
                  << vaimage.format.fourcc << std::endl;
        return false;
    }

    void* buf = nullptr;
    status = vaMapBuffer(display, vaimage.buf, &buf);
    if (status != VA_STATUS_SUCCESS) {
        std::cerr << "dump_va_surface: vaMapBuffer failed, code=" << status << std::endl;
        return false;
    }
    auto map_guard = make_scope_guard([&] {
        status = vaUnmapBuffer(display, vaimage.buf);
        if (status != VA_STATUS_SUCCESS)
            std::cerr << "dump_va_surface: vaUnmapBuffer failed, code=" << status << std::endl;
    });

    // dump y_plane
    const auto* y_buf = static_cast<char*>(buf);
    const uint32_t y_pitch = vaimage.pitches[0];
    for (size_t i = 0; i < vaimage.height; i++) {
        fwrite(y_buf + y_pitch * i, vaimage.width, 1, fp);
    }

    // dump uv_plane
    const auto* uv_buf = static_cast<char*>(buf) + vaimage.offsets[1];
    const uint32_t uv_pitch = vaimage.pitches[1];
    for (size_t i = 0; i < vaimage.height / 2; i++) {
        fwrite(uv_buf + uv_pitch * i, vaimage.width, 1, fp);
    }

    std::cout << "dump_va_surface: done, wxh=" << vaimage.width << 'x' << vaimage.height
              << std::endl;
    return true;
}