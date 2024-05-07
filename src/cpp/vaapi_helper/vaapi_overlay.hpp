#pragma once

#include <vaapi_frame.hpp>
#include <vaapi_context.hpp>

#include "logger.hpp"

// Box coordinates in xyxy format
struct OverlayBox {
    int x1;
    int y1;
    int x2;
    int y2;
};

struct OverlayText {
    int x;
    int y;
    std::string text;
};

// Overlay based on VAAPI blending
class VaApiOverlay final {
  public:
    struct FontConfig {
        int face = 0; // FONT_HERSHEY_SIMPLEX
        int thickness = 2;
        double scale = 1.0;
    };

    struct Stats {
        std::chrono::microseconds last_total;
        std::chrono::microseconds last_mask_prepare;
        std::chrono::microseconds last_mask_to_surface;
        std::chrono::microseconds last_blend;
    };

    VaApiOverlay(VaDpyWrapper va_display);

    ~VaApiOverlay();

    // No copy
    VaApiOverlay(const VaApiOverlay&) = delete;
    VaApiOverlay& operator=(const VaApiOverlay&) = delete;

    std::unique_ptr<VaApiFrame> draw(VaApiFrame& src_frame, std::vector<OverlayBox> boxes,
                                     std::vector<OverlayText> texts);

    void set_sync(bool enable) { sync_surface_ = enable; }
    bool get_sync() const { return sync_surface_; }

    VaApiOverlay::Stats stats() const { return stats_; }

  private:
    Logger logger_;
    VADisplay va_display_;
    VaApiContextPtr context_;
    bool sync_surface_ = true;

    FontConfig font_cfg_;

    Stats stats_;

    std::unique_ptr<VaApiFrame> blend(VaApiFrame& src, VaApiFrame& mask, VARectangle mask_region);
};
