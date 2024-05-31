#pragma once

#include <utility>
#include <memory>

#include "vaapi_context.hpp"
#include "vaapi_frame_pool.hpp"

// TODO: std::enable_shared_from_this should be resolved/improved either by:
// - Update VaApiFramePool so it returns smart pointer by default and get rid of
// enable_shared_from_this
// - Introduce factory methods for object construction that returns shared_ptr
class VaApiFrameConverter : public std::enable_shared_from_this<VaApiFrameConverter> {
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
    VaApiFrameConverter(VaDpyWrapper va_display, uint32_t out_width, uint32_t out_height,
                        uint32_t out_color_format);
    VaApiFrame* convert(const VaApiFrame& src_frame);
    VaApiFrame* convert(const VaApiFrame& src_frame, VARectangle src_region);

    // Return type for convert_ex operation.
    // Returns a pair of resulting frame as smart pointer
    // and flag that indicates whether dynamic surface allocation took place
    using ConvertResult = std::pair<std::unique_ptr<VaApiFrame>, bool>;

    // Extended version of convert.
    // It has option to enable dynamic VA surface allocation if case of empty pool
    ConvertResult convert_ex(const VaApiFrame& src_frame, bool allow_dyn_allocation);

    ConvertResult convert_ex(const VaApiFrame& src_frame, VARectangle src_region,
                             bool allow_dyn_allocation);

    void release_frame(VaApiFrame* frame); // TODO can we do it better?
    void set_output_color_format(uint32_t format);
    void set_ouput_resolution(int width, int height);
    void set_pool_size(uint32_t pool_size);
    uint32_t get_pool_size() const { return pool_size_; }

  private:
    // Allocates a frame from internal pool.
    // If nowait is true and pool is empty returns nullptr, otherwise waits for a free frame
    VaApiFrame* get_frame_from_pool(bool nowait);

    void convert_internal(const VaApiFrame& src_frame, VaApiFrame* dst_frame,
                          VARectangle src_region);
};
