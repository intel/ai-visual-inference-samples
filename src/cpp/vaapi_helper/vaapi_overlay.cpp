#include "vaapi_overlay.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vaapi_frame.hpp>
#include <scope_guard.hpp>

using FontConfig = VaApiOverlay::FontConfig;

cv::Rect calc_bounding_rect_for_primitives(const std::vector<cv::Rect>& boxes,
                                           const std::vector<OverlayText>& texts,
                                           const std::vector<cv::Size>& texts_size) {
    assert(texts.size() == texts_size.size());
    std::vector<cv::Point> points;
    // 2 points for every rect and text
    points.reserve(boxes.size() * 2 + texts.size() * 2);

    // Boxes to points
    for (auto& b : boxes) {
        points.emplace_back(b.tl());
        points.emplace_back(b.br());
    }

    // Texts to points
    for (size_t i = 0; i < texts.size(); i++) {
        const auto& t = texts[i];
        const auto& tsize = texts_size[i];
        const cv::Point tl(t.x, t.y);
        points.emplace_back(tl);
        points.emplace_back(tl + cv::Point(tsize)); // Bottom-right
    }

    return cv::boundingRect(points);
}

std::tuple<cv::Mat, cv::Rect> prepare_mat(const std::vector<OverlayBox>& boxes,
                                          const std::vector<OverlayText>& texts, FontConfig font) {
    constexpr int LINE_THICKNESS = 2;

    std::vector<cv::Rect> cvboxes;
    cvboxes.reserve(boxes.size());

    std::transform(boxes.begin(), boxes.end(), std::back_inserter(cvboxes),
                   [](const OverlayBox& b) {
                       return cv::Rect{cv::Point{b.x1, b.y1}, cv::Point{b.x2, b.y2}};
                   });

    // Get texts sizes
    std::vector<cv::Size> texts_size;
    texts_size.reserve(texts.size());

    std::transform(texts.cbegin(), texts.cend(), std::back_inserter(texts_size),
                   [&font](const OverlayText& t) {
                       return cv::getTextSize(t.text, font.face, font.scale, font.thickness,
                                              nullptr);
                   });

    const auto bounding_rect = calc_bounding_rect_for_primitives(cvboxes, texts, texts_size);

    const auto color = cv::Scalar(255, 0, 0, 255);
    const auto margin_size = cv::Size(LINE_THICKNESS / 2, LINE_THICKNESS / 2);
    cv::Mat mat(bounding_rect.size() + margin_size, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    // Draw boxes
    for (auto& b : cvboxes) {
        b -= bounding_rect.tl() - cv::Point(margin_size);
        cv::rectangle(mat, b, color, LINE_THICKNESS);
    }

    // Draw texts
    for (size_t i = 0; i < texts.size(); i++) {
        const auto& t = texts[i];
        const auto& tsize = texts_size[i];
        const auto org = cv::Point{t.x - bounding_rect.tl().x, t.y + tsize.height};
        cv::putText(mat, t.text, org, font.face, font.scale, cv::Scalar(0, 255, 0, 255),
                    font.thickness);
    }

    return {std::move(mat), bounding_rect};
}

// FIXME: can we draw directly on mapped surface
std::unique_ptr<VaApiFrame> cvmat_to_vaapi_frame(VADisplay va_display, cv::Mat& mat) {
    assert(mat.type() == CV_8UC4);
    auto frame = std::make_unique<VaApiFrame>(va_display, mat.size().width, mat.size().height,
                                              VA_FOURCC_RGBA);

    VAImage surface_img;
    auto status = vaDeriveImage(va_display, frame->desc.va_surface_id, &surface_img);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error(std::string("vaDeriveImage failed, error: ") + vaErrorStr(status));
    auto sg_image = make_scope_guard([&] {
        auto sts = vaDestroyImage(va_display, surface_img.image_id);
        if (sts != VA_STATUS_SUCCESS)
            throw std::runtime_error(std::string("vaDestroyImage failed, error: ") +
                                     vaErrorStr(status));
    });

    void* data_ptr = nullptr;
    status = vaMapBuffer(va_display, surface_img.buf, &data_ptr);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error(std::string("vaMapBuffer failed, error: ") + vaErrorStr(status));
    auto sg_buffer = make_scope_guard([&] {
        auto sts = vaUnmapBuffer(va_display, surface_img.buf);
        if (sts != VA_STATUS_SUCCESS)
            throw std::runtime_error(std::string("vaUnmapBuffer failed, error: ") +
                                     vaErrorStr(status));
    });

    const uint8_t* src = mat.data;
    const auto src_step = mat.step[0];
    uint8_t* dst = static_cast<uint8_t*>(data_ptr) + surface_img.offsets[0];

    for (uint32_t row = 0; row < surface_img.height; row++) {
        memcpy(dst, src, src_step);
        dst += surface_img.pitches[0];
        src += src_step;
    }

    return frame;
}

std::tuple<cv::Rect, bool> clip_region(cv::Rect region, cv::Size img_size) {
    auto clipped = region & cv::Rect{{0, 0}, img_size};
    return {clipped, region != clipped};
}

std::tuple<cv::Mat, cv::Rect, bool> clip_mat(cv::Mat mat, cv::Point mat_pos, cv::Size img_size) {
    auto [mat_region, clipped] = clip_region(cv::Rect{mat_pos, mat.size()}, img_size);
    if (clipped)
        mat = mat(cv::Rect({0, 0}, mat_region.size()));

    return {mat, mat_region, clipped};
}

VARectangle cvrect2varect(cv::Rect r) {
    return VARectangle{.x = int16_t(r.x),
                       .y = int16_t(r.y),
                       .width = uint16_t(r.width),
                       .height = uint16_t(r.height)};
}

VaApiOverlay::VaApiOverlay(VaDpyWrapper va_display)
    : va_display_(va_display.native()),
      context_(std::make_shared<VaApiContext>(std::move(va_display))) {
    logger_.debug("overlay::ctor: display={:#x}, sync={}", reinterpret_cast<uintptr_t>(va_display_),
                  sync_surface_);
}

VaApiOverlay::~VaApiOverlay() {
}

std::unique_ptr<VaApiFrame> VaApiOverlay::draw(VaApiFrame& src_frame, std::vector<OverlayBox> boxes,
                                               std::vector<OverlayText> texts) {
    using namespace std::chrono;

    logger_.debug("overlay::draw: surface={}, wxh={}x{}, num boxes={}, num texts={}",
                  src_frame.desc.va_surface_id, src_frame.desc.width, src_frame.desc.height,
                  boxes.size(), texts.size());

    const auto tp0 = high_resolution_clock::now();

    auto [mat_first, bounding_rec] = prepare_mat(boxes, texts, font_cfg_);

    // Make sure that mat fits on image at desired position, clip if needed
    auto [mat, mask_region, clipped] =
        clip_mat(mat_first, bounding_rec.tl(), {src_frame.desc.width, src_frame.desc.height});

    if (clipped)
        logger_.warn("Drawing area was cropped due to elements outside the field. Check "
                     "position & size of elements.");

    const auto tp1 = high_resolution_clock::now();

    auto mask_frame = cvmat_to_vaapi_frame(context_->display_native(), mat);
    if (!mask_frame)
        throw std::runtime_error("overlay: couldn't convert cvmat to vasurface");

    const auto tp2 = high_resolution_clock::now();

    auto out_frame = blend(src_frame, *mask_frame, cvrect2varect(mask_region));

    const auto tp3 = high_resolution_clock::now();
    stats_.last_total = duration_cast<microseconds>(tp3 - tp0);
    stats_.last_mask_prepare = duration_cast<microseconds>(tp1 - tp0);
    stats_.last_mask_to_surface = duration_cast<microseconds>(tp2 - tp1);
    stats_.last_blend = duration_cast<microseconds>(tp3 - tp2);

    // Link to OpenCV imgcodecs if stuff below is needed
    // if (dump_mask_)
    //     imwrite("./overlay-mask.png", mat);

    return std::move(out_frame);
}

std::unique_ptr<VaApiFrame> VaApiOverlay::blend(VaApiFrame& src, VaApiFrame& mask,
                                                VARectangle mask_region) {
    auto out_frame = std::make_unique<VaApiFrame>(context_->display_native(), src.desc.width,
                                                  src.desc.height, src.desc.format);

    const auto out_surface = out_frame->desc.va_surface_id;

    auto status = vaBeginPicture(va_display_, context_->id_native(), out_surface);
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error(std::string("overlay: vaBeginPicture failed, error: ") +
                                 vaErrorStr(status));

    VAProcPipelineParameterBuffer params[2] = {};

    auto out_region =
        VARectangle{.x = 0, .y = 0, .width = src.desc.width, .height = src.desc.height};
    params[0].surface = src.desc.va_surface_id;
    params[0].surface_region = &out_region;
    params[0].output_region = &out_region;
    params[0].output_background_color = 0xff000000;
    params[0].filter_flags = VA_FRAME_PICTURE;

    // Blend
    params[1].surface = mask.desc.va_surface_id;
    params[1].filter_flags = VA_FRAME_PICTURE;
    params[1].surface_region = nullptr;
    params[1].output_region = &mask_region;

    VABlendState blend_state = {};
    blend_state.flags = VA_BLEND_PREMULTIPLIED_ALPHA;
    params[1].blend_state = &blend_state;
    params[1].output_background_color = 0xff000000;

    for (size_t i = 0; i < std::size(params); i++) {
        VABufferID params_id;
        status =
            vaCreateBuffer(va_display_, context_->id_native(), VAProcPipelineParameterBufferType,
                           sizeof(VAProcPipelineParameterBuffer), 1, &params[i], &params_id);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error(std::string("overlay: vaCreateBuffer failed (") +
                                     std::to_string(i) + "), error:" + vaErrorStr(status));

        auto sg_buffer = make_scope_guard([&] {
            status = vaDestroyBuffer(va_display_, params_id);
            if (status != VA_STATUS_SUCCESS)
                logger_.error("overlay: vaDestroyBuffer failed ({}): bufid={}, error={}", i,
                              params_id, vaErrorStr(status));
        });

        status = vaRenderPicture(va_display_, context_->id_native(), &params_id, 1);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error(std::string("overlay: vaRenderPicture failed (") +
                                     std::to_string(i) + "), error:" + vaErrorStr(status));
    }

    status = vaEndPicture(va_display_, context_->id_native());
    if (status != VA_STATUS_SUCCESS)
        throw std::runtime_error(std::string("overlay: vaEndPicture failed, error: ") +
                                 vaErrorStr(status));

    if (sync_surface_) {
#if 0
        // This is not required. But using this we can tell once rendering is done.
        VASurfaceStatus surface_status = static_cast<VASurfaceStatus>(0);
        do {
            status = vaQuerySurfaceStatus(va_display_, out_surface, &surface_status);
            if (status != VA_STATUS_SUCCESS)
                throw std::runtime_error(
                    std::string("overlay: vaQuerySurfaceStatus failed, error: ") +
                    vaErrorStr(status));

            logger_.debug("overlay: surface status={}, id={}", static_cast<int>(surface_status), out_surface);
        } while (surface_status != VASurfaceReady);
#endif

        status = vaSyncSurface(va_display_, out_surface);
        if (status != VA_STATUS_SUCCESS)
            throw std::runtime_error(std::string("overlay: vaSyncSurface failed, error: ") +
                                     vaErrorStr(status));
    }

    return out_frame;
}
