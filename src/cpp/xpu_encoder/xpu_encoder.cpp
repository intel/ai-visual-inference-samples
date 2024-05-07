#include <pybind11/pybind11.h>

#include "context_manager.hpp"
#include "vaapi_context.hpp"
#include "vaapi_frame.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/pixdesc.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
}

namespace py = pybind11;

struct VideoInfo {
    std::string codec_name;
    unsigned int width;
    unsigned int height;
    unsigned int bitrate;
    struct Framerate {
        int num; // Numerator for frame rate
        int den; // Denominator for frame rate
    } framerate;
};

class XpuEncoder {
    AVFormatContext* output_ctx_;
    AVCodecContext* encoder_context_;
    AVHWFramesContext* frames_ctx;
    VideoInfo video_info_;
    AVStream* video_stream_;
    int frame_id_ = 0;
    void configure_encoder_context(const AVCodec* codec, const VADisplay& va_display,
                                   const VideoInfo& video_info);
    AVFrame* get_av_frame_from_va_frame(uint32_t va_surface_id);
    void encode_write(AVFrame* frame);
    bool is_closed_ = false;

  public:
    XpuEncoder(const std::string& output_file, const std::string& device_name,
               const VideoInfo& video_info);
    XpuEncoder(const std::string& output_file, const VADisplay& va_display,
               const VideoInfo& video_info);
    XpuEncoder(const XpuEncoder&) = delete;
    XpuEncoder& operator=(const XpuEncoder&) = delete;
    XpuEncoder& operator=(XpuEncoder&&) = delete;
    XpuEncoder(XpuEncoder&&) = delete;
    ~XpuEncoder();

    void write(uint32_t va_surface_id);
    void flush();
    void close();
};

inline std::string_view av_err2string(int errnum) {
    thread_local char str[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, str, AV_ERROR_MAX_STRING_SIZE);
    return std::string_view(str);
}

XpuEncoder::XpuEncoder(const std::string& output_file, const std::string& device_name,
                       const VideoInfo& video_info)
    : XpuEncoder(
          output_file,
          ContextManager::get_instance()
              .get_va_display(
                  ContextManager::get_instance().get_device_path_from_device_name(device_name))
              .native(),
          video_info) {
}

XpuEncoder::XpuEncoder(const std::string& output_file, const VADisplay& va_display,
                       const VideoInfo& video_info)
    : video_info_(video_info) {

    int err = avformat_alloc_output_context2(&output_ctx_, nullptr, nullptr, output_file.c_str());
    if (err < 0) {
        throw std::runtime_error(
            "Failed to deduce output format from file extension. Error code: " +
            std::string(av_err2string(err)));
    }

    err = avio_open(&output_ctx_->pb, output_file.c_str(), AVIO_FLAG_WRITE);
    if (err < 0) {
        throw std::runtime_error("Cannot open output file. Error code: " +
                                 std::string(av_err2string(err)));
    }

    const AVCodec* codec = avcodec_find_encoder_by_name(video_info.codec_name.c_str());
    if (!codec) {
        throw std::runtime_error("Cannot find encoder '" + video_info.codec_name + "'.");
    }

    configure_encoder_context(codec, va_display, video_info);

    err = avcodec_open2(encoder_context_, codec, nullptr);
    if (err < 0) {
        throw std::runtime_error("Failed to open codec. Error code: " +
                                 std::string(av_err2string(err)));
    }

    video_stream_ = avformat_new_stream(output_ctx_, codec);
    if (!video_stream_) {
        throw std::runtime_error("Failed to allocate stream for output format.");
    }
    video_stream_->time_base = encoder_context_->time_base;

    err = avcodec_parameters_from_context(video_stream_->codecpar, encoder_context_);
    if (err < 0) {
        throw std::runtime_error("avcodec_parameters_from_context failed. Error code: " +
                                 std::string(av_err2string(err)));
    }

    err = avformat_write_header(output_ctx_, nullptr);
    if (err < 0) {
        throw std::runtime_error("Error while writing stream header. Error code: " +
                                 std::string(av_err2string(err)));
    }
}

XpuEncoder::~XpuEncoder() {
    close();
}

void XpuEncoder::close() {
    if (is_closed_) {
        return;
    }

    try {
        flush();
    } catch (std::exception& ex) {
        std::cerr << "Exception occurred while trying to flush encoder: " << ex.what() << std::endl;
    }

    avformat_close_input(&output_ctx_);
    avcodec_free_context(&encoder_context_);
    is_closed_ = true;
}

void XpuEncoder::configure_encoder_context(const AVCodec* codec, const VADisplay& va_display,
                                           const VideoInfo& video_info) {
    encoder_context_ = avcodec_alloc_context3(codec);
    if (!encoder_context_) {
        throw std::runtime_error("avcodec_alloc_context3 failed");
    }
    encoder_context_->bit_rate = video_info.bitrate;
    encoder_context_->width = video_info.width;
    encoder_context_->height = video_info.height;
    encoder_context_->time_base = {video_info.framerate.den, video_info.framerate.num};
    encoder_context_->framerate = {video_info.framerate.num, video_info.framerate.den};

    encoder_context_->sample_aspect_ratio = {1, 1};
    encoder_context_->pix_fmt = AV_PIX_FMT_VAAPI;
    encoder_context_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    encoder_context_->gop_size = video_info.framerate.num / video_info.framerate.den;

    /*
        MUST FIX!
        Long story, if B frame is set, our VA Surface must survive
        for as long as the encoder needs the whole frame for.
        Right now because we are passing the frames batch by batch
        this means the python will kill the previous batch that has
        already been sent to the encoder thinking that it is done with
        those frames. But encoder still relies on them, this will cause
        VA INVALID by encoder. Workaround is either store all NV12 frames
        in python or set b frames to 0.
    */
    encoder_context_->max_b_frames = 0;

    encoder_context_->global_quality = 20;

    AVBufferRef* hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VAAPI);
    if (!hw_device_ctx) {
        throw std::runtime_error("Failed to allocate hardware device context");
    }
    AVHWDeviceContext* hwctx = reinterpret_cast<AVHWDeviceContext*>(hw_device_ctx->data);
    AVVAAPIDeviceContext* vactx = reinterpret_cast<AVVAAPIDeviceContext*>(hwctx->hwctx);
    vactx->display = va_display;

    int err = av_hwdevice_ctx_init(hw_device_ctx);
    if (err < 0) {
        av_buffer_unref(&hw_device_ctx);
        throw std::runtime_error("av_hwdevice_ctx_init failed with error: " +
                                 std::string(av_err2string(err)));
    }
    encoder_context_->hw_device_ctx = hw_device_ctx;

    AVBufferRef* hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ctx) {
        av_buffer_unref(&hw_device_ctx);
        throw std::runtime_error("Failed to allocate hardware frames context");
    }
    frames_ctx = reinterpret_cast<AVHWFramesContext*>(hw_frames_ctx->data);
    frames_ctx->format = AV_PIX_FMT_VAAPI;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width = video_info.width;
    frames_ctx->height = video_info.height;
    frames_ctx->initial_pool_size = 20;

    err = av_hwframe_ctx_init(hw_frames_ctx);
    if (err < 0) {
        // TODO: simplify
        char err_buf[AV_ERROR_MAX_STRING_SIZE]{};
        std::string msg("Failed to initialize VAAPI HW frames context: ");
        msg.append(std::string(av_err2string(err)));
        throw std::runtime_error(msg);
    }

    encoder_context_->hw_frames_ctx = hw_frames_ctx;
    if (!encoder_context_->hw_frames_ctx)
        throw std::runtime_error("Failed to set encoder's HW frames context");
}

AVFrame* XpuEncoder::get_av_frame_from_va_frame(uint32_t va_surface_id) {
    AVFrame* hw_frame = av_frame_alloc();
    if (!hw_frame) {
        throw std::runtime_error("Failed to allocate AVFrame.");
    }

    hw_frame->buf[0] = av_buffer_pool_get(frames_ctx->pool);
    if (!hw_frame->buf[0]) {
        av_frame_free(&hw_frame);
        throw std::runtime_error("Failed to allocate buffer for AVFrame.");
    }

    hw_frame->width = encoder_context_->width;
    hw_frame->height = encoder_context_->height;
    hw_frame->data[3] = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(va_surface_id));
    hw_frame->format = static_cast<AVPixelFormat>(AV_PIX_FMT_VAAPI);
    hw_frame->pts = AV_NOPTS_VALUE;

    return hw_frame;
}

void XpuEncoder::encode_write(AVFrame* frame) {
    int err = 0;
    AVPacket* enc_pkt = av_packet_alloc();

    if (!enc_pkt) {
        throw std::runtime_error("Failed to allocate AV packet.");
    }

    /*
        frame can be nullptr during flush()
        pts operation is only attempted on valid frames
    */
    if (frame != nullptr) {
        frame->pts = frame_id_++; // Increment frame ID for each frame
    }
    err = avcodec_send_frame(encoder_context_, frame);
    if (err < 0) {
        throw std::runtime_error("avcodec_send_frame failed with err code: " +
                                 std::string(av_err2string(err)));
    }

    while (true) {
        err = avcodec_receive_packet(encoder_context_, enc_pkt);
        if (err) {
            break;
        }

        av_packet_rescale_ts(enc_pkt,
                             encoder_context_->time_base,         // Source time base
                             output_ctx_->streams[0]->time_base); // Destination time base

        enc_pkt->stream_index = 0;
        err = av_interleaved_write_frame(output_ctx_, enc_pkt);
        if (err < 0) {
            throw std::runtime_error("Failed to export VASurface. Err code: " +
                                     std::string(av_err2string(err)));
        }
        av_packet_unref(enc_pkt);
    }
    av_packet_free(&enc_pkt);
}

static void av_frame_deleter(AVFrame* frame) {
    av_frame_free(&frame);
}

void XpuEncoder::write(uint32_t va_surface_id) {
    std::unique_ptr<AVFrame, decltype(&av_frame_deleter)> hw_frame(
        get_av_frame_from_va_frame(va_surface_id), av_frame_deleter);
    encode_write(hw_frame.get());
}

void XpuEncoder::flush() {
    encode_write(nullptr);
    auto err = av_write_trailer(output_ctx_);
    if (err < 0) {
        throw std::runtime_error("av_write_trailer failed. Err code: " +
                                 std::string(av_err2string(err)));
    }
}

void pybind11_submodule_videowriter(py::module_& parent_module) {
    py::module_ m = parent_module.def_submodule("videowriter");
    py::class_<VideoInfo>(m, "VideoInfo")
        .def(py::init<>())
        .def_readwrite("codec_name", &VideoInfo::codec_name)
        .def_readwrite("width", &VideoInfo::width)
        .def_readwrite("height", &VideoInfo::height)
        .def_readwrite("bitrate", &VideoInfo::bitrate)
        .def_readwrite("framerate", &VideoInfo::framerate);

    py::class_<VideoInfo::Framerate>(m, "Framerate")
        .def(py::init<>())
        .def_readwrite("num", &VideoInfo::Framerate::num)
        .def_readwrite("den", &VideoInfo::Framerate::den);

    py::class_<XpuEncoder>(m, "XpuEncoder")
        .def(py::init<const std::string&, const std::string&, const VideoInfo&>())
        .def(py::init<const std::string&, const VADisplay&, const VideoInfo&>())
        .def("write", &XpuEncoder::write)
        .def("flush", &XpuEncoder::flush)
        .def("close", &XpuEncoder::close);
}
