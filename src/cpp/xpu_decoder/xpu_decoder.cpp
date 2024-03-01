#include <optional>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include "l0_context.hpp"
#include "context_manager.hpp"
#include "vaapi_context.hpp"
#include "vaapi_frame_converter.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // everything needed for embedding
#include "dlpack/dlpack.h"

#include <fcntl.h>
#include <level_zero/ze_api.h>

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

class OnExit : public std::function<void()> {
  public:
    OnExit(const OnExit&) = delete;
    OnExit& operator=(const OnExit& other) = delete;

    template <class T>
    OnExit(T&& arg) : std::function<void()>(std::forward<T>(arg)){};

    ~OnExit() { operator()(); };
};

// SharedFrame owns VASurface and USM memory block, and destroys them when SharedFrame is destroyed
struct SharedFrame {
    using FrameDelFn = std::function<void(VaApiFrame*)>;

    uint64_t offset = 0;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    VaApiFrame* va_frame = nullptr;
    FrameDelFn frame_deleter; // FIXME: better solution

    ze_context_handle_t ze_context{};
    void* usm_ptr = nullptr;

    SharedFrame() = default;
    SharedFrame(VaApiFrame* frame, FrameDelFn del_fn);
    SharedFrame(const SharedFrame&) = delete;
    SharedFrame& operator=(const SharedFrame&) = delete;
    SharedFrame(SharedFrame&& other);
    SharedFrame& operator=(SharedFrame&& other);

    ~SharedFrame();

    VASurfaceID va_surface() const {
        assert(va_frame);
        return va_frame->desc.va_surface_id;
    }
};

SharedFrame::SharedFrame(VaApiFrame* frame, FrameDelFn del_fn)
    : va_frame(frame), frame_deleter(del_fn) {
}

SharedFrame::SharedFrame(SharedFrame&& other) {
    *this = std::move(other);
}

SharedFrame& SharedFrame::operator=(SharedFrame&& other) {
    if (this == &other)
        return *this;

    shape = std::move(other.shape);
    strides = std::move(other.strides);
    offset = other.offset;

    std::swap(usm_ptr, other.usm_ptr);
    std::swap(ze_context, other.ze_context);
    std::swap(va_frame, other.va_frame);
    std::swap(frame_deleter, other.frame_deleter);

    return *this;
}

SharedFrame::~SharedFrame() {
    // Cleanup code for usm_ptr
    // std::cout << "[" << this << "] Trying to free usm_ptr: " << usm_ptr << " ze_context: " <<
    // ze_context << std::endl;
    if (usm_ptr) {
        ze_result_t res = zeMemFree(ze_context, usm_ptr);
        if (res != ZE_RESULT_SUCCESS)
            std::cout << "[ERROR]: Failed to free USM pointer(" << usm_ptr
                      << "), code: " << std::hex << res << std::dec << std::endl;
        usm_ptr = nullptr;
    }

    // Cleanup code for va_frame or va_surface
    if (va_frame) {
        assert(frame_deleter);
        frame_deleter(va_frame);
    }
}

struct SharedTensor : DLManagedTensor {
    SharedFrame frame;

    SharedTensor(SharedFrame&& f);
    ~SharedTensor() = default;
};

SharedTensor::SharedTensor(SharedFrame&& f) : frame(std::move(f)) {
    dl_tensor.data = frame.usm_ptr;
    dl_tensor.dtype.code = kDLUInt;
    dl_tensor.dtype.bits = 8;
    dl_tensor.dtype.lanes = 1;
    dl_tensor.shape = frame.shape.data();
    dl_tensor.ndim = frame.shape.size();
    dl_tensor.device.device_type = kDLOneAPI;
    dl_tensor.device.device_id = 0; // use first device
    dl_tensor.strides = frame.strides.data();
    dl_tensor.byte_offset = frame.offset;

    // Set the deleter for DLManagedTensor
    deleter = [](DLManagedTensor* self) { delete static_cast<SharedTensor*>(self); };
}

enum class MemoryFormat { unknown = 0, pt_planar_rgbp, pt_packed_rgba, ov_planar_nv12 };

class XpuDecoder {
  public:
    XpuDecoder(const std::string& filename);
    XpuDecoder(const std::string& filename, const std::string& device_name);
    XpuDecoder(const std::string& filename, VADisplay va_display);
    XpuDecoder(const std::string& filename, VaApiContextPtr va_context, L0ContextPtr l0_context);
    XpuDecoder(const XpuDecoder&) = delete;
    XpuDecoder(XpuDecoder&&);
    XpuDecoder& operator=(const XpuDecoder&) = delete;
    XpuDecoder& operator=(XpuDecoder&&) = delete;
    ~XpuDecoder();

    XpuDecoder get_iterator();
    std::pair<py::object, std::unique_ptr<VaApiFrame>> get_next_frame();
    void set_memory_format(MemoryFormat format);
    void set_output_resolution(int width, int height);
    VADisplay get_va_device() const { return va_context_->DisplayRaw(); }
    void set_frame_pool_params(uint32_t pool_size) { preproc_->set_pool_size(pool_size); }
    void set_loop_mode(bool loop_video) { loop_mode_ = loop_video; }
    void set_output_original_nv12(bool output_original_nv12) {
        output_original_nv12_ = output_original_nv12;
    }
    void set_batch_size(size_t batch_size) { batch_size_ = batch_size; }

  private:
    size_t batch_size_ = 1;
    size_t processed_frames_counter_ = 0;
    AVFormatContext* input_ctx_ = nullptr;
    AVCodecContext* decoder_ctx_ = nullptr;
    AVCodecParameters* codecpar_ = nullptr;
    int video_stream_ = AVERROR_STREAM_NOT_FOUND;

    ze_context_handle_t ze_context_ = nullptr;
    ze_device_handle_t ze_device_ = nullptr;

    L0ContextPtr l0_context_;
    VaApiContextPtr va_context_;
    std::shared_ptr<VaApiFrameConverter> preproc_;

    MemoryFormat memory_format_ = MemoryFormat::unknown;

    bool loop_mode_ = false;
    bool output_original_nv12_ = false;
    bool resource_owner_ = false; // true means that current instance of the class owns resources
    uint32_t ze_device_id_ = 0;

    void init_ffmpeg(const std::string& filename);
    AVCodecContext* configure_decoder_context(const AVCodec* codec, AVCodecParameters* codecpar);
    void vaapi_to_usm(SharedFrame& shared_frame);
    void workaround_memory();

    bool get_next_av_frame(AVFrame* avframe);
    /* process_frame now returns a tuple of tensor & vaframe which changes the __next__ */
    std::pair<py::object, std::unique_ptr<VaApiFrame>> process_frame(AVFrame* avframe);

    py::object process_frame_to_openvino(VaApiFrame& vaframe);
    py::object process_frame_to_pytorch(VaApiFrame& vaframe);
};

XpuDecoder::XpuDecoder(const std::string& filename)
    : XpuDecoder(filename, ContextManager::get_instance().get_va_context("/dev/dri/renderD128"),
                 ContextManager::get_instance().get_l0_context(0)) {
}

XpuDecoder::XpuDecoder(const std::string& filename, const std::string& device_name)
    : XpuDecoder(filename,
                 ContextManager::get_instance().get_va_context(
                     ContextManager::get_device_path_from_device_name(device_name)),
                 ContextManager::get_instance().get_l0_context(
                     ContextManager::get_ze_device_id(device_name))) {
}

/*
 fixme - need to provide device_id which used for created VADisplay.
 */
XpuDecoder::XpuDecoder(const std::string& filename, VADisplay va_display)
    : XpuDecoder(filename, std::make_shared<VaApiContext>(va_display), 0) {
}

XpuDecoder::XpuDecoder(const std::string& filename, VaApiContextPtr va_context,
                       L0ContextPtr l0_context)
    : va_context_(va_context), resource_owner_(true), l0_context_(l0_context) {

    init_ffmpeg(filename);
    ze_context_ = l0_context_->get_ze_context();
    ze_device_ = l0_context_->get_ze_device();

    // Init preproc with default setting.
    // Set output resolution to match video resolution. (Meaning no preprocessing by default for OV
    // case) preproc can be customized by calling set_ methods TODO maybe rename preproc_set_ ?
    preproc_ = std::make_shared<VaApiFrameConverter>(va_context_);
    set_output_resolution(codecpar_->width, codecpar_->height);
    set_memory_format(MemoryFormat::pt_planar_rgbp);

    // NOTICE: A script should set correct size!
    set_frame_pool_params(128);
}

XpuDecoder::XpuDecoder(XpuDecoder&& other) {
    input_ctx_ = other.input_ctx_;
    decoder_ctx_ = other.decoder_ctx_;
    video_stream_ = other.video_stream_;

    ze_context_ = other.ze_context_;
    ze_device_ = other.ze_device_;

    va_context_ = other.va_context_;

    preproc_ = std::move(other.preproc_);

    memory_format_ = other.memory_format_;

    loop_mode_ = other.loop_mode_;

    // transfer resource ownership
    resource_owner_ = other.resource_owner_;
    other.resource_owner_ = false;
}

XpuDecoder::~XpuDecoder() {
    if (!resource_owner_)
        return;

    avformat_close_input(&input_ctx_);
    avcodec_free_context(&decoder_ctx_);
}

void XpuDecoder::init_ffmpeg(const std::string& filename) {
    const AVCodec* codec = nullptr;
    if (avformat_open_input(&input_ctx_, filename.c_str(), nullptr, nullptr) < 0) {
        // Handle error: file couldn't be opened, or out of memory, etc.
        throw std::runtime_error("Failed to open input file: " + filename);
    }

    if (avformat_find_stream_info(input_ctx_, nullptr) < 0) {
        throw std::runtime_error("Cannot find input stream information.");
    }

    // Now it should be safe to use input_ctx_
    video_stream_ = av_find_best_stream(input_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (video_stream_ < 0) {
        // Handle error: no suitable stream found, or another error occurred
        throw std::runtime_error("Failed to find a suitable video stream.");
    }

    // New code: print the timebase of the video stream
    // AVRational timebase = input_ctx_->streams[video_stream_]->time_base; // This is debug code

    codecpar_ = input_ctx_->streams[video_stream_]->codecpar;

    std::cout << "Opened stream:  " << filename << std::endl;
    std::cout << "    Codec:      " << avcodec_get_name(codecpar_->codec_id) << std::endl;
    std::cout << "    Resolution: " << codecpar_->width << 'x' << codecpar_->height << std::endl;
    std::cout << "    Bitrate:    " << codecpar_->bit_rate << std::endl;
    // std::cout << "    Framerate:  " << av_q2d(codecpar_->framerate);

    decoder_ctx_ = configure_decoder_context(codec, codecpar_);
}

// Configure the AV decoder context
AVCodecContext* XpuDecoder::configure_decoder_context(const AVCodec* codec,
                                                      AVCodecParameters* codecpar) {
    AVCodecContext* decoder_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(decoder_ctx, codecpar);
    decoder_ctx->pix_fmt = AV_PIX_FMT_VAAPI;

    AVBufferRef* hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VAAPI);
    auto* hwctx = reinterpret_cast<AVHWDeviceContext*>(hw_device_ctx->data);
    auto* vactx = static_cast<AVVAAPIDeviceContext*>(hwctx->hwctx);
    vactx->display = va_context_->DisplayRaw();
    av_hwdevice_ctx_init(hw_device_ctx);

    decoder_ctx->hw_device_ctx = hw_device_ctx;

    if (avcodec_open2(decoder_ctx, codec, NULL) < 0)
        throw std::runtime_error("Failed to open codec");

    AVBufferRef* hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    auto frames_ctx = (AVHWFramesContext*)(hw_frames_ctx->data);
    frames_ctx->format = AV_PIX_FMT_VAAPI;
    frames_ctx->sw_format = AV_PIX_FMT_RGB0; // this is the RGB format for VAAPI but this is not
                                             // used, just that we cant run without this line...
    frames_ctx->width = decoder_ctx->coded_width;
    frames_ctx->height = decoder_ctx->coded_height;
    frames_ctx->initial_pool_size = 20;

    int err = 0;
    if ((err = av_hwframe_ctx_init(hw_frames_ctx)) < 0) {
        // TODO: simplify
        char err_buf[AV_ERROR_MAX_STRING_SIZE]{};
        std::string msg("Failed to initialize VAAPI HW frames context: ");
        msg.append(av_make_error_string(err_buf, sizeof err_buf, err));
        throw std::runtime_error(msg);
    }

    decoder_ctx->hw_frames_ctx = hw_frames_ctx;
    if (!decoder_ctx->hw_frames_ctx)
        throw std::runtime_error("Failed to set decoder's HW frames context");

    return decoder_ctx;
}

static void check_va_status(VAStatus sts, const char* msg) {
    if (sts != VA_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " failed: " + std::to_string(sts));
    }
}

static void check_ze_status(ze_result_t sts, const char* msg) {
    if (sts != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " failed: " + std::to_string(sts));
    }
}

static void check_ptr(void* ptr, const char* msg) {
    if (!ptr) {
        throw std::runtime_error(std::string(msg) + " failed: nullptr");
    }
}

py::object dl_tensor_to_pytorch(std::unique_ptr<SharedTensor> managed_tensor) {
    py::object torch = py::module_::import("torch");
    py::object torch_utils = torch.attr("utils");
    py::object dlpack = torch_utils.attr("dlpack");
    py::object from_dlpack = dlpack.attr("from_dlpack");
    return from_dlpack(
        py::capsule(managed_tensor.release(), "dltensor", PyCapsule_Destructor(nullptr)));
}

std::pair<py::object, std::unique_ptr<VaApiFrame>> XpuDecoder::get_next_frame() {
    if (!resource_owner_) {
        throw py::stop_iteration();
    }

    AVFrame* avframe = av_frame_alloc();
    check_ptr(avframe, "av_frame_alloc");
    OnExit avf_deleter([&]() { av_frame_free(&avframe); });

    bool end_of_file = !get_next_av_frame(avframe);
    if (end_of_file && loop_mode_) {
        const int ret = av_seek_frame(input_ctx_, video_stream_, 0, AVSEEK_FLAG_FRAME);
        avcodec_flush_buffers(decoder_ctx_);
        if (ret >= 0)
            end_of_file = !get_next_av_frame(avframe);
        else
            std::cout << "[ERROR]: av_seek_frame to start failed, code: " << ret << std::endl;
    }
    if (end_of_file)
        throw py::stop_iteration();

    return process_frame(avframe);
}

bool XpuDecoder::get_next_av_frame(AVFrame* avframe) {
    AVPacket* avpacket = av_packet_alloc();
    check_ptr(avpacket, "av_packet_alloc");
    OnExit avp_deleter([&]() { av_packet_free(&avpacket); });

    while (true) {
        // read from file
        int ret = av_read_frame(input_ctx_, avpacket);
        bool end_of_file = (ret < 0);

        // skip audio
        if (!end_of_file && avpacket->stream_index != video_stream_) {
            av_packet_unref(avpacket);
            continue;
        }

        // send to decoder
        ret = avcodec_send_packet(decoder_ctx_, end_of_file ? nullptr : avpacket);
        av_packet_unref(avpacket);
        if (ret < 0 && ret != AVERROR_EOF) {
            throw std::runtime_error("Error sending packet to decoder: " + std::to_string(ret));
        }

        // get decoded frame
        ret = avcodec_receive_frame(decoder_ctx_, avframe);
        if (!end_of_file && ret == AVERROR(EAGAIN)) {
            // Decoder needs more data, read more packets
            continue;
        }

        if (ret == 0) {
            // frame is ready
            return true;
        }
        if (ret == AVERROR_EOF) {
            // no more buffered frames in decoder
            return false;
        }
        throw std::runtime_error("Error receiving frame from decoder");
    }
}

struct VaApiFrameWrap final : public VaApiFrame {
    using FrameDelFn = std::function<void(VaApiFrame*)>;

    VaApiFrame* frame = nullptr;
    FrameDelFn frame_deleter;

    VaApiFrameWrap(VaApiFrame* f, FrameDelFn del_fn)
        : VaApiFrame(f->desc.va_display, f->desc.va_surface_id, f->desc.width, f->desc.height,
                     f->desc.format, false),
          frame(f), frame_deleter(del_fn) {}

    ~VaApiFrameWrap() {
        if (frame) {
            assert(frame_deleter);
            frame_deleter(frame);
        }
    }

    // No Copy
    VaApiFrameWrap(const VaApiFrameWrap&) = delete;
    VaApiFrameWrap& operator=(const VaApiFrameWrap&) = delete;
};

/*
 * TODO: This changes the behaviour of __next__ to return tuple of tensor and VaApiFrame
 *       Need to figure out a better design
 */
std::pair<py::object, std::unique_ptr<VaApiFrame>> XpuDecoder::process_frame(AVFrame* avframe) {
    auto vasurface = static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>(avframe->data[3]));
    // false => doesn't own surface
    VaApiFrame vaframe(va_context_->DisplayRaw(), vasurface, avframe->width, avframe->height,
                       VA_FOURCC_NV12, false);
    std::unique_ptr<VaApiFrame> nv12_copy = nullptr;
    if (output_original_nv12_)
        auto nv12_copy = VaApiFrame::copy_from(vaframe);
    py::object py_object;
    if (memory_format_ == MemoryFormat::ov_planar_nv12) {
        py_object = process_frame_to_openvino(vaframe);
    } else {
        py_object = process_frame_to_pytorch(vaframe);
    }

    return {py_object, std::move(nv12_copy)};
}

py::object XpuDecoder::process_frame_to_openvino(VaApiFrame& vaframe) {
    VaApiFrame* processed_frame = preproc_->convert(vaframe);
    // Create wrapped pointer and cast it to base
    std::unique_ptr<VaApiFrame> wrapped(new VaApiFrameWrap(
        processed_frame, [pp = this->preproc_](VaApiFrame* f) { pp->release_frame(f); }));

    return pybind11::cast(std::move(wrapped));
}

py::object XpuDecoder::process_frame_to_pytorch(VaApiFrame& vaframe) {
    auto processed_frame = preproc_->convert(vaframe);

    SharedFrame shared_frame(processed_frame,
                             [pp = this->preproc_](VaApiFrame* f) { pp->release_frame(f); });

    vaapi_to_usm(shared_frame);
    workaround_memory();

    // this call transfers ownership from "shared_frame" to "shared_tensor"
    auto shared_tensor = std::make_unique<SharedTensor>(std::move(shared_frame));

    // transfers ownership form "shared_tensor" to "py_object"
    py::object py_object = dl_tensor_to_pytorch(std::move(shared_tensor));

    return std::move(py_object);
}

void XpuDecoder::vaapi_to_usm(SharedFrame& shared_frame) {
    // Warning: In this section, we perform surface synchronization only once per batch.
    // However, we return tensors for each individual frame.
    // This could potentially lead to a situation where a user runs a tensor operation on
    // non-synchronized memory, which may result in undefined behavior. As a future improvement,
    // TODO consider rewriting the decoder so it returns a tensor for the entire batch, rather than
    // on a per-frame basis.
    if (++processed_frames_counter_ >= batch_size_) {
        VASurfaceStatus surface_sts = VASurfaceRendering;
        // Warning: Here we use polling status loop because of VaSyncSurface on batch 512 issue.
        // TODO: Change it to VaSyncSurface when issue fixed
        while (surface_sts != VASurfaceReady) {
            auto sts = vaQuerySurfaceStatus(va_context_->DisplayRaw(), shared_frame.va_surface(),
                                            &surface_sts);
            check_va_status(sts, "vaQuerySurfaceStatus");
        }
        processed_frames_counter_ = 0;
    }

    VADRMPRIMESurfaceDescriptor prime_desc;
    auto sts =
        vaExportSurfaceHandle(va_context_->DisplayRaw(), shared_frame.va_surface(),
                              VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                              VA_EXPORT_SURFACE_READ_WRITE /*VA_EXPORT_SURFACE_READ_ONLY |
                                                              VA_EXPORT_SURFACE_COMPOSED_LAYERS*/
                              ,
                              &prime_desc);
    check_va_status(sts, "vaExportSurfaceHandle");

    OnExit dma_fd_deleter([&]() { close(prime_desc.objects->fd); });

    uint32_t dma_size = prime_desc.objects->size;

    ze_external_memory_import_fd_t import_fd{};
    import_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    import_fd.pNext = nullptr;
    import_fd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    import_fd.fd = prime_desc.objects->fd;

    ze_device_mem_alloc_desc_t alloc_desc{};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.pNext = &import_fd;

    ze_result_t ze_res =
        zeMemAllocDevice(ze_context_, &alloc_desc, dma_size, 1, ze_device_, &shared_frame.usm_ptr);
    check_ze_status(ze_res, "Failed to convert DMA to USM pointer:");

    shared_frame.offset = prime_desc.layers->offset[0];
    shared_frame.ze_context = ze_context_;

    if (memory_format_ == MemoryFormat::pt_planar_rgbp) {
        shared_frame.shape = {3, prime_desc.height, prime_desc.width};
        shared_frame.strides = {prime_desc.layers->pitch[0] * prime_desc.height,
                                prime_desc.layers->pitch[0], 1};
    } else {
        // default, RGB packed
        shared_frame.shape = {prime_desc.height, prime_desc.width, 4};
        shared_frame.strides = {prime_desc.layers->pitch[0], 4, 1};
    }
}

void XpuDecoder::workaround_memory() {
    static ze_device_mem_alloc_desc_t mem_desc = {.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                                  .pNext = nullptr,
                                                  .flags = 0,
                                                  .ordinal = 0};

    void* ptr = nullptr;
    auto ret = zeMemAllocDevice(ze_context_, &mem_desc, 1, 1, ze_device_, &ptr);
    if (ret != ZE_RESULT_SUCCESS) {
        std::cout << "[ERROR]: zeMemAllocDevice() failed: " << ret << std::endl;
        return;
    }

    ret = zeMemFree(ze_context_, ptr);
    if (ret != ZE_RESULT_SUCCESS)
        std::cout << "[ERROR]: zeMemFree() failed: " << ret << std::endl;
}

XpuDecoder XpuDecoder::get_iterator() {
    return std::move(*this);
}

void XpuDecoder::set_memory_format(MemoryFormat format) {
    assert(format != MemoryFormat::unknown);
    assert(preproc_);

    uint32_t va_color_fmt = 0;
    if (format == MemoryFormat::pt_packed_rgba)
        va_color_fmt = VA_FOURCC_RGBA;
    else if (format == MemoryFormat::pt_planar_rgbp)
        va_color_fmt = VA_FOURCC_RGBP;
    else if (format == MemoryFormat::ov_planar_nv12)
        va_color_fmt = VA_FOURCC_NV12;
    else
        throw std::runtime_error("unsupported or invalid memory format");

    preproc_->set_output_color_format(va_color_fmt);
    memory_format_ = format;
}

void XpuDecoder::set_output_resolution(int width, int height) {
    preproc_->set_ouput_resolution(width, height);
}

// Video reader uses simplified approach to create iterator - returns itself as an iterator. This
// leads to all kinds of issues with resource sharing when we use it in Python. Ideally, we have to
// use separate iterator class. For now, we just make video reader unique, i.e. allow single
// instance of the class. That works fine in most cases.
//
// Usage example
//     import libvideoreader
//     video = libvideoreader.XpuDecoder("filename.mp4")
//     video.set_memory_format(XpuMemoryFormat.torch_contiguous_format)
//     video.set_output_resolution(224, 224)
//     next_frame = next(video)
//     print(next_frame)
//
// Another example
//     video = libvideoreader.XpuDecoder("filename.mp4")
//     for next_frame in video:
//         print(next_frame)
//
// It is possible to mix both usages
//     video = libvideoreader.XpuDecoder("filename.mp4")
//     first_frame = next(video)
//     print(first_frame)
//
//     for next_frame in video: #from second frame till the end of the stream
//         print(next_frame)
//
void pybind11_submodule_videoreader(py::module_& parent_module) {
    py::module_ m = parent_module.def_submodule("videoreader");
    py::enum_<MemoryFormat>(m, "XpuMemoryFormat")
        .value("torch_contiguous_format", MemoryFormat::pt_planar_rgbp)
        .value("torch_channels_last", MemoryFormat::pt_packed_rgba)
        .value("openvino_planar", MemoryFormat::ov_planar_nv12);

    py::class_<VaApiFrame, std::unique_ptr<VaApiFrame>>(m, "VaFrame")
        .def_property_readonly("width", [](const VaApiFrame& f) { return f.desc.width; })
        .def_property_readonly("height", [](const VaApiFrame& f) { return f.desc.height; })
        .def_property_readonly("va_display", [](const VaApiFrame& f) { return f.desc.va_display; })
        .def_property_readonly("va_surface_id",
                               [](const VaApiFrame& f) { return f.desc.va_surface_id; })
        .def("__repr__", [](const VaApiFrame& a) {
            std::ostringstream oss;
            oss << "<libvideoreader.VaFrame surface_id=" << a.desc.va_surface_id
                << " w=" << a.desc.width << " h=" << a.desc.height << ">";
            return oss.str();
        });

    py::class_<XpuDecoder>(m, "XpuDecoder")
        .def(py::init<const std::string&>())
        .def(py::init<const std::string&, const std::string&>())
        .def(py::init<const std::string&, VADisplay>())
        .def("__iter__", &XpuDecoder::get_iterator)
        .def("__next__", &XpuDecoder::get_next_frame, py::return_value_policy::take_ownership)
        .def("set_memory_format", &XpuDecoder::set_memory_format)
        .def("set_output_resolution", &XpuDecoder::set_output_resolution)
        .def("get_va_device", &XpuDecoder::get_va_device)
        .def("set_frame_pool_params", &XpuDecoder::set_frame_pool_params)
        .def("set_loop_mode", &XpuDecoder::set_loop_mode)
        .def("set_output_original_nv12", &XpuDecoder::set_output_original_nv12)
        .def("set_batch_size", &XpuDecoder::set_batch_size);
}
