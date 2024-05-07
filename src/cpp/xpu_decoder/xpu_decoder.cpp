#include <optional>
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <memory>
#include <optional>
#include <map>
#include "l0_context.hpp"
#include "l0_utils.hpp"
#include "usm_frame.hpp"
#include "context_manager.hpp"
#include "vaapi_context.hpp"
#include "vaapi_utils.hpp"
#include "vaapi_frame_converter.hpp"
#include "logger.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // everything needed for embedding
#include "dlpack/dlpack.h"
#include "visual_ai/memory_format.hpp"
#include "scope_guard.hpp"

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

struct SharedTensor : DLManagedTensor {
    std::shared_ptr<UsmFrame> usm_frame;

    SharedTensor(std::shared_ptr<UsmFrame> usm_f) : usm_frame(std::move(usm_f)) {
        dl_tensor.data = usm_frame->usm_ptr;
        dl_tensor.dtype.code = kDLUInt;
        dl_tensor.dtype.bits = 8;
        dl_tensor.dtype.lanes = 1;
        dl_tensor.shape = usm_frame->shape.data();
        dl_tensor.ndim = usm_frame->shape.size();
        dl_tensor.device.device_type = kDLOneAPI;
        dl_tensor.device.device_id = 0; // use first device
        dl_tensor.strides = usm_frame->strides.data();
        dl_tensor.byte_offset = usm_frame->offset;

        // Set the deleter for DLManagedTensor
        deleter = [](DLManagedTensor* self) { delete static_cast<SharedTensor*>(self); };
    }

    ~SharedTensor() {
        // Usm frame might be cached, so need to free parent frame explicitly
        // See vaapi_to_usm_cached for more details.
        // This can be changed/improved by making actual USM pointer a shared object.
        usm_frame->parent_frame.reset();
    }

    // NoCopy
    SharedTensor(const SharedTensor&) = delete;
    SharedTensor& operator=(const SharedTensor&) = delete;
};

// Decode & process result: 0 - processed frame, 1 - original frame, 2 - dynamically allocated
using DecProcResult = std::tuple<std::unique_ptr<VaApiFrame>, std::unique_ptr<VaApiFrame>, bool>;

class XpuDecoder {
  public:
    XpuDecoder(const std::string& filename);
    XpuDecoder(const std::string& filename, const std::string& device_name);
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
    void set_async_depth(int depth);
    const VaDpyWrapper& get_va_device() const { return va_context_->display(); }
    void set_frame_pool_params(uint32_t pool_size) { preproc_->set_pool_size(pool_size); }
    void set_loop_mode(bool loop_video) { loop_mode_ = loop_video; }
    void set_output_original_nv12(bool output_original_nv12) {
        output_original_nv12_ = output_original_nv12;
    }
    void set_batch_size(size_t batch_size) { batch_size_ = batch_size; }
    std::tuple<uint32_t, uint32_t> get_original_size() {
        if (codecpar_)
            return {codecpar_->width, codecpar_->height};
        return {0, 0};
    }

  private:
    Logger logger_;
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

    int async_depth_ = 0; // default to synchronous decoding

    std::thread decode_thread_;

    std::mutex decode_mutex_;
    std::atomic<bool> stop_decoding_ = false;

    std::condition_variable frame_ready_{}; // wakes up python thread
    std::condition_variable need_frame_{};  // wakes up decoder thread

    std::queue<std::optional<DecProcResult>> frame_queue_;

    // Maps VASurfaces to USM pointers to avoid unneccesary VA-to-USM conversions
    std::map<VASurfaceID, std::shared_ptr<UsmFrame>> usm_ptr_cache_;

    void lazy_init();
    void decode_routine();

    uint32_t ze_device_id_ = 0;

    void init_ffmpeg(const std::string& filename);
    AVCodecContext* configure_decoder_context(const AVCodec* codec, AVCodecParameters* codecpar);

    std::shared_ptr<UsmFrame> vaapi_to_usm(std::unique_ptr<VaApiFrame> vaframe, bool sync_frame);
    std::shared_ptr<UsmFrame> vaapi_to_usm_cached(std::unique_ptr<VaApiFrame> vaframe,
                                                  bool sync_frame);

    std::optional<DecProcResult> decode_and_process_next_frame();
    py::object export_frame_to_python(std::unique_ptr<VaApiFrame> vaframe, bool allow_cache);

    bool get_next_av_frame(AVFrame* avframe);

    bool decode_thread_active() const { return decode_thread_.joinable(); }
};

void XpuDecoder::lazy_init() {
    if (async_depth_ == 0 || decode_thread_active()) {
        return;
    }

    decode_thread_ = std::thread(&XpuDecoder::decode_routine, this);
    logger_.debug("Decode thread started, depth={}", async_depth_);
}

void XpuDecoder::decode_routine() {
    while (!stop_decoding_) {
        // Locked section
        {
            std::unique_lock<std::mutex> lock(decode_mutex_);
            if (frame_queue_.size() >= async_depth_)
                // wait for free slot in queue
                need_frame_.wait(
                    lock, [&] { return frame_queue_.size() < async_depth_ || stop_decoding_; });
        }

        auto opt_result = decode_and_process_next_frame();
        // Locked section
        {
            std::lock_guard<std::mutex> lock(decode_mutex_);
            frame_queue_.push(std::move(opt_result));
        }
        frame_ready_.notify_one();
    }
}

XpuDecoder::XpuDecoder(const std::string& filename)
    : XpuDecoder(filename,
                 std::make_shared<VaApiContext>(
                     ContextManager::get_instance().get_va_display("/dev/dri/renderD128")),
                 ContextManager::get_instance().get_l0_context(0)) {
}

XpuDecoder::XpuDecoder(const std::string& filename, const std::string& device_name)
    : XpuDecoder(filename,
                 std::make_shared<VaApiContext>(ContextManager::get_instance().get_va_display(
                     ContextManager::get_device_path_from_device_name(device_name))),
                 ContextManager::get_instance().get_l0_context(
                     ContextManager::get_ze_device_id(device_name))) {
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

    // Note: It's script responsibility to set required pool size
    set_frame_pool_params(128);
}

XpuDecoder::XpuDecoder(XpuDecoder&& other) {
    if (other.decode_thread_active()) {
        throw std::runtime_error("could not move XpuDecoder instance during active decoding");
    }

    input_ctx_ = other.input_ctx_;
    decoder_ctx_ = other.decoder_ctx_;
    video_stream_ = other.video_stream_;

    ze_context_ = other.ze_context_;
    ze_device_ = other.ze_device_;

    va_context_ = other.va_context_;

    preproc_ = std::move(other.preproc_);

    memory_format_ = other.memory_format_;

    loop_mode_ = other.loop_mode_;

    async_depth_ = other.async_depth_;

    // transfer resource ownership
    resource_owner_ = other.resource_owner_;
    other.resource_owner_ = false;
}

XpuDecoder::~XpuDecoder() {
    if (!resource_owner_)
        return;

    stop_decoding_ = true;
    need_frame_.notify_one();
    if (decode_thread_.joinable()) {
        decode_thread_.join();
    }

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

    logger_.info("Opened stream:\n"
                 "  File:       {}\n"
                 "  Codec:      {}\n"
                 "  Resolution: {}x{}\n"
                 "  Bitrate:    {}",
                 filename, avcodec_get_name(codecpar_->codec_id), codecpar_->width,
                 codecpar_->height, codecpar_->bit_rate);

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
    vactx->display = va_context_->display_native();
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

    lazy_init(); // sets "async_depth_" to zero for OV case

    std::optional<DecProcResult> opt_result;
    if (async_depth_ == 0) {
        // synchronous decoding
        opt_result = decode_and_process_next_frame();
    } else {
        // asynchronous decoding
        // Locked section
        {
            std::unique_lock<std::mutex> guard(decode_mutex_);
            frame_ready_.wait(guard, [&] { return !frame_queue_.empty(); });
            opt_result = std::move(frame_queue_.front());
            frame_queue_.pop();
        }

        need_frame_.notify_one();
    }

    // If result is empty - EOF is reached
    if (!opt_result)
        throw py::stop_iteration();

    // Unpack decode and processing results from optional
    auto [processed, original, dyn_alloc] = std::move(opt_result.value());

    // Export processed frame to Python.
    // Skip cache if frame is dynamically allocated (i.e. not from pool)
    py::object py_object = export_frame_to_python(std::move(processed), !dyn_alloc);

    return {std::move(py_object), std::move(original)};
}

bool XpuDecoder::get_next_av_frame(AVFrame* avframe) {
    AVPacket* avpacket = av_packet_alloc();
    check_ptr(avpacket, "av_packet_alloc");
    auto avp_deleter = make_scope_guard([&]() { av_packet_free(&avpacket); });

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

std::optional<DecProcResult> XpuDecoder::decode_and_process_next_frame() {
    AVFrame* avframe = av_frame_alloc();
    check_ptr(avframe, "av_frame_alloc");
    auto avf_deleter = make_scope_guard([&]() { av_frame_free(&avframe); });

    bool end_of_file = !get_next_av_frame(avframe);
    if (end_of_file && loop_mode_) {
        const int ret = av_seek_frame(input_ctx_, video_stream_, 0, AVSEEK_FLAG_FRAME);
        avcodec_flush_buffers(decoder_ctx_);
        if (ret >= 0)
            end_of_file = !get_next_av_frame(avframe);
        else
            logger_.error("av_seek_frame to start failed, code: {}", ret);
    }
    // Return empty optional in case of EOF
    if (end_of_file)
        return {};

    // Frame processing
    auto vasurface = static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>(avframe->data[3]));

    // false => doesn't own surface
    VaApiFrame vaframe(va_context_->display_native(), vasurface, avframe->width, avframe->height,
                       VA_FOURCC_NV12, false);

    // Full frame
    VARectangle region = {.x = int16_t(0),
                          .y = int16_t(0),
                          .width = vaframe.desc.width,
                          .height = vaframe.desc.height};
    auto [processed_frame, dyn_alloc] = preproc_->convert_ex(vaframe, region, false);

    std::unique_ptr<VaApiFrame> nv12_copy;
    if (output_original_nv12_)
        nv12_copy = VaApiFrame::copy_from(vaframe);

    return DecProcResult{std::move(processed_frame), std::move(nv12_copy), dyn_alloc};
}

py::object XpuDecoder::export_frame_to_python(std::unique_ptr<VaApiFrame> vaframe,
                                              bool allow_cache) {
    if (memory_format_ == MemoryFormat::ov_planar_nv12) {
        return pybind11::cast(std::move(vaframe));
    }

    //
    // PyTorch + IPEX w/USM memory
    //

    /*
     * WARNING: Surface synchronization is done only ONCE per batch.
     * However, we return tensors for each individual frame.
     * This could potentially lead to a situation where a user runs a tensor operation on
     * non-synchronized memory, which may result in undefined behavior.
     * TODO:
     * As a future improvement, consider rewriting the decoder so it returns a tensor
     * for the entire batch, rather than on a per-frame basis.
     */
    const bool sync_surface = ++processed_frames_counter_ >= batch_size_;
    if (sync_surface)
        processed_frames_counter_ = 0;

    // transfers ownership of vaframe to usm_frame (i.e. vaframe is invalid after this call)
    std::shared_ptr<UsmFrame> usm_frame;
    if (allow_cache)
        usm_frame = vaapi_to_usm_cached(std::move(vaframe), sync_surface);
    else
        usm_frame = vaapi_to_usm(std::move(vaframe), sync_surface);

    // this call transfers ownership from "shared_frame" to "shared_tensor"
    auto shared_tensor = std::make_unique<SharedTensor>(std::move(usm_frame));

    // transfers ownership form "shared_tensor" to "py_object"
    py::object py_object = dl_tensor_to_pytorch(std::move(shared_tensor));

    return std::move(py_object);
}

// TODO: should be common function and used within FrameTransform & XpuDecoder
std::shared_ptr<UsmFrame> XpuDecoder::vaapi_to_usm(std::unique_ptr<VaApiFrame> vaframe,
                                                   bool sync_frame) {
    auto va_dpy = vaframe->desc.va_display;
    auto va_surface = vaframe->desc.va_surface_id;

    VADRMPRIMESurfaceDescriptor prime_desc;
    VAStatus sts =
        vaExportSurfaceHandle(va_dpy, va_surface, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                              VA_EXPORT_SURFACE_READ_WRITE, /*VA_EXPORT_SURFACE_READ_ONLY |
                                                               VA_EXPORT_SURFACE_COMPOSED_LAYERS*/
                              &prime_desc);
    if (sts != VA_STATUS_SUCCESS)
        throw std::runtime_error(std::string("vaExportSurfaceHandle") +
                                 " failed: " + std::to_string(sts));

    auto dma_fd_deleter = make_scope_guard([fd = prime_desc.objects->fd] { close(fd); });

    const uint32_t dma_size = prime_desc.objects->size;

    ze_external_memory_import_fd_t import_fd{};
    import_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    import_fd.pNext = nullptr;
    import_fd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    import_fd.fd = prime_desc.objects->fd;

    ze_device_mem_alloc_desc_t alloc_desc{};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.pNext = &import_fd;

    void* usm_ptr;
    ze_result_t ze_res = zeMemAllocDevice(l0_context_->get_ze_context(), &alloc_desc, dma_size, 1,
                                          l0_context_->get_ze_device(), &usm_ptr);
    if (sts != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to convert DMA to USM pointer: " + std::to_string(sts));
    }

    // Call sync after export/import to delay this blocking call as much as possible.
    // Potentially it can be called even outside this function,
    // as closer as possible to actual return of pyobject
    if (sync_frame)
        vaframe->sync();

    auto usm_frame = std::make_shared<UsmFrame>(usm_ptr, l0_context_, std::move(vaframe));

    usm_frame->offset = prime_desc.layers->offset[0];

    if (memory_format_ == MemoryFormat::pt_planar_rgbp) {
        usm_frame->shape = {3, prime_desc.height, prime_desc.width};
        usm_frame->strides = {prime_desc.layers->pitch[0] * prime_desc.height,
                              prime_desc.layers->pitch[0], 1};
    } else {
        // default, RGB packed
        usm_frame->shape = {prime_desc.height, prime_desc.width, 4};
        usm_frame->strides = {prime_desc.layers->pitch[0], 4, 1};
    }

    l0_workaround_memory(*l0_context_);

    return usm_frame;
}

std::shared_ptr<UsmFrame> XpuDecoder::vaapi_to_usm_cached(std::unique_ptr<VaApiFrame> vaframe,
                                                          bool sync_frame) {
    const auto surface_id = vaframe->desc.va_surface_id;
    const auto it = usm_ptr_cache_.find(surface_id);
    if (it != usm_ptr_cache_.end()) {
        // Hit! Assign parent frame, sync if needed and return
        auto usm_frame = it->second;
        if (sync_frame)
            vaframe->sync();
        usm_frame->parent_frame = std::move(vaframe);
        return usm_frame;
    }

    auto usm_frame = vaapi_to_usm(std::move(vaframe), sync_frame);
    // Add cache entry
    usm_ptr_cache_.insert({surface_id, usm_frame});
    return usm_frame;
}

XpuDecoder XpuDecoder::get_iterator() {
    return std::move(*this);
}

void XpuDecoder::set_memory_format(MemoryFormat format) {
    if (format == MemoryFormat::unknown)
        throw std::invalid_argument("format cannot be set to unknown");

    assert(preproc_);

    if (decode_thread_active()) {
        throw std::runtime_error("coud not change memory format because decoder is running");
    }

    const uint32_t va_color_fmt = memory_format_to_fourcc(format);

    preproc_->set_output_color_format(va_color_fmt);
    memory_format_ = format;
}

void XpuDecoder::set_output_resolution(int width, int height) {
    if (decode_thread_active()) {
        throw std::runtime_error("coud not change output resolution because decoder is running");
    }

    preproc_->set_ouput_resolution(width, height);
}

void XpuDecoder::set_async_depth(int depth) {
    if (decode_thread_active()) {
        throw std::runtime_error("coud not change async depth because decoder is running");
    }

    // sanity check
    if (depth < 0 || depth > 1024) {
        throw std::runtime_error("wrong async depth");
    }

    async_depth_ = depth;
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

    py::class_<VaDpyWrapper>(m, "VaDpyWrapper")
        .def("native", &VaDpyWrapper::native)
        .def("__repr__", [](const VaDpyWrapper& self) {
            std::stringstream ss;
            ss << "<libvideoreader.VaDisplayWrapper native=" << std::hex << self.native()
               << std::dec << '>';
            return ss.str();
        });
    py::class_<XpuDecoder>(m, "XpuDecoder")
        .def(py::init<const std::string&>())
        .def(py::init<const std::string&, const std::string&>())
        .def("__iter__", &XpuDecoder::get_iterator)
        .def("__next__", &XpuDecoder::get_next_frame, py::return_value_policy::take_ownership)
        .def("set_memory_format", &XpuDecoder::set_memory_format)
        .def("set_output_resolution", &XpuDecoder::set_output_resolution)
        .def("set_async_depth", &XpuDecoder::set_async_depth)
        .def("get_va_device", &XpuDecoder::get_va_device, py::return_value_policy::reference)
        .def("get_original_size", &XpuDecoder::get_original_size)
        .def("set_frame_pool_params", &XpuDecoder::set_frame_pool_params)
        .def("set_loop_mode", &XpuDecoder::set_loop_mode)
        .def("set_output_original_nv12", &XpuDecoder::set_output_original_nv12)
        .def("set_batch_size", &XpuDecoder::set_batch_size);
}
