#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "visual_ai/memory_format.hpp"

#include <assert.h>
#include <dlpack/dlpack.h>

#include "vaapi_context.hpp"
#include "context_manager.hpp"
#include "vaapi_frame_converter.hpp"
#include "vaapi_utils.hpp"
#include "l0_context.hpp"
#include "l0_utils.hpp"
#include "usm_frame.hpp"
#include "scope_guard.hpp"
#include "logger.hpp"

extern "C" {
#include <va/va_drm.h>
#include <va/va_drmcommon.h>
}

namespace py = pybind11;

namespace {

using CropRegion = VARectangle;

} // namespace

// TODO: think about merging with similar one in XpuDecoder.
struct SharedUsmTensor : DLManagedTensor {
    std::unique_ptr<UsmFrame> frame;

    SharedUsmTensor(std::unique_ptr<UsmFrame> usm_frame) : frame(std::move(usm_frame)) {
        dl_tensor.data = frame->usm_ptr;
        dl_tensor.dtype.code = kDLUInt;
        dl_tensor.dtype.bits = 8;
        dl_tensor.dtype.lanes = 1;
        dl_tensor.shape = frame->shape.data();
        dl_tensor.ndim = frame->shape.size();
        dl_tensor.device.device_type = kDLOneAPI;
        dl_tensor.device.device_id = 0; // use first device
        dl_tensor.strides = frame->strides.data();
        dl_tensor.byte_offset = frame->offset;

        // Set the deleter for DLManagedTensor
        deleter = [](DLManagedTensor* self) { delete static_cast<SharedUsmTensor*>(self); };
    }

    static py::object to_pytorch(std::unique_ptr<SharedUsmTensor> managed_tensor) {
        // TODO: try to reduce repeating calls. In general, it'd be great to get from_dlpack only
        // once for whole program execution. Static's doesn't work out of the box because
        // destruction occurs without GIL lock and this leads to a crash.
        py::object torch = py::module_::import("torch");
        py::object torch_utils = torch.attr("utils");
        py::object torch_utils_dlpack = torch_utils.attr("dlpack");
        py::object torch_from_dlpack = torch_utils_dlpack.attr("from_dlpack");

        return torch_from_dlpack(
            py::capsule(managed_tensor.release(), "dltensor", PyCapsule_Destructor(nullptr)));
    }
};

class FrameTransform final {
    std::shared_ptr<VaApiContext> va_context_;
    std::shared_ptr<VaApiFrameConverter> converter_;
    std::shared_ptr<L0Context> l0_context_;

    MemoryFormat memory_format_ = MemoryFormat::pt_planar_rgbp;
    Logger logger_;

  public:
    FrameTransform(const std::string& device_name, int out_width, int out_height)
        : va_context_(std::make_shared<VaApiContext>(
              ContextManager::get_vaapi_diplay_by_device_name(device_name))),
          l0_context_(ContextManager::get_l0_ctx_by_device_name(device_name)) {
        logger_.debug("frame-tform: display={:#x}, out wxh={}x{}",
                      reinterpret_cast<uintptr_t>(va_context_->display_native()), out_width,
                      out_height);

        converter_ = std::make_shared<VaApiFrameConverter>(va_context_);
        converter_->set_ouput_resolution(out_width, out_height);

        // TODO: introduce pool policy
        // Calling side should set proper pool size according to it's usage.
        set_frame_pool_params(32);

        set_memory_format(memory_format_);
    }

    // No copy
    FrameTransform(const FrameTransform&) = delete;
    FrameTransform& operator=(const FrameTransform&) = delete;

    void set_frame_pool_params(uint32_t pool_size) { converter_->set_pool_size(pool_size); }

    void set_memory_format(MemoryFormat format) {
        assert(format != MemoryFormat::unknown);
        assert(converter_);

        const uint32_t va_color_fmt = memory_format_to_fourcc(format);

        converter_->set_output_color_format(va_color_fmt);
        memory_format_ = format;
    }

    auto get_va_context() const { return va_context_; }

    py::object transform_one(VaApiFrame& src_frame, CropRegion crop_region) {
        logger_.debug("frame-tform: surface={}, crop={}", src_frame.desc.va_surface_id,
                      crop_region);

        auto [result_frame, dyn_allocated] = converter_->convert_ex(src_frame, crop_region, true);
        if (dyn_allocated)
            show_out_of_pool_allocation_warn();

        if (memory_format_ == MemoryFormat::ov_planar_nv12)
            return pybind11::cast(std::move(result_frame));

        // PyTorch path
        assert(memory_format_ == MemoryFormat::pt_planar_rgbp ||
               memory_format_ == MemoryFormat::pt_packed_rgba);
        return frame_to_pytorch(std::move(result_frame));
    }

    py::object transform(VaApiFrame& src_frame, std::vector<CropRegion> crop_regions) {
        logger_.debug("frame-tform: surface={}, regions_cout={}", src_frame.desc.va_surface_id,
                      crop_regions.size());

        std::vector<std::unique_ptr<VaApiFrame>> result_frames;
        result_frames.reserve(crop_regions.size());

        bool has_dyn_allocations = false;
        for (auto& r : crop_regions) {
            auto [frame, dyn_allocated] = converter_->convert_ex(src_frame, r, true);
            result_frames.emplace_back(std::move(frame));
            has_dyn_allocations |= dyn_allocated;
        }

        if (has_dyn_allocations)
            show_out_of_pool_allocation_warn();

        if (memory_format_ == MemoryFormat::ov_planar_nv12)
            return pybind11::cast(std::move(result_frames));

        // PyTorch path
        assert(memory_format_ == MemoryFormat::pt_planar_rgbp ||
               memory_format_ == MemoryFormat::pt_packed_rgba);
        std::vector<py::object> py_tensor_objects;
        py_tensor_objects.reserve(result_frames.size());

        // TODO: possible optimizations
        //   - call sync surface only for last surface
        //   - call l0_workaround_memory only once
        for (auto it = result_frames.begin(); it != result_frames.end(); it++) {
            auto py_obj = frame_to_pytorch(std::move(*it));
            py_tensor_objects.emplace_back(std::move(py_obj));
        }
        // NB: result_frames is not valid at this point

        return pybind11::cast(std::move(py_tensor_objects));
    }

    std::unique_ptr<VaApiFrame> crop_frame_region(VaApiFrame& frame, CropRegion region) {
        VaApiFrame* cropped_frame = converter_->convert(frame, region);

        // Create wrapped pointer and cast it to base
        std::unique_ptr<VaApiFrame> wrapped(
            new VaApiFrameWrap(cropped_frame, [converter = this->converter_](VaApiFrame* f) {
                converter->release_frame(f);
            }));

        return wrapped;
    }

    py::object frame_to_pytorch(std::unique_ptr<VaApiFrame> vaframe) {
        // Convert VA memory to USM
        std::unique_ptr<UsmFrame> usm_frame = vaapi_to_usm(std::move(vaframe));

        l0_workaround_memory(*l0_context_);

        // this call transfers ownership from "usm_frame" to "managed_tensor"
        auto managed_tensor = std::make_unique<SharedUsmTensor>(std::move(usm_frame));

        // transfers ownership form "managed_tensor" to "py_object"
        py::object py_object = SharedUsmTensor::to_pytorch(std::move(managed_tensor));

        return std::move(py_object);
    }

    // TODO: should be common function and used within FrameTransform & XpuDecoder
    std::unique_ptr<UsmFrame> vaapi_to_usm(std::unique_ptr<VaApiFrame> vaframe) {
        auto va_dpy = vaframe->desc.va_display;
        auto va_surface = vaframe->desc.va_surface_id;

        // Make sure that surface is ready before exporting it
        VASurfaceStatus surface_status = VASurfaceRendering;
        VAStatus sts = vaQuerySurfaceStatus(va_dpy, va_surface, &surface_status);
        if (sts != VA_STATUS_SUCCESS)
            throw std::runtime_error(std::string("vaQuerySurfaceStatus") +
                                     " failed: " + std::to_string(sts));

        if (surface_status != VASurfaceReady) {
            sts = vaSyncSurface(va_dpy, va_surface);
            if (sts != VA_STATUS_SUCCESS)
                throw std::runtime_error(std::string("vaSyncSurface") +
                                         " failed: " + std::to_string(sts));
        }

        VADRMPRIMESurfaceDescriptor prime_desc;
        sts = vaExportSurfaceHandle(
            va_dpy, va_surface, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
            VA_EXPORT_SURFACE_READ_WRITE /*VA_EXPORT_SURFACE_READ_ONLY |
                                            VA_EXPORT_SURFACE_COMPOSED_LAYERS*/
            ,
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
        ze_result_t ze_res = zeMemAllocDevice(l0_context_->get_ze_context(), &alloc_desc, dma_size,
                                              1, l0_context_->get_ze_device(), &usm_ptr);
        if (sts != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("Failed to convert DMA to USM pointer: " +
                                     std::to_string(sts));
        }

        auto usm_frame = std::make_unique<UsmFrame>(usm_ptr, l0_context_, std::move(vaframe));

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

        return usm_frame;
    }

    void show_out_of_pool_allocation_warn() const {
        logger_.warn("frame-tform: out-of-pool allocation detected! "
                     "Consider increasing pool size (current size: {})",
                     converter_->get_pool_size());
    }
};

void pybind11_submodule_transform(py::module_& parent_module) {
    py::module_ m = parent_module.def_submodule("transform");

    py::class_<CropRegion>(m, "CropRegion")
        .def(py::init<>())
        .def(py::init([](const py::tuple& tup) {
            if (py::len(tup) != 4) {
                throw py::cast_error("Invalid size: 4 elements are expected");
            }
            const int16_t x0 = tup[0].cast<int16_t>();
            const int16_t y0 = tup[1].cast<int16_t>();
            const int16_t x1 = tup[2].cast<int16_t>();
            const int16_t y1 = tup[3].cast<int16_t>();
            return CropRegion{.x = x0,
                              .y = y0,
                              .width = static_cast<uint16_t>(std::abs(x1 - x0)),
                              .height = static_cast<uint16_t>(std::abs(y1 - y0))};
        }))
        .def_readwrite("x", &CropRegion::x)
        .def_readwrite("y", &CropRegion::y)
        .def_readwrite("width", &CropRegion::width)
        .def_readwrite("height", &CropRegion::height)
        .def("__repr__", [](const CropRegion& self) {
            return py::str("<CropRegion x={}, y={}, w={}, h={}>")
                .format(self.x, self.y, self.width, self.height);
        });

    py::implicitly_convertible<py::tuple, CropRegion>();

    py::class_<FrameTransform>(m, "FrameTransform")
        .def(py::init<const std::string&, int, int>())
        .def("set_frame_pool_params", &FrameTransform::set_frame_pool_params)
        .def("set_memory_format", &FrameTransform::set_memory_format)
        .def("transform_one", &FrameTransform::transform_one)
        .def("transform", &FrameTransform::transform);
}
