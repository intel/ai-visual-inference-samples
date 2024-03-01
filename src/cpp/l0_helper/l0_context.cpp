#include "l0_context.hpp"

template <auto Func, typename... Args>
auto ze_call(Args&&... args) -> decltype(Func(args...)) {
    auto result = Func(std::forward<Args>(args)...);
    if (result != ZE_RESULT_SUCCESS) {
        std::ostringstream oss;
        oss << "Level Zero call failed in function " << typeid(Func).name()
            << " with error code: " << result;
        throw std::runtime_error(oss.str());
    }
    return result;
}

L0Context::L0Context(uint32_t ze_device_id) {
    try {
        ze_call<zeInit>(0);

        uint32_t driver_count = 0;
        ze_call<zeDriverGet>(&driver_count, nullptr);

        if (driver_count == 0) {
            throw std::runtime_error("No driver instances found");
        }

        std::vector<ze_driver_handle_t> drivers(driver_count);
        ze_call<zeDriverGet>(&driver_count, drivers.data());

        ze_driver_ = drivers[0];

        uint32_t device_count = 0;
        ze_call<zeDeviceGet>(ze_driver_, &device_count, nullptr);

        if (device_count == 0) {
            throw std::runtime_error("No devices found within driver");
        } else if (device_count < ze_device_id + 1) {
            throw std::runtime_error("Requested device was not found");
        }

        std::vector<ze_device_handle_t> devices(device_count);
        ze_call<zeDeviceGet>(ze_driver_, &device_count, devices.data());

        ze_device_ = devices[ze_device_id];

        ze_context_desc_t context_desc = {};
        context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;

        ze_call<zeContextCreate>(ze_driver_, &context_desc, &ze_context_);
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
    }
}

ze_context_handle_t L0Context::get_ze_context() const {
    return ze_context_;
}

ze_device_handle_t L0Context::get_ze_device() const {
    return ze_device_;
}
