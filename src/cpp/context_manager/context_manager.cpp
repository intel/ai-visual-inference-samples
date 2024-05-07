#include "context_manager.hpp"

ContextManager& ContextManager::get_instance() {
    static ContextManager instance;
    return instance;
}

std::shared_ptr<L0Context> ContextManager::get_l0_context(const uint32_t device) {
    auto it = l0_contexts_.find(device);
    if (it != l0_contexts_.end()) {
        return it->second;
    }
    // Create L0 context for the device
    std::shared_ptr<L0Context> new_l0_context = std::make_shared<L0Context>(device);
    l0_contexts_[device] = new_l0_context;
    return new_l0_context;
}

VaDpyWrapper ContextManager::get_va_display(const std::string& device) {
    auto it = vaapi_display_wrappers_.find(device);
    if (it != vaapi_display_wrappers_.end()) {
        return it->second;
    }
    vaapi_display_wrappers_[device] = VaDpyWrapper(device);
    return vaapi_display_wrappers_[device];
}

std::string ContextManager::get_device_path_from_device_name(const std::string& device_name) {
    static const std::string default_device_path = "/dev/dri/renderD128";
    static const int start_device_id = 128;
    if (device_name.find("xpu") == std::string::npos)
        throw std::runtime_error("Unsupported device: " + device_name);
    auto del_pos = device_name.find(":");
    if (del_pos == std::string::npos)
        return default_device_path;
    try {
        auto device_id = std::stoi(device_name.substr(del_pos + 1, device_name.size()));
        return "/dev/dri/renderD" + std::to_string(start_device_id + device_id);
    } catch (std::invalid_argument const& ex) {
        throw std::invalid_argument("Unsupported device id in device name: " + device_name);
    } catch (std::out_of_range const& ex) {
        // log_warn("No device id passed after xpu: returning default device");
        return default_device_path;
    }
}

/* Assume that numeration of level zero devices is identical to IPEX and have correspondence
   to vaDisplay creation like
   0 device /dev/dri/renderD128
   1 device /dev/dri/renderD129
   ....
   It might be wrong for multicards configuration and should be updated later accordingly.*/
uint32_t ContextManager::get_ze_device_id(const std::string& device_name) {
    auto del_pos = device_name.find(":");
    if (del_pos == std::string::npos) {
        return 0;
    }
    uint32_t device_id = std::stoi(device_name.substr(del_pos + 1, device_name.size()));
    return device_id;
}
