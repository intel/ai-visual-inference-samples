#include "l0_utils.hpp"

void l0_workaround_memory(L0Context& l0_context) {
    static ze_device_mem_alloc_desc_t mem_desc = {.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                                  .pNext = nullptr,
                                                  .flags = 0,
                                                  .ordinal = 0};

    void* ptr = nullptr;
    auto ret = zeMemAllocDevice(l0_context.get_ze_context(), &mem_desc, 1, 1,
                                l0_context.get_ze_device(), &ptr);
    if (ret != ZE_RESULT_SUCCESS) {
        std::cout << "[ERROR]: zeMemAllocDevice() failed: " << ret << std::endl;
        return;
    }

    ret = zeMemFree(l0_context.get_ze_context(), ptr);
    if (ret != ZE_RESULT_SUCCESS)
        std::cout << "[ERROR]: zeMemFree() failed: " << ret << std::endl;
}