#include <assert.h>

#include "l0_context.hpp"
#include "usm_frame.hpp"

UsmFrame::UsmFrame(void* usm_ptr, std::shared_ptr<L0Context> l0_context,
                   std::unique_ptr<Frame> parent_frame)
    : usm_ptr(usm_ptr), context(std::move(l0_context)), parent_frame(std::move(parent_frame)) {
    assert(this->usm_ptr);
    assert(this->context);
}

UsmFrame::~UsmFrame() {
    if (!usm_ptr)
        return;

    assert(context);
    ze_result_t res = zeMemFree(context->get_ze_context(), usm_ptr);
    if (res != ZE_RESULT_SUCCESS)
        std::cout << "[ERROR]: Failed to free USM pointer(" << usm_ptr << "), code: " << std::hex
                  << res << std::dec << std::endl;
    usm_ptr = nullptr;
}