#pragma once

#include "visual_ai/frame.hpp"

// FWD
class L0Context;

// Frame backed by USM memory
struct UsmFrame : public Frame {
    void* usm_ptr = nullptr;
    uint64_t offset = 0;

    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    std::shared_ptr<L0Context> context;
    std::unique_ptr<Frame> parent_frame;

    UsmFrame() = default;
    UsmFrame(void* usm_ptr, std::shared_ptr<L0Context> l0_context,
             std::unique_ptr<Frame> parent_frame);
    ~UsmFrame();
};
