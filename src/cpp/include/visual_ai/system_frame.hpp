#pragma once

#include <memory>
#include <vector>
#include <stdexcept>

#include "visual_ai/frame.hpp"

struct SystemFrame : public Frame {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    SystemFrame() = default;
    SystemFrame(std::vector<uint8_t> system_memory, std::vector<int64_t> shape,
                std::vector<int64_t> strides, uint64_t offset = 0,
                std::unique_ptr<Frame> parent_frame = nullptr)
        : system_memory(std::move(system_memory)), shape(shape), strides(strides), offset(offset),
          parent_frame_(std::move(parent_frame)) {}

    void sync() const override {
        if (parent_frame_) {
            parent_frame_->sync();
        }
    }
    void* raw() noexcept { return system_memory.data() + offset; }

  private:
    std::vector<uint8_t> system_memory;
    uint64_t offset = 0;
    std::unique_ptr<Frame> parent_frame_;
};
