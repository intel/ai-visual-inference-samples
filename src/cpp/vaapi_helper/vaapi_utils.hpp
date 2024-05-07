#pragma once

#include <string>
#include "visual_ai/memory_format.hpp"

extern "C" {
#include <va/va.h>
}

uint32_t memory_format_to_fourcc(MemoryFormat format);

bool dump_va_surface(VADisplay display, VASurfaceID surface, const std::string& filename);
