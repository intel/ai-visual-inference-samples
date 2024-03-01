#pragma once

#include <string>

extern "C" {
#include <va/va.h>
}

bool dump_va_surface(VADisplay display, VASurfaceID surface, const std::string& filename);