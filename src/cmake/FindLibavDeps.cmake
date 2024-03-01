# FindLibavDeps.cmake

# Guard variable to prevent multiple inclusion effects
if(NOT FIND_LIBAV_DEPS_INCLUDED)
    set(FIND_LIBAV_DEPS_INCLUDED TRUE)

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
        libavdevice
        libavfilter
        libavformat
        libavcodec
        libavutil
    )
endif()
