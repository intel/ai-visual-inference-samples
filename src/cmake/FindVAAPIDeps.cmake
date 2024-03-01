# FindVAAPIDeps.cmake

# Guard variable to prevent multiple inclusion effects
if(NOT FIND_VAAPI_DEPS_INCLUDED)
    set(FIND_VAAPI_DEPS_INCLUDED TRUE)

    find_package(PkgConfig)
    pkg_search_module(VA va libva REQUIRED)
    pkg_search_module(VADRM va-drm libva-drm REQUIRED)
endif()
