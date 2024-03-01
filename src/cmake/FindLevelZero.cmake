# FindLevelZero.cmake

# Guard variable to prevent multiple inclusion effects
if(NOT FIND_LEVEL_ZERO_DEPS_INCLUDED)
    set(FIND_LEVEL_ZERO_DEPS_INCLUDED TRUE)

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBZE_LOADER REQUIRED IMPORTED_TARGET libze_loader)

    if(NOT TARGET LevelZero::LevelZero)
        add_library(LevelZero::LevelZero INTERFACE IMPORTED)
        set_target_properties(LevelZero::LevelZero PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LIBZE_LOADER_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${LIBZE_LOADER_LINK_LIBRARIES}"
        )
    endif()
endif()
