
function(add_green_dependency TARGET)
    Include(FetchContent)

    set(dependency_repository https://github.com/Green-Phys/${TARGET}.git)
    set(dependency_revision ${GREEN_RELEASE})
    if("${TARGET}" STREQUAL "green-symmetry")
        set(GREEN_SYMMETRY_GIT_REPOSITORY "https://github.com/Green-Phys/green-symmetry.git" CACHE STRING "SG reader dependency repository")
        set(GREEN_SYMMETRY_GIT_TAG "${GREEN_RELEASE}" CACHE STRING "Revision containing the SG reader API")
        set(dependency_repository ${GREEN_SYMMETRY_GIT_REPOSITORY})
        set(dependency_revision ${GREEN_SYMMETRY_GIT_TAG})
    endif()
    FetchContent_Declare(
        ${TARGET}
        GIT_REPOSITORY ${dependency_repository}
        GIT_TAG ${dependency_revision}
    )

    FetchContent_MakeAvailable(${TARGET})
endfunction()
