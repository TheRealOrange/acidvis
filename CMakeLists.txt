cmake_minimum_required(VERSION 3.24)
project(polynomial C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")

# compiler flags
if(MSVC)
    set(CMAKE_C_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")

    # static CRT linkage (no msvc runtime dependency)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    # favor speed, enable intrinsics
    add_compile_options("$<$<CONFIG:Release>:/Oi>" "$<$<CONFIG:Release>:/Ot>")

elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    # enable linker garbage collection
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

    # arch tuning
    if(NOT PORTABLE_BUILD)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            string(APPEND CMAKE_C_FLAGS_RELEASE " -mcpu=native")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
            string(APPEND CMAKE_C_FLAGS_RELEASE " -march=native -mtune=native")
        endif()
    endif()

    # auto-vectorization
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        string(APPEND CMAKE_C_FLAGS_RELEASE " -ftree-vectorize")
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang")
        string(APPEND CMAKE_C_FLAGS_RELEASE " -fvectorize")
    endif()

    # linker garbage collection to keep binary small
    if(APPLE)
        string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " -Wl,-dead_strip")
    else()
        string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " -Wl,--gc-sections")
    endif()

    # static linking option for gcc on linux
    option(STATIC_LIBGCC "statically link libgcc" OFF)
    if(STATIC_LIBGCC AND NOT APPLE)
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -static-libgcc")
    endif()
endif()

message(STATUS "C flags (Release): ${CMAKE_C_FLAGS_RELEASE}")

# Link-Time Optimization (for release only)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    message(STATUS "LTO: enabled")
else()
    message(STATUS "LTO: not supported (${ipo_error})")
endif()

# OpenMP for multithreading
# macOS with Homebrew libomp needs special handling
if(APPLE)
    execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )
    if(HOMEBREW_LIBOMP_PREFIX AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "Found homebrew libomp: ${HOMEBREW_LIBOMP_PREFIX}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        include_directories("${HOMEBREW_LIBOMP_PREFIX}/include")
        link_directories("${HOMEBREW_LIBOMP_PREFIX}/lib")
    endif()
endif()

find_package(OpenMP)
message(STATUS "OpenMP: ${OpenMP_C_FOUND}")

# LAPACK support
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        set(LAPACK_FOUND TRUE)
        set(LAPACK_LIBRARIES ${ACCELERATE_FRAMEWORK})
        set(LAPACK_DEFINITIONS HAVE_LAPACK ACCELERATE_NEW_LAPACK)
        message(STATUS "LAPACK: Accelerate framework")
    endif()
else()
    find_package(LAPACK)
    if(LAPACK_FOUND)
        set(LAPACK_DEFINITIONS HAVE_LAPACK)
        message(STATUS "LAPACK: ${LAPACK_LIBRARIES}")
    else()
        message(STATUS "LAPACK: not found")
    endif()
endif()

# OpenCL support
find_package(OpenCL)
if(OpenCL_FOUND)
    message(STATUS "OpenCL: enabled")
    add_compile_definitions(HAVE_OPENCL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/embed_file.cmake)
    embed_opencl_kernel(
            INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/render.cl"
            OUTPUT_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/inc/generated/kernel_source.h"
            VARIABLE_NAME "kernel_source"
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/render.cl"
    )
else()
    message(STATUS "OpenCL: not found (GPU acceleration disabled)")
endif()

# dependencies
include(FetchContent)

# cJSON has old cmake_minimum_required, suppress the warning/error
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# SDL3
message(STATUS "fetching SDL3...")

# disable everything we don't need
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
set(SDL_HIDAPI OFF CACHE BOOL "" FORCE)
set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
set(SDL_POWER OFF CACHE BOOL "" FORCE)
set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
set(SDL_DIALOG OFF CACHE BOOL "" FORCE)

# disable unused video drivers
set(SDL_DIRECTX OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D11 OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D12 OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_VULKAN OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_GPU OFF CACHE BOOL "" FORCE)
set(SDL_GPU OFF CACHE BOOL "" FORCE)
set(SDL_VULKAN OFF CACHE BOOL "" FORCE)

# need OpenGL for rendering
set(SDL_OPENGL ON CACHE BOOL "" FORCE)
set(SDL_OPENGLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-3.2.26
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(SDL3)

message(STATUS "fetching cJSON...")

# cJSON
set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
set(CJSON_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
FetchContent_Declare(cJSON
        GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
        GIT_TAG        v1.7.19
        GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cJSON)

# optional libs used by polynomial code
set(POLY_LIBS "")
if(OpenMP_C_FOUND)
    list(APPEND POLY_LIBS OpenMP::OpenMP_C)
endif()
if(LAPACK_FOUND)
    list(APPEND POLY_LIBS ${LAPACK_LIBRARIES})
endif()

# sources
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

set(POLY_LIB_SOURCES
        src/polynomial.c
        src/polysolve.c
        src/companion.c
        src/util.c
)

# main executable
add_executable(polynomial ${SOURCES})
target_include_directories(polynomial PRIVATE inc inc/generated ${cjson_SOURCE_DIR})
target_link_libraries(polynomial PRIVATE SDL3::SDL3-static cjson ${POLY_LIBS})
if(LAPACK_FOUND)
    target_compile_definitions(polynomial PRIVATE ${LAPACK_DEFINITIONS})
endif()
if(OpenCL_FOUND)
    target_include_directories(polynomial PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(polynomial PRIVATE ${OpenCL_LIBRARIES})
endif()
if(WIN32)
    target_link_libraries(polynomial PRIVATE winmm imm32 version setupapi)
endif()

# benchmarks
option(BUILD_BENCHMARK "build benchmarks" ON)
if(BUILD_BENCHMARK)
    add_executable(test_multithread tests/polymp.c ${POLY_LIB_SOURCES})
    target_include_directories(test_multithread PRIVATE inc)
    target_link_libraries(test_multithread PRIVATE m ${POLY_LIBS})
    if(LAPACK_FOUND)
        target_compile_definitions(test_multithread PRIVATE ${LAPACK_DEFINITIONS})
    endif()

    add_executable(test_throughput tests/throughput.c ${POLY_LIB_SOURCES})
    target_include_directories(test_throughput PRIVATE inc)
    target_link_libraries(test_throughput PRIVATE m ${POLY_LIBS})
    if(LAPACK_FOUND)
        target_compile_definitions(test_throughput PRIVATE ${LAPACK_DEFINITIONS})
    endif()
endif()

# tests
option(BUILD_TESTS "build test suite" ON)
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_polynomial tests/polytest.c ${POLY_LIB_SOURCES})
    target_include_directories(test_polynomial PRIVATE inc)
    target_link_libraries(test_polynomial PRIVATE m ${POLY_LIBS})
    if(LAPACK_FOUND)
        target_compile_definitions(test_polynomial PRIVATE ${LAPACK_DEFINITIONS})
    endif()

    add_test(NAME polynomial_tests COMMAND test_polynomial)

    add_custom_target(run_tests
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
            DEPENDS test_polynomial
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "running polynomial test suite"
    )
endif()

message(STATUS "")
message(STATUS "build configuration")
message(STATUS "  build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  openmp:        ${OpenMP_C_FOUND}")
message(STATUS "  opencl:        ${OpenCL_FOUND}")
message(STATUS "  lapack:        ${LAPACK_FOUND}")
message(STATUS "  lto:           ${ipo_supported}")
message(STATUS "")