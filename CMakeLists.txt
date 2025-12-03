cmake_minimum_required(VERSION 3.24)
project(polynomial C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# msvc doesn't support c23 (smh), use c17
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

# options
option(STATIC_LINK "static link runtime libraries" ON)
option(PORTABLE_BUILD "disable -march=native for portable binaries" OFF)
option(COMPAT_COMPLEX_DOUBLE_ONLY "use double instead of long double for complex (msvc compat)" OFF)

# force double-only mode on msvc (long double == double anyway)
if(MSVC)
    set(COMPAT_COMPLEX_DOUBLE_ONLY ON)
endif()

if(COMPAT_COMPLEX_DOUBLE_ONLY)
    add_compile_definitions(COMPAT_COMPLEX_DOUBLE_ONLY)
    message(STATUS "Complex: double precision only")
else()
    message(STATUS "Complex: long double precision")
endif()

# compiler flags
if(MSVC)
    set(CMAKE_C_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")

    # static crt linkage
    if(STATIC_LINK)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:DebugDLL>")
    endif()

    # favor speed, enable intrinsics
    add_compile_options("$<$<CONFIG:Release>:/Oi>" "$<$<CONFIG:Release>:/Ot>")

    # c99 compatibility for msvc
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_definitions(_USE_MATH_DEFINES)
else()
    # gcc/clang/mingw
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

    # linker: gc sections
    if(APPLE)
        string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " -Wl,-dead_strip")
    else()
        string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " -Wl,--gc-sections")
    endif()

    # static linking
    if(STATIC_LINK AND NOT APPLE)
        if(MINGW)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
        else()
            # linux: static libgcc/libstdc++ only (glibc must be dynamic)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc")
        endif()
    endif()
endif()

message(STATUS "C flags (Release): ${CMAKE_C_FLAGS_RELEASE}")

# link-time optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    message(STATUS "LTO: enabled")
else()
    message(STATUS "LTO: not supported (${ipo_error})")
endif()

# OpenMP
if(APPLE)
    # macos with homebrew libomp needs special handling
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
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP: found")
else()
    message(STATUS "OpenMP: not found")
endif()

# LAPACK
set(LAPACK_FOUND FALSE)

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        set(LAPACK_FOUND TRUE)
        set(LAPACK_LIBRARIES ${ACCELERATE_FRAMEWORK})
        set(LAPACK_DEFINITIONS HAVE_LAPACK ACCELERATE_NEW_LAPACK)
        message(STATUS "LAPACK: Accelerate framework")
    endif()
elseif(WIN32)
    # try intel mkl
    if(NOT DEFINED ENV{MKLROOT})
        if(EXISTS "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
            set(ENV{MKLROOT} "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
        endif()
    endif()

    set(BLA_VENDOR Intel10_64lp_seq)
    if(STATIC_LINK)
        set(BLA_STATIC ON)
    endif()
    find_package(LAPACK QUIET)
    if(LAPACK_FOUND)
        set(LAPACK_DEFINITIONS HAVE_LAPACK)
        message(STATUS "LAPACK: Intel MKL")
    else()
        message(STATUS "LAPACK: not found (install Intel MKL or set MKLROOT)")
    endif()
else()
    # linux: try system lapack
    find_package(LAPACK QUIET)
    if(LAPACK_FOUND)
        set(LAPACK_DEFINITIONS HAVE_LAPACK)
        message(STATUS "LAPACK: system")
    else()
        message(STATUS "LAPACK: not found")
    endif()
endif()

# OpenCL
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

# cJSON has old cmake_minimum_required, suppress the warning
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# SDL3
message(STATUS "Fetching SDL3...")

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

set(SDL_DIRECTX OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D11 OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_D3D12 OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_VULKAN OFF CACHE BOOL "" FORCE)
set(SDL_RENDER_GPU OFF CACHE BOOL "" FORCE)
set(SDL_GPU OFF CACHE BOOL "" FORCE)
set(SDL_VULKAN OFF CACHE BOOL "" FORCE)

set(SDL_OPENGL ON CACHE BOOL "" FORCE)
set(SDL_OPENGLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-3.2.26
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(SDL3)

# cJSON
message(STATUS "Fetching cJSON...")

set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
set(CJSON_OVERRIDE_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(CJSON_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
FetchContent_Declare(cJSON
        GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
        GIT_TAG        v1.7.19
        GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cJSON)

# suppress cjson float-conversion warnings
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cjson PRIVATE "-Wno-float-conversion")
elseif(MSVC)
    target_compile_options(cjson PRIVATE "/wd4244")
endif()

# collect optional libs
set(POLY_LIBS "")

if(OpenMP_C_FOUND)
    list(APPEND POLY_LIBS OpenMP::OpenMP_C)
endif()

if(LAPACK_FOUND)
    list(APPEND POLY_LIBS ${LAPACK_LIBRARIES})
endif()

# math library (not needed on windows)
if(NOT WIN32)
    list(APPEND POLY_LIBS m)
endif()

# sources
set(CORE_SOURCES
        src/polynomial.c
        src/polysolve.c
        src/companion.c
        src/util.c
)

set(RENDER_SOURCES
        src/render.c
        src/render_cpu.c
        src/render_gpu.c
)

set(APP_SOURCES
        src/main.c
        src/app.c
        src/input.c
        src/ui.c
        src/cloud.c
        src/animation.c
)

set(SOURCES ${CORE_SOURCES} ${RENDER_SOURCES} ${APP_SOURCES})
set(POLY_LIB_SOURCES ${CORE_SOURCES})

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
    target_link_libraries(test_multithread PRIVATE ${POLY_LIBS})
    if(LAPACK_FOUND)
        target_compile_definitions(test_multithread PRIVATE ${LAPACK_DEFINITIONS})
    endif()

    add_executable(test_throughput tests/throughput.c ${POLY_LIB_SOURCES})
    target_include_directories(test_throughput PRIVATE inc)
    target_link_libraries(test_throughput PRIVATE ${POLY_LIBS})
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
    target_link_libraries(test_polynomial PRIVATE ${POLY_LIBS})
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

# summary
message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  Static link:   ${STATIC_LINK}")
message(STATUS "  Portable:      ${PORTABLE_BUILD}")
message(STATUS "  OpenMP:        ${OpenMP_C_FOUND}")
message(STATUS "  OpenCL:        ${OpenCL_FOUND}")
message(STATUS "  LAPACK:        ${LAPACK_FOUND}")
message(STATUS "  LTO:           ${ipo_supported}")
message(STATUS "")