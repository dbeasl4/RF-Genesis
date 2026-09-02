cmake_minimum_required(VERSION 3.20)
project(vulkan_raytracer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(SHADER_SOURCES
    ${CMAKE_SOURCE_DIR}/shaders/raygen.rgen
    ${CMAKE_SOURCE_DIR}/shaders/closesthit.rchit
    ${CMAKE_SOURCE_DIR}/shaders/miss.rmiss
    ${CMAKE_SOURCE_DIR}/shaders/shadow.rmiss
)

set(SPIRV_BINARY_FILES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV_OUTPUT ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPIRV_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${GLSLANG_VALIDATOR} --target-env vulkan1.2 -V ${SHADER} -o ${SPIRV_OUTPUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${SHADER_NAME} -> SPIR-V"
    )
    list(APPEND SPIRV_BINARY_FILES ${SPIRV_OUTPUT})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_BINARY_FILES})

add_executable(vulkan_raytracer src/main.cpp)
target_link_libraries(vulkan_raytracer PRIVATE Vulkan::Vulkan)
add_dependencies(vulkan_raytracer shaders)
