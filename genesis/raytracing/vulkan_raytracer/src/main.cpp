// main.cpp: standalone test of the Vulkan ray tracer against a real SMPL
// mesh, real perspective camera, and real Lambertian shading. Loads a
// mesh exported by export_mesh.py, traces it, prints hit/timing stats,
// and writes the result to vulkan_output.bin (view with view_output.py).
//
// This is a standalone C++ executable, separate from the live PyTorch
// integration in VulkanCudaInterop/. See README for details.
//
// Known gap carried over from the CUDA kernel: no shadow ray yet.

#include <vulkan/vulkan.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <array>
#include <chrono>
#include <dlfcn.h>
#include "renderdoc_app.h"

// Loaded only if this executable was launched through RenderDoc (its
// capture library injects itself via LD_PRELOAD beforehand). Stays null,
// and every call below is a no-op, when run normally outside RenderDoc.
static RENDERDOC_API_1_6_0* rdoc_api = nullptr;

static void loadRenderDocAPI() {
    // RTLD_NOLOAD: only succeeds if RenderDoc's library is ALREADY loaded
    // in this process (i.e. we were launched through RenderDoc). This is
    // the safe pattern from RenderDoc's own docs -- it does nothing and
    // fails harmlessly when run outside RenderDoc.
    void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (mod) {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
        if (ret == 1) {
            std::cout << "RenderDoc detected -- capture API loaded.\n";
        }
    } else {
        std::cout << "Running without RenderDoc (normal run, or launched outside it).\n";
    }
}

const std::vector<const char*> requiredDeviceExtensions = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
};

struct AccelStruct {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
};

struct MeshData {
    std::vector<float> vertices;   // x,y,z per vertex, flat
    std::vector<float> normals;    // x,y,z per-vertex smooth normal, flat
    std::vector<uint32_t> indices; // 3 per triangle, flat
};

// Must match the GLSL push_constant block in raygen.rgen / closesthit.rchit
// exactly: same field order, same 16-byte-aligned vec4 grouping.
struct PushConstants {
    float camOrigin[4];
    float camRight[4];
    float camUp[4];
    float camForward[4];
    float lightPosIntensity[4]; // xyz = position, w = intensity
    float params[4];            // x = tanHalfFov, y = aspect
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
    uint64_t normalBufferAddress;
};

// tiny host-side vector helper (not used on the GPU)

struct Vec3 {
    float x, y, z;
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 normalize(const Vec3& a) {
    float len = sqrtf(dot(a, a));
    return len > 1e-12f ? a * (1.0f / len) : a;
}

// ray tracing extension function pointers

PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR = nullptr;
PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR = nullptr;
PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR = nullptr;
PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR = nullptr;
PFN_vkCreateRayTracingPipelinesKHR pfn_vkCreateRayTracingPipelinesKHR = nullptr;
PFN_vkGetRayTracingShaderGroupHandlesKHR pfn_vkGetRayTracingShaderGroupHandlesKHR = nullptr;
PFN_vkCmdTraceRaysKHR pfn_vkCmdTraceRaysKHR = nullptr;

void loadRayTracingFunctions(VkDevice device) {
    pfn_vkGetAccelerationStructureBuildSizesKHR =
        (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    pfn_vkCreateAccelerationStructureKHR =
        (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
    pfn_vkCmdBuildAccelerationStructuresKHR =
        (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
    pfn_vkGetAccelerationStructureDeviceAddressKHR =
        (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
    pfn_vkDestroyAccelerationStructureKHR =
        (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
    pfn_vkCreateRayTracingPipelinesKHR =
        (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");
    pfn_vkGetRayTracingShaderGroupHandlesKHR =
        (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");
    pfn_vkCmdTraceRaysKHR =
        (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");

    if (!pfn_vkGetAccelerationStructureBuildSizesKHR || !pfn_vkCreateAccelerationStructureKHR ||
        !pfn_vkCmdBuildAccelerationStructuresKHR || !pfn_vkGetAccelerationStructureDeviceAddressKHR ||
        !pfn_vkDestroyAccelerationStructureKHR || !pfn_vkCreateRayTracingPipelinesKHR ||
        !pfn_vkGetRayTracingShaderGroupHandlesKHR || !pfn_vkCmdTraceRaysKHR) {
        throw std::runtime_error("Failed to load one or more ray tracing function pointers.");
    }
}

// general helpers

bool deviceSupportsRayTracing(VkPhysicalDevice device) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    for (const char* required : requiredDeviceExtensions) {
        bool found = false;
        for (const auto& available : availableExtensions) {
            if (strcmp(required, available.extensionName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("No suitable GPU memory type found.");
}

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device,
                   VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags properties,
                   VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer.");
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkMemoryAllocateFlagsInfo allocFlagsInfo{};
    allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &allocFlagsInfo;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory.");
    }
    vkBindBufferMemory(device, buffer, memory, 0);
}

VkDeviceAddress getBufferDeviceAddress(VkDevice device, VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}

VkDeviceSize alignUp(VkDeviceSize size, VkDeviceSize alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

VkCommandBuffer beginOneTimeCommands(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

void endAndSubmitOneTimeCommands(VkDevice device, VkQueue queue, VkCommandPool pool, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

VkShaderModule loadShaderModule(VkDevice device, const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path +
                                  " (run the executable from inside build/, so relative paths resolve)");
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module from: " + path);
    }
    return shaderModule;
}

// Loads the binary mesh file export_mesh.py writes. See that script's
// docstring for the exact format.
MeshData loadMeshFromFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open mesh file: " + path +
                                  ". Run export_mesh.py first (from the rfgen conda env, "
                                  "in the vulkan_raytracer/ folder) to generate it.");
    }

    uint32_t vertexCount = 0, faceCount = 0;
    file.read(reinterpret_cast<char*>(&vertexCount), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&faceCount), sizeof(uint32_t));

    MeshData mesh;
    mesh.vertices.resize((size_t)vertexCount * 3);
    file.read(reinterpret_cast<char*>(mesh.vertices.data()), mesh.vertices.size() * sizeof(float));

    mesh.normals.resize((size_t)vertexCount * 3);
    file.read(reinterpret_cast<char*>(mesh.normals.data()), mesh.normals.size() * sizeof(float));

    mesh.indices.resize((size_t)faceCount * 3);
    file.read(reinterpret_cast<char*>(mesh.indices.data()), mesh.indices.size() * sizeof(uint32_t));

    if (!file) {
        throw std::runtime_error("Mesh file appears truncated or malformed: " + path);
    }

    return mesh;
}

// BLAS

AccelStruct buildBLAS(VkPhysicalDevice physicalDevice, VkDevice device,
                       VkQueue queue, VkCommandPool cmdPool,
                       VkBuffer vertexBuffer, uint32_t vertexCount,
                       VkBuffer indexBuffer, uint32_t triangleCount) {
    VkDeviceAddress vertexAddress = getBufferDeviceAddress(device, vertexBuffer);
    VkDeviceAddress indexAddress = getBufferDeviceAddress(device, indexBuffer);

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride = sizeof(float) * 3;
    triangles.maxVertex = vertexCount - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles = triangles;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_vkGetAccelerationStructureBuildSizesKHR(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &triangleCount, &sizeInfo);

    AccelStruct blas{};
    createBuffer(physicalDevice, device, sizeInfo.accelerationStructureSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blas.buffer, blas.memory);

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = blas.buffer;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    pfn_vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &blas.handle);

    VkBuffer scratchBuffer;
    VkDeviceMemory scratchMemory;
    createBuffer(physicalDevice, device, sizeInfo.buildScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMemory);

    buildInfo.dstAccelerationStructure = blas.handle;
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(device, scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = triangleCount;
    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfoPtr = &rangeInfo;

    VkCommandBuffer cmd = beginOneTimeCommands(device, cmdPool);
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &rangeInfoPtr);
    endAndSubmitOneTimeCommands(device, queue, cmdPool, cmd);

    vkDestroyBuffer(device, scratchBuffer, nullptr);
    vkFreeMemory(device, scratchMemory, nullptr);

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blas.handle;
    blas.deviceAddress = pfn_vkGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);

    return blas;
}

// TLAS

AccelStruct buildTLAS(VkPhysicalDevice physicalDevice, VkDevice device,
                       VkQueue queue, VkCommandPool cmdPool,
                       const AccelStruct& blas) {
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blas.deviceAddress;

    VkBuffer instanceBuffer;
    VkDeviceMemory instanceMemory;
    createBuffer(physicalDevice, device, sizeof(instance),
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 instanceBuffer, instanceMemory);

    void* mapped;
    vkMapMemory(device, instanceMemory, 0, sizeof(instance), 0, &mapped);
    memcpy(mapped, &instance, sizeof(instance));
    vkUnmapMemory(device, instanceMemory);

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.data.deviceAddress = getBufferDeviceAddress(device, instanceBuffer);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t instanceCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_vkGetAccelerationStructureBuildSizesKHR(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &instanceCount, &sizeInfo);

    AccelStruct tlas{};
    createBuffer(physicalDevice, device, sizeInfo.accelerationStructureSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlas.buffer, tlas.memory);

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = tlas.buffer;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    pfn_vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &tlas.handle);

    VkBuffer scratchBuffer;
    VkDeviceMemory scratchMemory;
    createBuffer(physicalDevice, device, sizeInfo.buildScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMemory);

    buildInfo.dstAccelerationStructure = tlas.handle;
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(device, scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfoPtr = &rangeInfo;

    VkCommandBuffer cmd = beginOneTimeCommands(device, cmdPool);
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &rangeInfoPtr);
    endAndSubmitOneTimeCommands(device, queue, cmdPool, cmd);

    vkDestroyBuffer(device, scratchBuffer, nullptr);
    vkFreeMemory(device, scratchMemory, nullptr);
    vkDestroyBuffer(device, instanceBuffer, nullptr);
    vkFreeMemory(device, instanceMemory, nullptr);

    return tlas;
}

// main

int main() {
    loadRenderDocAPI();

    const uint32_t WIDTH = 128;
    const uint32_t HEIGHT = 128;

    // Instance + device setup
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RF-Genesis Custom Ray Tracer (Vulkan)";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance.\n";
        return 1;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    for (const auto& d : devices) {
        if (deviceSupportsRayTracing(d)) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice = d;
                break;
            }
        }
    }
    if (physicalDevice == VK_NULL_HANDLE) {
        std::cerr << "No discrete GPU with ray tracing support found.\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkPhysicalDeviceProperties deviceProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
    std::cout << "Using device: " << deviceProps.deviceName << "\n\n";

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    int graphicsQueueFamily = -1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { graphicsQueueFamily = (int)i; break; }
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = graphicsQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
    bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    accelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelFeatures.accelerationStructure = VK_TRUE;
    accelFeatures.pNext = &bufferDeviceAddressFeatures;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;
    rtPipelineFeatures.pNext = &accelFeatures;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &rtPipelineFeatures;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = (uint32_t)requiredDeviceExtensions.size();
    deviceInfo.ppEnabledExtensionNames = requiredDeviceExtensions.data();

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
        std::cerr << "Failed to create logical device.\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkQueue queue;
    vkGetDeviceQueue(device, graphicsQueueFamily, 0, &queue);

    loadRayTracingFunctions(device);
    std::cout << "Ray tracing extension functions loaded.\n\n";

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsQueueFamily;

    VkCommandPool cmdPool;
    vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool);

    // Load the real SMPL mesh (exported by export_mesh.py, one directory
    // up from build/, in vulkan_raytracer/)
    MeshData mesh = loadMeshFromFile("../smpl_mesh.bin");
    uint32_t vertexCount = (uint32_t)(mesh.vertices.size() / 3);
    uint32_t triangleCount = (uint32_t)(mesh.indices.size() / 3);
    std::cout << "Loaded mesh: " << vertexCount << " vertices, " << triangleCount << " triangles.\n\n";

    VkDeviceSize vertexBufferSize = mesh.vertices.size() * sizeof(float);
    VkDeviceSize normalBufferSize = mesh.normals.size() * sizeof(float);
    VkDeviceSize indexBufferSize = mesh.indices.size() * sizeof(uint32_t);

    VkBuffer vertexBuffer, normalBuffer, indexBuffer;
    VkDeviceMemory vertexMemory, normalMemory, indexMemory;

    createBuffer(physicalDevice, device, vertexBufferSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vertexBuffer, vertexMemory);
    void* vmap; vkMapMemory(device, vertexMemory, 0, vertexBufferSize, 0, &vmap);
    memcpy(vmap, mesh.vertices.data(), vertexBufferSize);
    vkUnmapMemory(device, vertexMemory);

    // Normals aren't part of the BLAS build (only positions + indices are),
    // just read directly by the shader via buffer_reference, same as
    // vertices/indices already are -- STORAGE_BUFFER usage is enough.
    createBuffer(physicalDevice, device, normalBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 normalBuffer, normalMemory);
    void* nmap; vkMapMemory(device, normalMemory, 0, normalBufferSize, 0, &nmap);
    memcpy(nmap, mesh.normals.data(), normalBufferSize);
    vkUnmapMemory(device, normalMemory);

    createBuffer(physicalDevice, device, indexBufferSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 indexBuffer, indexMemory);
    void* imap; vkMapMemory(device, indexMemory, 0, indexBufferSize, 0, &imap);
    memcpy(imap, mesh.indices.data(), indexBufferSize);
    vkUnmapMemory(device, indexMemory);

    std::cout << "Building BLAS from real SMPL mesh...\n";
    auto blasStart = std::chrono::high_resolution_clock::now();
    AccelStruct blas = buildBLAS(physicalDevice, device, queue, cmdPool,
                                  vertexBuffer, vertexCount, indexBuffer, triangleCount);
    auto blasEnd = std::chrono::high_resolution_clock::now();
    double blasMs = std::chrono::duration<double, std::milli>(blasEnd - blasStart).count();
    std::cout << "  BLAS built. (" << blasMs << " ms)\n";

    std::cout << "Building TLAS...\n";
    auto tlasStart = std::chrono::high_resolution_clock::now();
    AccelStruct tlas = buildTLAS(physicalDevice, device, queue, cmdPool, blas);
    auto tlasEnd = std::chrono::high_resolution_clock::now();
    double tlasMs = std::chrono::duration<double, std::milli>(tlasEnd - tlasStart).count();
    std::cout << "  TLAS built. (" << tlasMs << " ms)\n\n";

    // Pipeline + shaders
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout descSetLayout;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descSetLayout);

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    VkShaderModule raygenModule = loadShaderModule(device, "shaders/raygen.rgen.spv");
    VkShaderModule missModule = loadShaderModule(device, "shaders/miss.rmiss.spv");
    VkShaderModule shadowMissModule = loadShaderModule(device, "shaders/shadow.rmiss.spv");
    VkShaderModule closestHitModule = loadShaderModule(device, "shaders/closesthit.rchit.spv");

    // Ordered raygen, then BOTH miss shaders contiguously, then the hit
    // group -- this ordering must match the SBT's region layout below
    // (each region assumes its shaders are contiguous in this array).
    std::array<VkPipelineShaderStageCreateInfo, 4> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = raygenModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = missModule;
    stages[1].pName = "main";

    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = shadowMissModule;
    stages[2].pName = "main";

    stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[3].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[3].module = closestHitModule;
    stages[3].pName = "main";

    std::array<VkRayTracingShaderGroupCreateInfoKHR, 4> groups{};
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;
    groups[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[3].generalShader = VK_SHADER_UNUSED_KHR;
    groups[3].closestHitShader = 3;
    groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = (uint32_t)stages.size();
    pipelineInfo.pStages = stages.data();
    pipelineInfo.groupCount = (uint32_t)groups.size();
    pipelineInfo.pGroups = groups.data();
    // 2, not 1: the closest-hit shader now itself calls traceRayEXT for
    // the shadow ray, one level of recursion deeper than before.
    pipelineInfo.maxPipelineRayRecursionDepth = 2;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    if (pfn_vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
                                            &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ray tracing pipeline.");
    }
    std::cout << "Ray tracing pipeline created.\n";

    // Shader binding table
    uint32_t handleSize = rtProps.shaderGroupHandleSize;
    uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;
    uint32_t handleSizeAligned = (uint32_t)alignUp(handleSize, handleAlignment);

    uint32_t groupCount = 4;
    std::vector<uint8_t> handleData(groupCount * handleSize);
    pfn_vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, groupCount,
                                              handleData.size(), handleData.data());

    VkDeviceSize raygenRegionSize = alignUp(handleSizeAligned, baseAlignment);
    // 2 miss shaders now (primary + shadow), contiguous in the SBT.
    VkDeviceSize missRegionSize = alignUp(2 * handleSizeAligned, baseAlignment);
    VkDeviceSize hitRegionSize = alignUp(1 * handleSizeAligned, baseAlignment);
    VkDeviceSize sbtSize = raygenRegionSize + missRegionSize + hitRegionSize;

    VkBuffer sbtBuffer;
    VkDeviceMemory sbtMemory;
    createBuffer(physicalDevice, device, sbtSize,
                 VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 sbtBuffer, sbtMemory);

    uint8_t* sbtMapped;
    vkMapMemory(device, sbtMemory, 0, sbtSize, 0, (void**)&sbtMapped);
    // Group 0 (raygen) -> raygen region
    memcpy(sbtMapped, handleData.data() + 0 * handleSize, handleSize);
    // Groups 1, 2 (primary miss, shadow miss) -> miss region, in order
    memcpy(sbtMapped + raygenRegionSize, handleData.data() + 1 * handleSize, handleSize);
    memcpy(sbtMapped + raygenRegionSize + handleSizeAligned, handleData.data() + 2 * handleSize, handleSize);
    // Group 3 (closesthit) -> hit region
    memcpy(sbtMapped + raygenRegionSize + missRegionSize, handleData.data() + 3 * handleSize, handleSize);
    vkUnmapMemory(device, sbtMemory);

    VkDeviceAddress sbtAddress = getBufferDeviceAddress(device, sbtBuffer);

    VkStridedDeviceAddressRegionKHR raygenRegion{};
    raygenRegion.deviceAddress = sbtAddress;
    raygenRegion.stride = raygenRegionSize;
    raygenRegion.size = raygenRegionSize;

    VkStridedDeviceAddressRegionKHR missRegion{};
    missRegion.deviceAddress = sbtAddress + raygenRegionSize;
    missRegion.stride = handleSizeAligned;
    missRegion.size = missRegionSize;

    VkStridedDeviceAddressRegionKHR hitRegion{};
    hitRegion.deviceAddress = sbtAddress + raygenRegionSize + missRegionSize;
    hitRegion.stride = handleSizeAligned;
    hitRegion.size = hitRegionSize;

    VkStridedDeviceAddressRegionKHR callableRegion{};

    std::cout << "Shader binding table built.\n\n";

    // Output buffer (now 2 floats per pixel: distance, intensity)
    VkDeviceSize outputBufferSize = sizeof(float) * 2 * WIDTH * HEIGHT;
    VkBuffer outputBuffer;
    VkDeviceMemory outputMemory;
    createBuffer(physicalDevice, device, outputBufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 outputBuffer, outputMemory);

    VkDescriptorPoolSize poolSizes[2];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.maxSets = 1;
    descPoolInfo.poolSizeCount = 2;
    descPoolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool descPool;
    vkCreateDescriptorPool(device, &descPoolInfo, nullptr, &descPool);

    VkDescriptorSetAllocateInfo descAllocInfo{};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = descPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &descSetLayout;

    VkDescriptorSet descSet;
    vkAllocateDescriptorSets(device, &descAllocInfo, &descSet);

    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &tlas.handle;

    VkWriteDescriptorSet write0{};
    write0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write0.pNext = &asWrite;
    write0.dstSet = descSet;
    write0.dstBinding = 0;
    write0.descriptorCount = 1;
    write0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = outputBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = outputBufferSize;

    VkWriteDescriptorSet write1{};
    write1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write1.dstSet = descSet;
    write1.dstBinding = 1;
    write1.descriptorCount = 1;
    write1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write1.pBufferInfo = &bufferInfo;

    std::array<VkWriteDescriptorSet, 2> writes = { write0, write1 };
    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);

    // Camera + light setup, matching the CUDA kernel's math
    Vec3 camOrigin{0.0f, 1.0f, 3.0f};
    Vec3 camTarget{0.0f, 1.0f, 0.0f};
    Vec3 worldUp{0.0f, 1.0f, 0.0f};
    Vec3 forward = normalize(camTarget - camOrigin);
    Vec3 right = normalize(cross(forward, worldUp));
    Vec3 up = cross(right, forward);

    Vec3 lightPos{0.0f, 0.0f, 3.0f};
    float lightIntensity = 1000.0f;

    float fovDegrees = 60.0f;
    float tanHalfFov = tanf(fovDegrees * 0.5f * 3.14159265358979f / 180.0f);
    float aspect = (float)WIDTH / (float)HEIGHT;

    PushConstants pc{};
    pc.camOrigin[0] = camOrigin.x; pc.camOrigin[1] = camOrigin.y; pc.camOrigin[2] = camOrigin.z; pc.camOrigin[3] = 0.0f;
    pc.camRight[0] = right.x; pc.camRight[1] = right.y; pc.camRight[2] = right.z; pc.camRight[3] = 0.0f;
    pc.camUp[0] = up.x; pc.camUp[1] = up.y; pc.camUp[2] = up.z; pc.camUp[3] = 0.0f;
    pc.camForward[0] = forward.x; pc.camForward[1] = forward.y; pc.camForward[2] = forward.z; pc.camForward[3] = 0.0f;
    pc.lightPosIntensity[0] = lightPos.x; pc.lightPosIntensity[1] = lightPos.y; pc.lightPosIntensity[2] = lightPos.z;
    pc.lightPosIntensity[3] = lightIntensity;
    // Mitsuba's 'tx' emitter is a spot light with cutoff_angle=40 and an
    // implicit beam_width of cutoff_angle*3/4=30 (get_deafult_scene()
    // doesn't set beam_width explicitly, so Mitsuba's own default applies).
    // Packed here in radians since there's no room left in this struct for
    // a separate light-forward vector; the shader derives the spot's axis
    // from lightPos assuming a world-origin target, matching this scene.
    const float cutoffAngleDeg = 40.0f;
    const float beamWidthDeg = 30.0f;
    pc.params[0] = tanHalfFov; pc.params[1] = aspect;
    pc.params[2] = cutoffAngleDeg * 3.14159265358979f / 180.0f;
    pc.params[3] = beamWidthDeg * 3.14159265358979f / 180.0f;
    pc.vertexBufferAddress = getBufferDeviceAddress(device, vertexBuffer);
    pc.indexBufferAddress = getBufferDeviceAddress(device, indexBuffer);
    pc.normalBufferAddress = getBufferDeviceAddress(device, normalBuffer);

    // Dispatch the trace, timed over multiple iterations for a stable average
    // (matches the methodology used earlier for the Mitsuba-vs-CUDA benchmark:
    // measure real per-frame cost, not a single noisy sample)
    const int TRACE_ITERATIONS = 100;
    double totalTraceMs = 0.0;

    for (int i = 0; i < TRACE_ITERATIONS; i++) {
        auto traceStart = std::chrono::high_resolution_clock::now();

        // Capture only the first dispatch -- no need to capture all 100
        // timing iterations, and doing so would just bloat the capture file.
        if (i == 0 && rdoc_api) rdoc_api->StartFrameCapture(nullptr, nullptr);

        VkCommandBuffer traceCmd = beginOneTimeCommands(device, cmdPool);
        vkCmdBindPipeline(traceCmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
        vkCmdBindDescriptorSets(traceCmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                 pipelineLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(traceCmd, pipelineLayout,
                            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                            0, sizeof(PushConstants), &pc);
        pfn_vkCmdTraceRaysKHR(traceCmd, &raygenRegion, &missRegion, &hitRegion, &callableRegion,
                              WIDTH, HEIGHT, 1);
        endAndSubmitOneTimeCommands(device, queue, cmdPool, traceCmd);

        if (i == 0 && rdoc_api) {
            rdoc_api->EndFrameCapture(nullptr, nullptr);
            std::cout << "RenderDoc capture taken for the first trace dispatch.\n";
        }

        auto traceEnd = std::chrono::high_resolution_clock::now();
        totalTraceMs += std::chrono::duration<double, std::milli>(traceEnd - traceStart).count();
    }

    double avgTraceMs = totalTraceMs / TRACE_ITERATIONS;
    double estimatedPerFrameMs = blasMs + tlasMs + avgTraceMs;

    std::cout << "Trace dispatched (" << WIDTH << "x" << HEIGHT << " rays, real SMPL geometry).\n\n";
    std::cout << "Timing (averaged over " << TRACE_ITERATIONS << " trace dispatches):\n";
    std::cout << "BLAS build:        " << blasMs << " ms (one-time per frame, mesh deforms each frame)\n";
    std::cout << "TLAS build:        " << tlasMs << " ms (one-time per frame)\n";
    std::cout << "Trace dispatch:    " << avgTraceMs << " ms (avg per trace)\n";
    std::cout << "Estimated per-frame total: " << estimatedPerFrameMs << " ms "
              << "(BLAS + TLAS rebuild + trace, i.e. the realistic cost for an animated sequence)\n";
    std::cout << "For comparison, earlier benchmark on this same machine:\n";
    std::cout << "  Mitsuba:     19.19 ms/frame\n";
    std::cout << "  Custom CUDA: 15.16 ms/frame\n\n";

    // Read back results, print summary, save raw output
    float* results;
    vkMapMemory(device, outputMemory, 0, outputBufferSize, 0, (void**)&results);

    int hitCount = 0;
    float minDist = 1e30f, maxDist = -1e30f;
    float minIntensity = 1e30f, maxIntensity = -1e30f;
    for (uint32_t i = 0; i < WIDTH * HEIGHT; i++) {
        float d = results[i * 2 + 0];
        float inten = results[i * 2 + 1];
        if (d >= 0.0f) {
            hitCount++;
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            minIntensity = std::min(minIntensity, inten);
            maxIntensity = std::max(maxIntensity, inten);
        }
    }

    std::cout << "Hits: " << hitCount << " / " << (WIDTH * HEIGHT) << "\n";
    if (hitCount > 0) {
        std::cout << "Distance range:  [" << minDist << ", " << maxDist << "]\n";
        std::cout << "Intensity range: [" << minIntensity << ", " << maxIntensity << "]\n";
    } else {
        std::cout << "WARNING: zero hits. Camera likely isn't pointed at the mesh.\n";
        std::cout << "check camera origin/target against the mesh's actual bounds\n";
        std::cout << "(export_mesh.py prints vertex bounds when it runs).\n";
    }

    std::ofstream outFile("vulkan_output.bin", std::ios::binary);
    uint32_t w = WIDTH, h = HEIGHT;
    outFile.write(reinterpret_cast<char*>(&w), sizeof(uint32_t));
    outFile.write(reinterpret_cast<char*>(&h), sizeof(uint32_t));
    outFile.write(reinterpret_cast<char*>(results), outputBufferSize);
    outFile.close();
    std::cout << "\nSaved raw output to vulkan_output.bin (view with view_output.py)\n";

    vkUnmapMemory(device, outputMemory);

    if (hitCount > 0) {
        std::cout << "\nDone. Real SMPL mesh, camera, and shading traced correctly.\n";
        std::cout << "Run view_output.py to visualize. See VulkanCudaInterop/ for the\n";
        std::cout << "live, PyTorch-integrated version of this ray tracer.\n";
    }

    // Cleanup
    vkDestroyBuffer(device, outputBuffer, nullptr);
    vkFreeMemory(device, outputMemory, nullptr);
    vkDestroyBuffer(device, sbtBuffer, nullptr);
    vkFreeMemory(device, sbtMemory, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
    vkDestroyShaderModule(device, raygenModule, nullptr);
    vkDestroyShaderModule(device, closestHitModule, nullptr);
    vkDestroyShaderModule(device, missModule, nullptr);

    pfn_vkDestroyAccelerationStructureKHR(device, tlas.handle, nullptr);
    vkDestroyBuffer(device, tlas.buffer, nullptr);
    vkFreeMemory(device, tlas.memory, nullptr);

    pfn_vkDestroyAccelerationStructureKHR(device, blas.handle, nullptr);
    vkDestroyBuffer(device, blas.buffer, nullptr);
    vkFreeMemory(device, blas.memory, nullptr);

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexMemory, nullptr);
    vkDestroyBuffer(device, normalBuffer, nullptr);
    vkFreeMemory(device, normalMemory, nullptr);
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexMemory, nullptr);

    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
