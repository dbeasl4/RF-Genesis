// raytracer_live.cpp: a persistent, Python-callable Vulkan ray tracer.
// Combines real hardware ray tracing (mesh, camera, Lambertian shading,
// validated against Mitsuba) with zero-copy Vulkan-CUDA memory sharing for
// the output buffer. 
//
// Exposes a `RayTracer` class to Python via pybind11, with a persistent
// Vulkan context (instance, device, pipeline, and buffers created once in
// the constructor and reused across many trace() calls).


#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <array>

// function pointers (loaded once, in the constructor)

static PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR = nullptr;
static PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR = nullptr;
static PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR = nullptr;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
static PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR = nullptr;
static PFN_vkCreateRayTracingPipelinesKHR pfn_vkCreateRayTracingPipelinesKHR = nullptr;
static PFN_vkGetRayTracingShaderGroupHandlesKHR pfn_vkGetRayTracingShaderGroupHandlesKHR = nullptr;
static PFN_vkCmdTraceRaysKHR pfn_vkCmdTraceRaysKHR = nullptr;
static PFN_vkGetMemoryFdKHR pfn_vkGetMemoryFdKHR = nullptr;

static void loadFunctions(VkDevice device) {
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
    pfn_vkGetMemoryFdKHR =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");

    if (!pfn_vkGetAccelerationStructureBuildSizesKHR || !pfn_vkCreateAccelerationStructureKHR ||
        !pfn_vkCmdBuildAccelerationStructuresKHR || !pfn_vkGetAccelerationStructureDeviceAddressKHR ||
        !pfn_vkDestroyAccelerationStructureKHR || !pfn_vkCreateRayTracingPipelinesKHR ||
        !pfn_vkGetRayTracingShaderGroupHandlesKHR || !pfn_vkCmdTraceRaysKHR || !pfn_vkGetMemoryFdKHR) {
        throw std::runtime_error("Failed to load one or more required Vulkan function pointers.");
    }
}

// small helpers (same as main.cpp / interop_test.cpp)

static uint32_t findMemoryType(VkPhysicalDevice pd, uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    throw std::runtime_error("No suitable GPU memory type found.");
}

static VkDeviceSize alignUp(VkDeviceSize size, VkDeviceSize alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

static void createBuffer(VkPhysicalDevice pd, VkDevice device, VkDeviceSize size,
                          VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                          VkBuffer& buffer, VkDeviceMemory& memory,
                          bool exportable = false) {
    VkExternalMemoryBufferCreateInfo extInfo{};
    extInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = exportable ? &extInfo : nullptr;
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer.");
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateFlagsInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    flagsInfo.pNext = exportable ? &exportInfo : nullptr;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &flagsInfo;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(pd, memReq.memoryTypeBits, props);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory.");
    }
    vkBindBufferMemory(device, buffer, memory, 0);
}

static VkDeviceAddress getBufferDeviceAddress(VkDevice device, VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}

static VkCommandBuffer beginOneTime(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

static void endAndSubmit(VkDevice device, VkQueue queue, VkCommandPool pool, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue); 
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

static VkShaderModule loadShaderModule(VkDevice device, const std::string& path) {
    std::vector<char> buf;
    {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Failed to open shader file: " + path);
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        buf.resize(size);
        fread(buf.data(), 1, size, f);
        fclose(f);
    }
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = buf.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(buf.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module: " + path);
    }
    return mod;
}

struct PushConstants {
    float camOrigin[4];
    float camRight[4];
    float camUp[4];
    float camForward[4];
    float lightPosIntensity[4];
    float params[4];
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
};

struct Vec3 {
    float x, y, z;
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};
static Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static float dot3(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 normalize3(const Vec3& a) {
    float len = sqrtf(dot3(a, a));
    return len > 1e-12f ? a * (1.0f / len) : a;
}

//  The persistent RayTracer class

class RayTracer {
public:
    RayTracer(torch::Tensor faces, int resolution, float fov, std::string shaderDir)
        : resolution_(resolution), fov_(fov), shaderDir_(shaderDir) {

        uint32_t triangleCount = (uint32_t)faces.size(0);
        // .needed because would cause seg fault if not
        // ran into this error
        auto facesAcc = faces.contiguous().to(torch::kInt32).cpu();

        // instance and device
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo instInfo{};
        instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instInfo.pApplicationInfo = &appInfo;
        if (vkCreateInstance(&instInfo, nullptr, &instance_) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan instance.");

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

        uint8_t vkUUID[VK_UUID_SIZE];
        physicalDevice_ = VK_NULL_HANDLE;
        for (const auto& d : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            VkPhysicalDeviceIDProperties idProps{};
            idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &idProps;
            vkGetPhysicalDeviceProperties2(d, &props2);
            physicalDevice_ = d;
            memcpy(vkUUID, idProps.deviceUUID, VK_UUID_SIZE);
            break;
        }
        if (physicalDevice_ == VK_NULL_HANDLE) throw std::runtime_error("No discrete GPU found.");

        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qfCount, qfs.data());
        int queueFamily = -1;
        for (uint32_t i = 0; i < qfCount; i++) if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { queueFamily = (int)i; break; }
        if (queueFamily == -1) throw std::runtime_error("No suitable queue family.");

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = queueFamily;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;

        VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeat{};
        bdaFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bdaFeat.bufferDeviceAddress = VK_TRUE;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat{};
        asFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        asFeat.accelerationStructure = VK_TRUE;
        asFeat.pNext = &bdaFeat;
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeat{};
        rtFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rtFeat.rayTracingPipeline = VK_TRUE;
        rtFeat.pNext = &asFeat;

        std::vector<const char*> deviceExts = {
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        };
        VkDeviceCreateInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        di.pNext = &rtFeat;
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = (uint32_t)deviceExts.size();
        di.ppEnabledExtensionNames = deviceExts.data();
        if (vkCreateDevice(physicalDevice_, &di, nullptr, &device_) != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device.");

        loadFunctions(device_);
        vkGetDeviceQueue(device_, queueFamily, 0, &queue_);

        VkCommandPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = queueFamily;
        vkCreateCommandPool(device_, &pi, nullptr, &cmdPool_);


        triangleCount_ = triangleCount;
        vertexCount_ = 0; // set on first update_pose()

        VkDeviceSize indexBufferSize = (VkDeviceSize)triangleCount * 3 * sizeof(uint32_t);
        createBuffer(physicalDevice_, device_, indexBufferSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     indexBuffer_, indexMemory_);
        void* imap;
        vkMapMemory(device_, indexMemory_, 0, indexBufferSize, 0, &imap);
        memcpy(imap, facesAcc.data_ptr<int32_t>(), indexBufferSize);
        vkUnmapMemory(device_, indexMemory_);

        // Vertex buffer capacity: allocate generously up front (SMPL is
        // always 6890 vertices, but keep this flexible)
        maxVertexCount_ = 20000;
        VkDeviceSize vertexBufferSize = (VkDeviceSize)maxVertexCount_ * 3 * sizeof(float);
        createBuffer(physicalDevice_, device_, vertexBufferSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_, vertexMemory_);

        // pipeline and shaders
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
        rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &rtProps;
        vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);

        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 2;
        li.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(device_, &li, nullptr, &descSetLayout_);

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        pcRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &descSetLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_);

        
        VkShaderModule raygenMod = loadShaderModule(device_, shaderDir_ + "/raygen.rgen.spv");
        VkShaderModule chitMod = loadShaderModule(device_, shaderDir_ + "/closesthit.rchit.spv");
        VkShaderModule missMod = loadShaderModule(device_, shaderDir_ + "/miss.rmiss.spv");
        VkShaderModule shadowMissMod = loadShaderModule(device_, shaderDir_ + "/shadow.rmiss.spv");

        std::array<VkPipelineShaderStageCreateInfo, 4> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        stages[0].module = raygenMod; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        stages[1].module = missMod; stages[1].pName = "main";
        stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        stages[2].module = chitMod; stages[2].pName = "main";
        stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[3].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        stages[3].module = shadowMissMod; stages[3].pName = "main";

        std::array<VkRayTracingShaderGroupCreateInfoKHR, 4> groups{};
        groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[0].generalShader = 0; groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
        groups[0].anyHitShader = VK_SHADER_UNUSED_KHR; groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;
        groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[1].generalShader = 1; groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
        groups[1].anyHitShader = VK_SHADER_UNUSED_KHR; groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;
        groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[2].generalShader = VK_SHADER_UNUSED_KHR; groups[2].closestHitShader = 2;
        groups[2].anyHitShader = VK_SHADER_UNUSED_KHR; groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;
        groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[3].generalShader = 3; groups[3].closestHitShader = VK_SHADER_UNUSED_KHR;
        groups[3].anyHitShader = VK_SHADER_UNUSED_KHR; groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

        VkRayTracingPipelineCreateInfoKHR rpci{};
        rpci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rpci.stageCount = 4; rpci.pStages = stages.data();
        rpci.groupCount = 4; rpci.pGroups = groups.data();
        // Primary ray (depth 1) + the shadow ray closesthit.rchit traces
        // for occlusion (depth 2).
        rpci.maxPipelineRayRecursionDepth = 2;
        rpci.layout = pipelineLayout_;
        if (pfn_vkCreateRayTracingPipelinesKHR(device_, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rpci, nullptr, &pipeline_) != VK_SUCCESS)
            throw std::runtime_error("Failed to create ray tracing pipeline.");

        vkDestroyShaderModule(device_, raygenMod, nullptr);
        vkDestroyShaderModule(device_, chitMod, nullptr);
        vkDestroyShaderModule(device_, missMod, nullptr);
        vkDestroyShaderModule(device_, shadowMissMod, nullptr);

        // shader binding table
        // handles[] order matches groups[] order: 0=raygen, 1=primary miss,
        // 2=hit group, 3=shadow miss.
        uint32_t handleSize = rtProps.shaderGroupHandleSize;
        uint32_t handleAligned = (uint32_t)alignUp(handleSize, rtProps.shaderGroupHandleAlignment);
        uint32_t baseAlign = rtProps.shaderGroupBaseAlignment;
        std::vector<uint8_t> handles(4 * handleSize);
        pfn_vkGetRayTracingShaderGroupHandlesKHR(device_, pipeline_, 0, 4, handles.size(), handles.data());

        raygenRegion_.stride = alignUp(handleAligned, baseAlign);
        raygenRegion_.size = raygenRegion_.stride;
        missRegion_.stride = handleAligned;
        // Two miss shaders now: primary (missIndex 0, raygen.rgen's trace)
        // and shadow (missIndex 1, closesthit.rchit's trace) -- their
        // order here must match those missIndex values exactly.
        missRegion_.size = alignUp(2 * handleAligned, baseAlign);
        hitRegion_.stride = handleAligned;
        hitRegion_.size = alignUp(handleAligned, baseAlign);
        VkDeviceSize sbtSize = raygenRegion_.size + missRegion_.size + hitRegion_.size;

        createBuffer(physicalDevice_, device_, sbtSize, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sbtBuffer_, sbtMemory_);
        uint8_t* sbtMap;
        vkMapMemory(device_, sbtMemory_, 0, sbtSize, 0, (void**)&sbtMap);
        memcpy(sbtMap, handles.data() + 0 * handleSize, handleSize);  // raygen
        memcpy(sbtMap + raygenRegion_.size + 0 * missRegion_.stride,
               handles.data() + 1 * handleSize, handleSize);          // primary miss (missIndex 0)
        memcpy(sbtMap + raygenRegion_.size + 1 * missRegion_.stride,
               handles.data() + 3 * handleSize, handleSize);          // shadow miss (missIndex 1)
        memcpy(sbtMap + raygenRegion_.size + missRegion_.size,
               handles.data() + 2 * handleSize, handleSize);          // hit group
        vkUnmapMemory(device_, sbtMemory_);

        VkDeviceAddress sbtAddr = getBufferDeviceAddress(device_, sbtBuffer_);
        raygenRegion_.deviceAddress = sbtAddr;
        missRegion_.deviceAddress = sbtAddr + raygenRegion_.size;
        hitRegion_.deviceAddress = sbtAddr + raygenRegion_.size + missRegion_.size;
        callableRegion_ = {};

        // EXPORTABLE output buffer (distance, intensity per pixel)
        outputBufferSize_ = (VkDeviceSize)resolution_ * resolution_ * 2 * sizeof(float);
        createBuffer(physicalDevice_, device_, outputBufferSize_,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     outputBuffer_, outputMemory_, /*exportable=*/true);

        VkMemoryGetFdInfoKHR fdInfo{};
        fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fdInfo.memory = outputMemory_;
        fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        if (pfn_vkGetMemoryFdKHR(device_, &fdInfo, &fd) != VK_SUCCESS)
            throw std::runtime_error("Failed to export output buffer memory.");

        int cudaDeviceCount = 0;
        cudaGetDeviceCount(&cudaDeviceCount);
        int matchedCudaDevice = -1;
        for (int i = 0; i < cudaDeviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if (memcmp(prop.uuid.bytes, vkUUID, VK_UUID_SIZE) == 0) { matchedCudaDevice = i; break; }
        }
        if (matchedCudaDevice == -1) throw std::runtime_error("No matching CUDA device found for Vulkan's physical device.");
        cudaDevice_ = matchedCudaDevice;
        cudaSetDevice(cudaDevice_);

        cudaExternalMemoryHandleDesc extMemDesc{};
        extMemDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        extMemDesc.handle.fd = fd;
        extMemDesc.size = outputBufferSize_;
        if (cudaImportExternalMemory(&cudaExtMem_, &extMemDesc) != cudaSuccess)
            throw std::runtime_error("cudaImportExternalMemory failed.");

        cudaExternalMemoryBufferDesc bufDesc{};
        bufDesc.offset = 0;
        bufDesc.size = outputBufferSize_;
        if (cudaExternalMemoryGetMappedBuffer(&outputDevPtr_, cudaExtMem_, &bufDesc) != cudaSuccess)
            throw std::runtime_error("cudaExternalMemoryGetMappedBuffer failed.");

        // descriptor set (created once; TLAS handle gets updated per-frame)
        VkDescriptorPoolSize poolSizes[2];
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1; dpi.poolSizeCount = 2; dpi.pPoolSizes = poolSizes;
        vkCreateDescriptorPool(device_, &dpi, nullptr, &descPool_);

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = descPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &descSetLayout_;
        vkAllocateDescriptorSets(device_, &dai, &descSet_);

        // Bind the output buffer once (fixed for the object's lifetime).
        // The TLAS binding gets (re-)written in trace() each call, since
        // the acceleration structure handle changes when the mesh deforms.
        VkDescriptorBufferInfo bi2{};
        bi2.buffer = outputBuffer_;
        bi2.offset = 0;
        bi2.range = outputBufferSize_;
        VkWriteDescriptorSet w1{};
        w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1.dstSet = descSet_; w1.dstBinding = 1; w1.descriptorCount = 1;
        w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w1.pBufferInfo = &bi2;
        vkUpdateDescriptorSets(device_, 1, &w1, 0, nullptr);

        // Default camera/light, matching Mitsuba's get_deafult_scene()
        camOrigin_ = {0.0f, 1.0f, 3.0f};
        camTarget_ = {0.0f, 1.0f, 0.0f};
        lightPos_ = {0.0f, 0.0f, 3.0f};
        lightIntensity_ = 1000.0f;
    }

    ~RayTracer() {
        if (blas_.handle) { pfn_vkDestroyAccelerationStructureKHR(device_, blas_.handle, nullptr); vkDestroyBuffer(device_, blas_.buffer, nullptr); vkFreeMemory(device_, blas_.memory, nullptr); }
        if (tlas_.handle) { pfn_vkDestroyAccelerationStructureKHR(device_, tlas_.handle, nullptr); vkDestroyBuffer(device_, tlas_.buffer, nullptr); vkFreeMemory(device_, tlas_.memory, nullptr); }
        cudaDestroyExternalMemory(cudaExtMem_);
        vkDestroyBuffer(device_, outputBuffer_, nullptr);
        vkFreeMemory(device_, outputMemory_, nullptr);
        vkDestroyBuffer(device_, sbtBuffer_, nullptr);
        vkFreeMemory(device_, sbtMemory_, nullptr);
        vkDestroyDescriptorPool(device_, descPool_, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descSetLayout_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        vkDestroyBuffer(device_, indexBuffer_, nullptr);
        vkFreeMemory(device_, indexMemory_, nullptr);
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroyInstance(instance_, nullptr);
    }

    void update_sensor(std::vector<float> origin, std::vector<float> target) {
        camOrigin_ = {origin[0], origin[1], origin[2]};
        camTarget_ = {target[0], target[1], target[2]};
    }

    // vertices: CPU float32 tensor, shape [V, 3] Rebuilds the BLAS/TLAS every call, since the mesh deforms.
    void update_pose(torch::Tensor vertices) {
        auto v = vertices.contiguous().to(torch::kFloat32).cpu();
        vertexCount_ = (uint32_t)v.size(0);
        if (vertexCount_ > maxVertexCount_) throw std::runtime_error("Vertex count exceeds allocated buffer capacity.");

        void* map;
        vkMapMemory(device_, vertexMemory_, 0, (VkDeviceSize)vertexCount_ * 3 * sizeof(float), 0, &map);
        memcpy(map, v.data_ptr<float>(), (size_t)vertexCount_ * 3 * sizeof(float));
        vkUnmapMemory(device_, vertexMemory_);

        rebuildAccelerationStructures();
    }

    // Returns (PIR [H,W,3] float32 CUDA tensor, pointclouds [H*W,3] float32 CUDA tensor)
    std::tuple<torch::Tensor, torch::Tensor> trace() {
        if (vertexCount_ == 0) throw std::runtime_error("Call update_pose() before trace().");

        Vec3 forward = normalize3(camTarget_ - camOrigin_);
        Vec3 worldUp{0.0f, 1.0f, 0.0f};
        Vec3 right = normalize3(cross3(forward, worldUp));
        Vec3 up = cross3(right, forward);

        PushConstants pc{};
        pc.camOrigin[0] = camOrigin_.x; pc.camOrigin[1] = camOrigin_.y; pc.camOrigin[2] = camOrigin_.z;
        pc.camRight[0] = right.x; pc.camRight[1] = right.y; pc.camRight[2] = right.z;
        pc.camUp[0] = up.x; pc.camUp[1] = up.y; pc.camUp[2] = up.z;
        pc.camForward[0] = forward.x; pc.camForward[1] = forward.y; pc.camForward[2] = forward.z;
        pc.lightPosIntensity[0] = lightPos_.x; pc.lightPosIntensity[1] = lightPos_.y; pc.lightPosIntensity[2] = lightPos_.z;
        pc.lightPosIntensity[3] = lightIntensity_;
        float tanHalfFov = tanf(fov_ * 0.5f * 3.14159265358979f / 180.0f);
        pc.params[0] = tanHalfFov; pc.params[1] = 1.0f;
        // Spot light cone (see closesthit.rchit): radians, no room left in
        // this push-constant block for a separate forward vector, so the
        // shader derives the axis from lightPos assuming a world-origin
        // target, matching this scene's fixed setup.
        pc.params[2] = cutoffAngleDeg_ * 3.14159265358979f / 180.0f;
        pc.params[3] = beamWidthDeg_ * 3.14159265358979f / 180.0f;
        pc.vertexBufferAddress = getBufferDeviceAddress(device_, vertexBuffer_);
        pc.indexBufferAddress = getBufferDeviceAddress(device_, indexBuffer_);

        VkCommandBuffer cmd = beginOneTime(device_, cmdPool_);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout_, 0, 1, &descSet_, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(pc), &pc);
        pfn_vkCmdTraceRaysKHR(cmd, &raygenRegion_, &missRegion_, &hitRegion_, &callableRegion_, resolution_, resolution_, 1);
        endAndSubmit(device_, queue_, cmdPool_, cmd);

        // Output is already in CUDA-visible memory (outputDevPtr_, imported
        // once at construction)
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cudaDevice_);
        torch::Tensor raw = torch::from_blob(outputDevPtr_, {resolution_, resolution_, 2}, options).clone();
        // .clone() so the returned tensor is independent of this object's
        // buffer, which trace() will overwrite again on the next call.

        torch::Tensor distance = raw.select(2, 0);
        torch::Tensor intensity = raw.select(2, 1);
        // miss.rmiss writes -1 for a miss; remap to 0 to match Mitsuba's
        // t>9999->0 and the CUDA kernel's convention. Without this, missed
        // rays produce hit_pos = origin - dir (a point behind the camera)
        // instead of being excluded like the other two backends.
        torch::Tensor missMask = distance < 0;
        distance = torch::where(missMask, torch::zeros_like(distance), distance);
        intensity = torch::where(missMask, torch::zeros_like(intensity), intensity);
        torch::Tensor velocity = torch::zeros_like(distance);
        torch::Tensor PIR = torch::stack({distance, intensity, velocity}, 2);

        // Reconstruct world-space hit position per pixel from distance and
        // ray direction (same math as the CUDA wrapper), computed on GPU.
        auto arangeOpts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, cudaDevice_);
        torch::Tensor xs = (torch::arange(resolution_, arangeOpts) + 0.5) / resolution_ * 2.0 - 1.0;
        torch::Tensor ys = (torch::arange(resolution_, arangeOpts) + 0.5) / resolution_ * 2.0 - 1.0;
        auto grids = torch::meshgrid({xs, ys}, "xy");
        torch::Tensor gridX = grids[0] * tanHalfFov;
        torch::Tensor gridY = grids[1] * tanHalfFov;

        torch::Tensor rightT = torch::tensor({right.x, right.y, right.z}, arangeOpts);
        torch::Tensor upT = torch::tensor({up.x, up.y, up.z}, arangeOpts);
        torch::Tensor fwdT = torch::tensor({forward.x, forward.y, forward.z}, arangeOpts);
        torch::Tensor originT = torch::tensor({camOrigin_.x, camOrigin_.y, camOrigin_.z}, arangeOpts);

        torch::Tensor dirs = gridX.unsqueeze(-1) * rightT + gridY.unsqueeze(-1) * upT + fwdT;
        dirs = torch::nn::functional::normalize(dirs, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        torch::Tensor hitPos = originT + dirs * distance.unsqueeze(-1);
        torch::Tensor pointclouds = hitPos.reshape({-1, 3});

        return {PIR, pointclouds};
    }

private:
    struct AccelStruct {
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceAddress deviceAddress = 0;
    };

    void rebuildAccelerationStructures() {
        // Destroy last frame's BLAS/TLAS before building this frame's, since
        // the mesh deforms every call and we don't reuse acceleration structures.
        if (blas_.handle) { pfn_vkDestroyAccelerationStructureKHR(device_, blas_.handle, nullptr); vkDestroyBuffer(device_, blas_.buffer, nullptr); vkFreeMemory(device_, blas_.memory, nullptr); blas_ = {}; }
        if (tlas_.handle) { pfn_vkDestroyAccelerationStructureKHR(device_, tlas_.handle, nullptr); vkDestroyBuffer(device_, tlas_.buffer, nullptr); vkFreeMemory(device_, tlas_.memory, nullptr); tlas_ = {}; }

        // BLAS
        VkAccelerationStructureGeometryTrianglesDataKHR tri{};
        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = getBufferDeviceAddress(device_, vertexBuffer_);
        tri.vertexStride = sizeof(float) * 3;
        tri.maxVertex = vertexCount_ - 1;
        tri.indexType = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = getBufferDeviceAddress(device_, indexBuffer_);

        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.geometry.triangles = tri;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        VkAccelerationStructureBuildGeometryInfoKHR bi{};
        bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1; bi.pGeometries = &geom;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        pfn_vkGetAccelerationStructureBuildSizesKHR(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &triangleCount_, &sizeInfo);

        createBuffer(physicalDevice_, device_, sizeInfo.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     blas_.buffer, blas_.memory);
        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = blas_.buffer; ci.size = sizeInfo.accelerationStructureSize;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        pfn_vkCreateAccelerationStructureKHR(device_, &ci, nullptr, &blas_.handle);

        VkBuffer scratchBuf; VkDeviceMemory scratchMem;
        createBuffer(physicalDevice_, device_, sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuf, scratchMem);
        bi.dstAccelerationStructure = blas_.handle;
        bi.scratchData.deviceAddress = getBufferDeviceAddress(device_, scratchBuf);
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = triangleCount_;
        const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = &range;
        VkCommandBuffer cmd = beginOneTime(device_, cmdPool_);
        pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bi, &rangePtr);
        endAndSubmit(device_, queue_, cmdPool_, cmd);
        vkDestroyBuffer(device_, scratchBuf, nullptr);
        vkFreeMemory(device_, scratchMem, nullptr);

        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = blas_.handle;
        blas_.deviceAddress = pfn_vkGetAccelerationStructureDeviceAddressKHR(device_, &addrInfo);

        // TLAS
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0] = 1.0f; inst.transform.matrix[1][1] = 1.0f; inst.transform.matrix[2][2] = 1.0f;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas_.deviceAddress;

        VkBuffer instBuf; VkDeviceMemory instMem;
        createBuffer(physicalDevice_, device_, sizeof(inst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, instBuf, instMem);
        void* imap;
        vkMapMemory(device_, instMem, 0, sizeof(inst), 0, &imap);
        memcpy(imap, &inst, sizeof(inst));
        vkUnmapMemory(device_, instMem);

        VkAccelerationStructureGeometryInstancesDataKHR instData{};
        instData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instData.data.deviceAddress = getBufferDeviceAddress(device_, instBuf);

        VkAccelerationStructureGeometryKHR tGeom{};
        tGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tGeom.geometry.instances = instData;

        VkAccelerationStructureBuildGeometryInfoKHR tbi{};
        tbi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tbi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tbi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tbi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tbi.geometryCount = 1; tbi.pGeometries = &tGeom;

        uint32_t instCount = 1;
        VkAccelerationStructureBuildSizesInfoKHR tSizeInfo{};
        tSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        pfn_vkGetAccelerationStructureBuildSizesKHR(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tbi, &instCount, &tSizeInfo);

        createBuffer(physicalDevice_, device_, tSizeInfo.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     tlas_.buffer, tlas_.memory);
        VkAccelerationStructureCreateInfoKHR tci{};
        tci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        tci.buffer = tlas_.buffer; tci.size = tSizeInfo.accelerationStructureSize;
        tci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        pfn_vkCreateAccelerationStructureKHR(device_, &tci, nullptr, &tlas_.handle);

        VkBuffer tScratchBuf; VkDeviceMemory tScratchMem;
        createBuffer(physicalDevice_, device_, tSizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tScratchBuf, tScratchMem);
        tbi.dstAccelerationStructure = tlas_.handle;
        tbi.scratchData.deviceAddress = getBufferDeviceAddress(device_, tScratchBuf);
        VkAccelerationStructureBuildRangeInfoKHR tRange{};
        tRange.primitiveCount = instCount;
        const VkAccelerationStructureBuildRangeInfoKHR* tRangePtr = &tRange;
        VkCommandBuffer tCmd = beginOneTime(device_, cmdPool_);
        pfn_vkCmdBuildAccelerationStructuresKHR(tCmd, 1, &tbi, &tRangePtr);
        endAndSubmit(device_, queue_, cmdPool_, tCmd);
        vkDestroyBuffer(device_, tScratchBuf, nullptr);
        vkFreeMemory(device_, tScratchMem, nullptr);
        vkDestroyBuffer(device_, instBuf, nullptr);
        vkFreeMemory(device_, instMem, nullptr);

        // Update the descriptor set's TLAS binding to point at this frame's TLAS
        VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
        asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asWrite.accelerationStructureCount = 1;
        asWrite.pAccelerationStructures = &tlas_.handle;
        VkWriteDescriptorSet w0{};
        w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0.pNext = &asWrite;
        w0.dstSet = descSet_; w0.dstBinding = 0; w0.descriptorCount = 1;
        w0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        vkUpdateDescriptorSets(device_, 1, &w0, 0, nullptr);
    }

    int resolution_;
    float fov_;
    std::string shaderDir_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    VkBuffer indexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory_ = VK_NULL_HANDLE;
    uint32_t vertexCount_, maxVertexCount_, triangleCount_;

    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;

    VkBuffer sbtBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory sbtMemory_ = VK_NULL_HANDLE;
    VkStridedDeviceAddressRegionKHR raygenRegion_{}, missRegion_{}, hitRegion_{}, callableRegion_{};

    VkBuffer outputBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory outputMemory_ = VK_NULL_HANDLE;
    VkDeviceSize outputBufferSize_;
    cudaExternalMemory_t cudaExtMem_;
    void* outputDevPtr_ = nullptr;
    int cudaDevice_ = 0;

    AccelStruct blas_, tlas_;

    Vec3 camOrigin_, camTarget_, lightPos_;
    float lightIntensity_;
    // Mitsuba's 'tx' emitter is a spot light with cutoff_angle=40 and an
    // implicit beam_width of cutoff_angle*3/4=30 (get_deafult_scene()
    // doesn't set beam_width explicitly, so Mitsuba's own default applies).
    float cutoffAngleDeg_ = 40.0f;
    float beamWidthDeg_ = 30.0f;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<RayTracer>(m, "RayTracer")
        .def(py::init<torch::Tensor, int, float, std::string>(),
             py::arg("faces"), py::arg("resolution") = 128, py::arg("fov") = 60.0f, py::arg("shader_dir"))
        .def("update_sensor", &RayTracer::update_sensor)
        .def("update_pose", &RayTracer::update_pose)
        .def("trace", &RayTracer::trace);
}
