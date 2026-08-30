// interop_test.cpp: verifies Vulkan and CUDA/PyTorch can share GPU
// memory directly (zero-copy), before relying on it in the real ray
// tracer. Tested on an NVIDIA RTX 4060 Laptop GPU.
//
// Vulkan writes known values into an exportable buffer, exports it as a
// file descriptor, CUDA imports it, and the result comes back as a
// torch::Tensor checked in Python.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>

PFN_vkGetMemoryFdKHR pfn_vkGetMemoryFdKHR = nullptr;

void loadExternalMemoryFunctions(VkDevice device) {
    pfn_vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!pfn_vkGetMemoryFdKHR) {
        throw std::runtime_error("Failed to load vkGetMemoryFdKHR.");
    }
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

torch::Tensor test_interop() {
    const int N = 16;
    VkDeviceSize bufferSize = N * sizeof(float);

    // 1. Match Vulkan's device to CUDA's by UUID (matters on multi-GPU systems)
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint8_t vkDeviceUUID[VK_UUID_SIZE];

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

        physicalDevice = d;
        memcpy(vkDeviceUUID, idProps.deviceUUID, VK_UUID_SIZE);
        break; // first discrete GPU, fine on a single-dGPU machine
    }
    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("No discrete GPU found.");
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    int queueFamily = -1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { queueFamily = (int)i; break; }
    }
    if (queueFamily == -1) {
        throw std::runtime_error("No suitable queue family found.");
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    std::vector<const char*> deviceExtensions = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device.");
    }

    loadExternalMemoryFunctions(device);

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily;
    VkCommandPool cmdPool;
    vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool);

    // 2. Create an exportable, device-local buffer
    VkExternalMemoryBufferCreateInfo extBufferInfo{};
    extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = &extBufferInfo;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create exportable buffer.");
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportInfo;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDeviceMemory memory;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate exportable memory.");
    }
    vkBindBufferMemory(device, buffer, memory, 0);

    // 3. Write known test values via the GPU (vkCmdUpdateBuffer)
    float testValues[N];
    for (int i = 0; i < N; i++) testValues[i] = (float)i;

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    vkCmdUpdateBuffer(cmd, buffer, 0, bufferSize, testValues);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);

    // Full queue wait (not async semaphores).
    vkQueueWaitIdle(queue);

    // 4. Export the memory as a file descriptor
    VkMemoryGetFdInfoKHR fdInfo{};
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    if (pfn_vkGetMemoryFdKHR(device, &fdInfo, &fd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to export Vulkan memory as a file descriptor.");
    }

    // 5. Import into CUDA on the matching physical device
    int cudaDeviceCount = 0;
    cudaGetDeviceCount(&cudaDeviceCount);
    int matchedCudaDevice = -1;
    for (int i = 0; i < cudaDeviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if (memcmp(prop.uuid.bytes, vkDeviceUUID, VK_UUID_SIZE) == 0) {
            matchedCudaDevice = i;
            break;
        }
    }
    if (matchedCudaDevice == -1) {
        throw std::runtime_error("Could not find a CUDA device matching the Vulkan physical device's UUID.");
    }
    cudaSetDevice(matchedCudaDevice);

    cudaExternalMemoryHandleDesc extMemDesc{};
    extMemDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    extMemDesc.handle.fd = fd; // CUDA takes ownership of the fd on successful import
    extMemDesc.size = memReq.size;

    cudaExternalMemory_t cudaExtMem;
    cudaError_t err = cudaImportExternalMemory(&cudaExtMem, &extMemDesc);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaImportExternalMemory failed: ") + cudaGetErrorString(err));
    }

    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size = bufferSize;

    void* devPtr = nullptr;
    err = cudaExternalMemoryGetMappedBuffer(&devPtr, cudaExtMem, &bufDesc);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaExternalMemoryGetMappedBuffer failed: ") + cudaGetErrorString(err));
    }

    // 6. Wrap as a torch tensor and verify
    // .clone() so the tensor survives after this throwaway context is torn down.
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, matchedCudaDevice);
    torch::Tensor result = torch::from_blob(devPtr, {N}, options).clone();

    // Cleanup
    cudaDestroyExternalMemory(cudaExtMem);
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_interop", &test_interop,
          "Vulkan writes known values to an exportable buffer; CUDA imports it via external memory; "
          "returns the result as a torch tensor to verify the values match.");
}