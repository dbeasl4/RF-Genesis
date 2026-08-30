#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload { float distance; float intensity; };
layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    payload.distance = -1.0;
    payload.intensity = 0.0;
}
