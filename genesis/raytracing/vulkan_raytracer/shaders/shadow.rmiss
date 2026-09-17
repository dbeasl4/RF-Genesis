#version 460
#extension GL_EXT_ray_tracing : require

// Shadow ray miss shader: reached only when the shadow ray hits nothing
// between the shading point and the light, meaning that point is NOT
// occluded. The closest-hit shader initializes shadowPayload to 1.0
// ("assume occluded") before tracing; this shader is what clears it to
// 0.0 if the path to the light is actually clear.

layout(location = 1) rayPayloadInEXT float shadowPayload;

void main() {
    shadowPayload = 0.0;
}
