#version 460
#extension GL_EXT_ray_tracing : require

// Paired with closesthit.rchit's shadow ray. The shadow ray is traced with
// gl_RayFlagsSkipClosestHitShaderEXT, so if it hits anything, no shader
// runs and the payload keeps whatever the caller initialized it to
// (true = occluded, set in closesthit.rchit before tracing). This shader
// only runs when the ray hits nothing at all, i.e. the light is visible.
layout(location = 1) rayPayloadInEXT bool shadowed;

void main() {
    shadowed = false;
}
