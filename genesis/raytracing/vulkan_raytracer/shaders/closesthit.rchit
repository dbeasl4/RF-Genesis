#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Lambertian shading matching the CUDA kernel's math (N.L, inverse-square
// falloff). Fetches the hit triangle's vertices via buffer_reference to
// compute a normal, since the hardware only gives us gl_PrimitiveID.
// No shadow ray yet. See README for details.

layout(buffer_reference, scalar) buffer Vertices { vec3 v[]; };
layout(buffer_reference, scalar) buffer Indices { uint i[]; };

layout(push_constant) uniform PushConstants {
    vec4 camOrigin;
    vec4 camRight;
    vec4 camUp;
    vec4 camForward;
    vec4 lightPosIntensity;
    vec4 params;
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
} pc;

struct RayPayload { float distance; float intensity; };
layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    Vertices vertices = Vertices(pc.vertexBufferAddress);
    Indices indices = Indices(pc.indexBufferAddress);

    uint i0 = indices.i[gl_PrimitiveID * 3 + 0];
    uint i1 = indices.i[gl_PrimitiveID * 3 + 1];
    uint i2 = indices.i[gl_PrimitiveID * 3 + 2];

    vec3 v0 = vertices.v[i0];
    vec3 v1 = vertices.v[i1];
    vec3 v2 = vertices.v[i2];

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 normal = normalize(cross(edge1, edge2));

    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Flip the normal to face the camera same as the CUDA kernel does,
    // since Moller-Trumbore-style normals 
    // don't have a guaranteed consistent winding relative to the viewer.
    vec3 toCam = normalize(-gl_WorldRayDirectionEXT);
    if (dot(normal, toCam) < 0.0) {
        normal = -normal;
    }

    vec3 lightPos = pc.lightPosIntensity.xyz;
    float lightIntensity = pc.lightPosIntensity.w;

    vec3 toLight = lightPos - worldPos;
    float distToLight = length(toLight);
    vec3 lightDir = toLight / max(distToLight, 1e-6);

    float ndotl = max(dot(normal, lightDir), 0.0);
    float falloff = lightIntensity / max(distToLight * distToLight, 1e-4);

    // Match Mitsuba's diffuse BSDF: albedo/pi * N.L * irradiance.
    const float albedo = 0.8;
    const float PI = 3.14159265359;

    payload.distance = gl_HitTEXT;
    payload.intensity = (albedo / PI) * ndotl * falloff;
}
