#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Lambertian shading matching the CUDA kernel's math (N.L, inverse-square
// falloff). Fetches the hit triangle's per-vertex normals via
// buffer_reference and interpolates them using barycentric coordinates,
// for smooth (not faceted) shading across curved surfaces. No shadow ray
// (reverted a second time after a reproducible GPU hang in the live
// PyTorch-integrated path, this time with a real recursive traceRayEXT
// shadow ray -- same fault as the earlier ray-query attempt, only in
// this exact live/persistent dispatch context; see README for details).

layout(buffer_reference, scalar) buffer Normals { vec3 n[]; };
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
    uint64_t normalBufferAddress;
} pc;

struct RayPayload { float distance; float intensity; };
layout(location = 0) rayPayloadInEXT RayPayload payload;

// Barycentric coordinates of the hit within the triangle, filled in by
// the hardware triangle intersector.
hitAttributeEXT vec2 attribs;

void main() {
    Normals normalsBuf = Normals(pc.normalBufferAddress);
    Indices indices = Indices(pc.indexBufferAddress);

    uint i0 = indices.i[gl_PrimitiveID * 3 + 0];
    uint i1 = indices.i[gl_PrimitiveID * 3 + 1];
    uint i2 = indices.i[gl_PrimitiveID * 3 + 2];

    vec3 n0 = normalsBuf.n[i0];
    vec3 n1 = normalsBuf.n[i1];
    vec3 n2 = normalsBuf.n[i2];

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 normal = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

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

    // Spot light cone (Mitsuba's 'tx' emitter). See README for details.
    float cutoffRad = pc.params.z;
    float beamRad = pc.params.w;
    vec3 lightForward = normalize(vec3(0.0) - lightPos);
    vec3 pointDir = normalize(worldPos - lightPos);
    float cosTheta = clamp(dot(pointDir, lightForward), -1.0, 1.0);
    float theta = acos(cosTheta);
    float spotFalloff;
    if (theta >= cutoffRad) {
        spotFalloff = 0.0;
    } else if (theta <= beamRad) {
        spotFalloff = 1.0;
    } else {
        spotFalloff = (cutoffRad - theta) / max(cutoffRad - beamRad, 1e-6);
    }

    const float albedo = 0.8;
    const float PI = 3.14159265359;

    payload.distance = gl_HitTEXT;
    payload.intensity = (albedo / PI) * ndotl * falloff * spotFalloff;
}
