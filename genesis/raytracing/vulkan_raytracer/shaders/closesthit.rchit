#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Lambertian shading matching the CUDA kernel's math (N.L, inverse-square
// falloff). Fetches the hit triangle's per-vertex normals via
// buffer_reference and interpolates them using barycentric coordinates,
// for smooth (not faceted) shading across curved surfaces. Shadow ray via
// a real recursive traceRayEXT call (not a ray query -- see README for
// why). See README for more.

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

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

// Payload for the shadow ray THIS shader traces (location 1, distinct
// from the location-0 payload we ourselves were invoked with above).
layout(location = 1) rayPayloadEXT float shadowPayload;

// Barycentric coordinates of the hit within the triangle, filled in by
// the hardware triangle intersector (not something we compute ourselves).
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

    // attribs = (u, v); the weight on vertex 0 is what's left over.
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 normal = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Flip the normal to face the camera, same as the CUDA kernel does.
    // With proper vertex normals this should rarely trigger, but kept as
    // a safety net in case any normals ended up facing inward.
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

    // Shadow ray: trace from this point toward the light. Assume occluded
    // (1.0) before tracing; shadow.rmiss clears it to 0.0 if the ray hits
    // nothing. gl_RayFlagsSkipClosestHitShaderEXT means an actual hit
    // just leaves the payload at 1.0 rather than needing its own
    // closest-hit shader -- we only need to know hit-or-miss, not what
    // was hit. Offset the origin along the normal to avoid immediately
    // self-intersecting the same triangle.
    bool occluded = false;
    if (ndotl > 0.0) {
        float epsilon = 0.01;
        vec3 shadowOrigin = worldPos + normal * epsilon;
        shadowPayload = 1.0;
        traceRayEXT(topLevelAS,
                    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                    0xFF, 0, 0, 1,
                    shadowOrigin, epsilon, lightDir, distToLight - epsilon * 2.0,
                    1);
        occluded = (shadowPayload > 0.5);
    }

    float shadowedNdotL = occluded ? 0.0 : ndotl;
    float falloff = lightIntensity / max(distToLight * distToLight, 1e-4);

    // Mitsuba's 'tx' emitter (pathtracer.py's get_deafult_scene()) is a
    // spot light, not an isotropic point light: linear angular falloff
    // from full intensity within beam_width out to 0 at cutoff_angle.
    // pc.params.z/w carry these as radians (packed there since the
    // push-constant block has no room left for a separate forward vector).
    // The spot targets the world origin in this scene, so its axis can be
    // derived from lightPos alone -- same hardcoded-scene assumption
    // already used for the default camera/light positions in this path.
    float cutoffRad = pc.params.z;
    float beamRad = pc.params.w;
    vec3 lightForward = normalize(vec3(0.0) - lightPos);
    vec3 pointDir = normalize(worldPos - lightPos);  // light -> surface point
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

    // Match Mitsuba's diffuse BSDF: albedo/pi * N.L * irradiance.
    // See README for why this matters.
    const float albedo = 0.8;
    const float PI = 3.14159265359;

    payload.distance = gl_HitTEXT;
    payload.intensity = (albedo / PI) * shadowedNdotL * falloff * spotFalloff;
}

