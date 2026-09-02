#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Lambertian shading matching the CUDA kernel's math (N.L, inverse-square
// falloff, spot cone, shadow ray). Fetches the hit triangle's vertices via
// buffer_reference to compute a normal, since the hardware only gives us
// gl_PrimitiveID.

layout(buffer_reference, scalar) buffer Vertices { vec3 v[]; };
layout(buffer_reference, scalar) buffer Indices { uint i[]; };

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

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
layout(location = 1) rayPayloadEXT bool shadowed;

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

    // Shadow ray: Mitsuba's 'direct' integrator traces a real
    // visibility/shadow ray as part of sampling direct illumination, so
    // points occluded by other parts of the body come out dark. Without
    // this, every front-facing point got full unoccluded light regardless
    // of what's between it and the light -- verified against Mitsuba: mean
    // intensity over hit pixels came out 60-130% too high with this
    // omitted, while max (rarely self-shadowed) stayed close.
    //
    // gl_RayFlagsSkipClosestHitShaderEXT + gl_RayFlagsTerminateOnFirstHitEXT
    // means this is a cheap boolean occlusion query, not a second full
    // shaded hit: if it hits anything, no shader runs and `shadowed`
    // keeps the `true` we set below; shadow.rmiss only runs (setting it
    // false) if the ray reaches the light unobstructed.
    vec3 shadowOrigin = worldPos + normal * 1e-3;
    shadowed = true;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT,
        0xFF,
        0, 0, 1,  // missIndex 1 -> shadow.rmiss (index within the SBT's miss region)
        shadowOrigin,
        0.0,
        lightDir,
        max(distToLight - 1e-3, 0.0),
        1
    );
    float shadow = shadowed ? 0.0 : 1.0;

    payload.distance = gl_HitTEXT;
    payload.intensity = (albedo / PI) * ndotl * falloff * spotFalloff * shadow;
}
