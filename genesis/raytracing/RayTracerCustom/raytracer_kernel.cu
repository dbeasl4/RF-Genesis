// raytracer_kernel.cu
//
// Brute-force GPU ray tracer replicating what RF-Genesis's pathtracer.py
// does via Mitsuba: cast one ray per pixel from a perspective camera,
// find the closest triangle hit against the SMPL body mesh, and shade it
// with a single point/spot light (no bounces, no shadow ray yet).
//
// Intentionally simple: for ~13.7k triangles at 128x128, brute force (no
// BVH) is fast enough on a modern GPU and easier to start with than an
// acceleration structure.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define EPSILON 1e-8f

// small float3 helpers 

__host__ __device__ __forceinline__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ __forceinline__ float3 sub3(float3 a, float3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ __forceinline__ float3 add3(float3 a, float3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ __forceinline__ float3 scale3(float3 a, float s) { return f3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ __forceinline__ float  dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ __forceinline__ float3 normalize3(float3 a) {
    float len = sqrtf(dot3(a, a));
    return len > 1e-12f ? scale3(a, 1.0f / len) : a;
}
// Moller-Trumbore ray-triangle intersection

__device__ __forceinline__ bool ray_triangle_intersect(
    float3 orig, float3 dir,
    float3 v0, float3 v1, float3 v2,
    float& t_out, float3& normal_out)
{
    float3 edge1 = sub3(v1, v0);
    float3 edge2 = sub3(v2, v0);
    float3 h = cross3(dir, edge2);
    float a = dot3(edge1, h);
    if (fabsf(a) < EPSILON) return false; // ray parallel to triangle

    float f = 1.0f / a;
    float3 s = sub3(orig, v0);
    float u = f * dot3(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross3(s, edge1);
    float v = f * dot3(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * dot3(edge2, q);
    if (t <= EPSILON) return false;

    t_out = t;
    normal_out = normalize3(cross3(edge1, edge2));
    return true;
}

// main kernel

__global__ void raytrace_kernel(
    const float* __restrict__ vertices, 
    const int*   __restrict__ faces,
    int num_faces,
    float3 cam_origin,
    float3 cam_right, float3 cam_up, float3 cam_forward,  // orthonormal camera basis
    float tan_half_fov,
    int width, int height,
    float3 light_pos,
    float light_intensity,
    float3 light_forward,  // normalized spot axis, light_pos -> light_target
    float cutoff_rad,      // Mitsuba 'cutoff_angle': falloff reaches 0 here
    float beam_rad,        // Mitsuba 'beam_width': full intensity within this angle
    float* __restrict__ out_distance,
    float* __restrict__ out_intensity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Build camera ray for this pixel (standard perspective camera)
    float aspect = (float)width / (float)height;
    float ndc_x = (2.0f * (x + 0.5f) / width - 1.0f) * tan_half_fov * aspect;
    float ndc_y = (1.0f - 2.0f * (y + 0.5f) / height) * tan_half_fov;

    float3 dir = add3(add3(scale3(cam_right, ndc_x), scale3(cam_up, ndc_y)), cam_forward);
    dir = normalize3(dir);

    float closest_t = 1e30f;
    float3 closest_normal = f3(0.0f, 0.0f, 0.0f);
    bool hit = false;

    // Brute-force: test every triangle. Fine for ~14k triangles at 128x128;
    // revisit with a BVH if you scale resolution or add more geometry.
    for (int f = 0; f < num_faces; f++) {
        int i0 = faces[f * 3 + 0];
        int i1 = faces[f * 3 + 1];
        int i2 = faces[f * 3 + 2];

        float3 v0 = f3(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        float3 v1 = f3(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        float3 v2 = f3(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        float t;
        float3 n;
        if (ray_triangle_intersect(cam_origin, dir, v0, v1, v2, t, n) && t < closest_t) {
            closest_t = t;
            closest_normal = n;
            hit = true;
        }
    }

    int idx = y * width + x;
    if (hit) {
        float3 hit_pos = add3(cam_origin, scale3(dir, closest_t));
        float3 to_light = sub3(light_pos, hit_pos);
        float dist_to_light = sqrtf(dot3(to_light, to_light));
        float3 light_dir = scale3(to_light, 1.0f / fmaxf(dist_to_light, 1e-6f));

        // Flip normal to face the camera using Moller
        float3 to_cam = normalize3(sub3(cam_origin, hit_pos));
        if (dot3(closest_normal, to_cam) < 0.0f) {
            closest_normal = scale3(closest_normal, -1.0f);
        }

        float ndotl = fmaxf(dot3(closest_normal, light_dir), 0.0f);
        float falloff = light_intensity / fmaxf(dist_to_light * dist_to_light, 1e-4f);

        // Shadow ray: Mitsuba's 'direct' integrator traces a real
        // visibility/shadow ray as part of sampling direct illumination,
        // so points on the body occluded by other parts of the body (the
        // underside of an arm, the back of a leg, etc.) come out dark.
        // Without this, every front-facing point gets full unoccluded
        // light regardless of what's between it and the light -- verified
        // against Mitsuba: mean intensity over hit pixels came out
        // 60-130% too high with this omitted, while max (rarely
        // self-shadowed) stayed close.
        //
        // Brute-force like the primary ray: fine for ~14k triangles;
        // revisit with a BVH/any-hit-only query if this doubles cost too
        // much at higher resolution.
        float3 shadow_origin = add3(hit_pos, scale3(closest_normal, 1e-3f));
        bool occluded = false;
        for (int f2 = 0; f2 < num_faces; f2++) {
            int j0 = faces[f2 * 3 + 0];
            int j1 = faces[f2 * 3 + 1];
            int j2 = faces[f2 * 3 + 2];
            float3 w0 = f3(vertices[j0 * 3 + 0], vertices[j0 * 3 + 1], vertices[j0 * 3 + 2]);
            float3 w1 = f3(vertices[j1 * 3 + 0], vertices[j1 * 3 + 1], vertices[j1 * 3 + 2]);
            float3 w2 = f3(vertices[j2 * 3 + 0], vertices[j2 * 3 + 1], vertices[j2 * 3 + 2]);
            float t2;
            float3 n2;
            // Only care whether *anything* blocks the light, and only
            // strictly between the surface and the light itself.
            if (ray_triangle_intersect(shadow_origin, light_dir, w0, w1, w2, t2, n2) &&
                t2 < dist_to_light - 1e-3f) {
                occluded = true;
                break;
            }
        }
        float shadow = occluded ? 0.0f : 1.0f;

        // Match Mitsuba's diffuse BSDF (albedo/pi * N.L * irradiance) and
        // the Vulkan closest-hit shader's shading, which both include this
        // factor. Without it this kernel is ~4x too bright relative to them.
        const float albedo = 0.8f;
        const float PI = 3.14159265359f;

        // Mitsuba's 'tx' emitter is a spot light, not an isotropic point
        // light: it has a cone with a linear angular falloff from
        // beam_width (full intensity) out to cutoff_angle (zero), per
        // https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_emitters.html.
        // Without this term this kernel over-brightens off-axis points,
        // increasingly so as the body gets closer (verified against
        // Mitsuba: the intensity gap grew from ~3% to ~16% across frames
        // as distance dropped from ~6.7 to ~3.3).
        float3 point_to_light = scale3(light_dir, -1.0f);  // light_pos -> hit_pos
        float cos_theta = dot3(point_to_light, light_forward);
        cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);
        float theta = acosf(cos_theta);
        float spot_falloff;
        if (theta >= cutoff_rad) {
            spot_falloff = 0.0f;
        } else if (theta <= beam_rad) {
            spot_falloff = 1.0f;
        } else {
            spot_falloff = (cutoff_rad - theta) / fmaxf(cutoff_rad - beam_rad, 1e-6f);
        }

        out_distance[idx] = closest_t;
        out_intensity[idx] = (albedo / PI) * ndotl * falloff * spot_falloff * shadow;
    } else {
        out_distance[idx] = 0.0f;
        out_intensity[idx] = 0.0f;
    }
}

// host-side launcher, exposed to Python

std::vector<torch::Tensor> raytrace_cuda(
    // Just labeling data here for easier implementation

    // float32 [num_verts, 3], CUDA
    torch::Tensor vertices,
    // int32 [num_faces, 3]
    torch::Tensor faces,
    // float32 [3]
    torch::Tensor cam_origin,
    // float32 [3], normalized
    torch::Tensor cam_right,
    torch::Tensor cam_up,
    torch::Tensor cam_forward,
    double fov_degrees,
    int64_t width,
    int64_t height,
    torch::Tensor light_pos,
    double light_intensity,
    torch::Tensor light_target,   // float32 [3]; spot light aims here
    double cutoff_angle_deg,
    double beam_width_deg)
{
    TORCH_CHECK(vertices.is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(faces.is_cuda(), "faces must be a CUDA tensor");
    TORCH_CHECK(vertices.dtype() == torch::kFloat32, "vertices must be float32");
    TORCH_CHECK(faces.dtype() == torch::kInt32, "faces must be int32");
    TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
    TORCH_CHECK(faces.is_contiguous(), "faces must be contiguous");

    int num_faces = faces.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(vertices.device());
    auto out_distance = torch::zeros({height, width}, options);
    auto out_intensity = torch::zeros({height, width}, options);

    
    auto co = cam_origin.cpu();
    auto cr = cam_right.cpu();
    auto cu = cam_up.cpu();
    auto cf = cam_forward.cpu();
    auto lp = light_pos.cpu();
    auto lt = light_target.cpu();

    float3 cam_origin_f  = f3(co[0].item<float>(), co[1].item<float>(), co[2].item<float>());
    float3 cam_right_f   = f3(cr[0].item<float>(), cr[1].item<float>(), cr[2].item<float>());
    float3 cam_up_f      = f3(cu[0].item<float>(), cu[1].item<float>(), cu[2].item<float>());
    float3 cam_forward_f = f3(cf[0].item<float>(), cf[1].item<float>(), cf[2].item<float>());
    float3 light_pos_f   = f3(lp[0].item<float>(), lp[1].item<float>(), lp[2].item<float>());
    float3 light_target_f = f3(lt[0].item<float>(), lt[1].item<float>(), lt[2].item<float>());
    float3 light_forward_f = normalize3(sub3(light_target_f, light_pos_f));

    float tan_half_fov = tanf((float)(fov_degrees * M_PI / 180.0) * 0.5f);
    float cutoff_rad = (float)(cutoff_angle_deg * M_PI / 180.0);
    float beam_rad   = (float)(beam_width_deg * M_PI / 180.0);

    dim3 block(16, 16);
    dim3 grid((unsigned int)((width + block.x - 1) / block.x),
              (unsigned int)((height + block.y - 1) / block.y));

    raytrace_kernel<<<grid, block>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        num_faces,
        cam_origin_f, cam_right_f, cam_up_f, cam_forward_f,
        tan_half_fov,
        (int)width, (int)height,
        light_pos_f, (float)light_intensity,
        light_forward_f, cutoff_rad, beam_rad,
        out_distance.data_ptr<float>(),
        out_intensity.data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return {out_distance, out_intensity};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("raytrace", &raytrace_cuda, "Brute-force CUDA ray tracer (Moller-Trumbore, single bounce)");
}
