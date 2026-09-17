"""
Isolates the intensity-scale discrepancy from all mesh/pose complexity.

Renders ONE flat plane, directly facing the spot light, at a known
distance, well within beam_width (so spot falloff = 1.0 exactly, no
angular cone math involved). At that point we can hand-calculate the
exact expected intensity from the formula used in raytracer_kernel.cu /
closesthit.rchit and compare it directly to what Mitsuba actually
renders for the identical setup -- no body mesh, no self-shadowing, no
silhouette edges, nothing else that could be contributing to the gap.

Run from genesis/raytracing/vulkan_raytracer/VulkanCudaInterop/:
    python diagnose_intensity_scale.py
"""
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import numpy as np

# Same light/material constants as pathtracer.py's get_deafult_scene()
LIGHT_POS = (0.0, 0.0, 3.0)
LIGHT_TARGET = (0.0, 0.0, 0.0)
LIGHT_INTENSITY = 1000.0
CUTOFF_ANGLE = 40.0
ALBEDO = 0.8

# Plane placed directly on the light's axis, facing straight back at it,
# so N.L = 1.0 and spot falloff = 1.0 (well within beam_width=30deg since
# the angle here is exactly 0deg). This removes every variable except
# the light_intensity / distance^2 falloff and the albedo/pi BSDF term.
PLANE_DISTANCE_FROM_LIGHT = 2.0  # plane sits at z = 3.0 - 2.0 = 1.0
plane_z = LIGHT_POS[2] - PLANE_DISTANCE_FROM_LIGHT

scene = mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'direct'},
    'sensor': {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f().look_at(
            origin=[0, 0, plane_z + 1.0], target=[0, 0, plane_z], up=[0, 1, 0]
        ),
        'fov': 10,  # narrow FOV so the plane fills the frame, minimizing AA/edge pixels
        'film': {
            'type': 'hdrfilm', 'width': 32, 'height': 32,
            'rfilter': {'type': 'gaussian'}, 'sample_border': True,
            'pixel_format': 'luminance', 'component_format': 'float32',
        },
        'sampler': {'type': 'independent', 'sample_count': 32, 'seed': 42},
    },
    'plane_bsdf': {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': (ALBEDO, ALBEDO, ALBEDO)},
    },
    'plane': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f().translate([0, 0, plane_z]).scale(5.0),
        'bsdf': {'type': 'ref', 'id': 'plane_bsdf'},
    },
    'light': {
        'type': 'spot',
        'cutoff_angle': CUTOFF_ANGLE,
        'to_world': mi.ScalarTransform4f().look_at(
            origin=list(LIGHT_POS), target=list(LIGHT_TARGET), up=[0, 1, 0]
        ),
        'intensity': LIGHT_INTENSITY,
    },
})

image = np.array(mi.render(scene, spp=32))[:, :, 0]
center = image[image.shape[0] // 2, image.shape[1] // 2]
# Average over the central patch too, to sanity check it's uniform
# (it should be, since the whole visible plane is equidistant-ish and
# flat within this narrow FOV).
patch = image[10:22, 10:22]

# Our formula (raytracer_kernel.cu / closesthit.rchit):
#   intensity = (albedo/pi) * N.L * (light_intensity / distance^2) * spot_falloff
# Here N.L = 1.0 (plane faces the light directly) and spot_falloff = 1.0
# (well within beam_width).
d = PLANE_DISTANCE_FROM_LIGHT
expected = (ALBEDO / np.pi) * 1.0 * (LIGHT_INTENSITY / (d * d)) * 1.0

print(f"Plane at distance {d} from light, facing it directly (N.L=1, spot_falloff=1)")
print(f"Mitsuba center pixel:      {center:.5f}")
print(f"Mitsuba patch mean:        {patch.mean():.5f}")
print(f"Our formula's prediction:  {expected:.5f}")
print(f"Ratio (Mitsuba / ours):    {center / expected:.5f}")
