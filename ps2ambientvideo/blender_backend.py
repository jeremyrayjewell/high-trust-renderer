from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .audio import analyze_audio
from .timeline import build_timeline, debug_targets_for_timeline


def _find_blender() -> str | None:
    env_blender = os.environ.get("CITYPROMISEVID_BLENDER") or os.environ.get("PS2AMBIENTVIDEO_BLENDER")
    if env_blender:
        candidate = Path(env_blender).expanduser()
        if candidate.exists():
            return str(candidate)

    blender = shutil.which("blender")
    if blender:
        return blender

    candidates: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if program_files:
        foundation = Path(program_files) / "Blender Foundation"
        if foundation.exists():
            candidates.extend(sorted(foundation.glob("Blender*\\blender.exe"), reverse=True))
    if local_app_data:
        local_foundation = Path(local_app_data) / "Programs" / "Blender Foundation"
        if local_foundation.exists():
            candidates.extend(sorted(local_foundation.glob("Blender*\\blender.exe"), reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _assert_blender() -> str:
    blender = _find_blender()
    if not blender:
        raise RuntimeError(
            "Blender backend requested, but no Blender executable was found on PATH, in CITYPROMISEVID_BLENDER, "
            "PS2AMBIENTVIDEO_BLENDER, or in standard install locations. Install Blender and add it to PATH, or set "
            "CITYPROMISEVID_BLENDER."
        )
    return blender


def _assert_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required on PATH to encode the Blender proof output.")
    return ffmpeg


def _build_blender_cube_smoke_script() -> str:
    return r'''
import json
import time
import sys
from pathlib import Path

argv = sys.argv
if "--" not in argv:
    raise RuntimeError("Expected config path after --")
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

import bpy
from mathutils import Vector
from mathutils import Vector
from mathutils import Vector


def available_engines(scene):
    return list(scene.render.bl_rna.properties["engine"].enum_items.keys())


def set_engine(scene, preferred):
    items = available_engines(scene)
    for name in preferred:
        if name in items:
            scene.render.engine = name
            return name
    raise RuntimeError(f"No supported engine from {preferred}; available={list(items)}")


started_at = time.perf_counter()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
available = available_engines(scene)
scene.render.resolution_x = int(config["width"])
scene.render.resolution_y = int(config["height"])
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = config["output_path"]
scene.render.film_transparent = False
requested = config.get("diagnostic_engine") or "workbench"
engine_preferences = {
    "workbench": ["BLENDER_WORKBENCH"],
    "eevee": ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
    "opengl": ["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
}
engine = set_engine(scene, engine_preferences.get(requested, ["BLENDER_WORKBENCH"]))
render_method = "opengl" if requested == "opengl" else "render"
world = bpy.data.worlds.new("DiagWorld")
scene.world = world
world.color = (0.74, 0.82, 0.92)
if engine == "BLENDER_WORKBENCH":
    wb = scene.display.shading
    wb.light = "STUDIO"
    wb.color_type = "MATERIAL"
    wb.show_specular_highlight = True
    wb.show_object_outline = False
    if hasattr(scene.display, "render_aa"):
        scene.display.render_aa = "FXAA"
else:
    eevee = scene.eevee
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 1
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom = False
    if hasattr(eevee, "use_gtao"):
        eevee.use_gtao = False
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = False
    if hasattr(eevee, "use_ssr_refraction"):
        eevee.use_ssr_refraction = False
    if hasattr(eevee, "use_motion_blur"):
        eevee.use_motion_blur = False

camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data)
camera.location = (5.4, -7.6, 3.8)
camera.rotation_euler = (1.10, 0.0, 0.62)
camera_data.lens = 32
scene.collection.objects.link(camera)
scene.camera = camera

light_data = bpy.data.lights.new(name="Key", type="SUN")
light = bpy.data.objects.new("Key", light_data)
light.rotation_euler = (0.95, 0.0, -0.75)
light.data.energy = 2.4
scene.collection.objects.link(light)

bpy.ops.mesh.primitive_plane_add(size=18.0, location=(0.0, 0.0, -1.15))
floor = bpy.context.active_object
floor_mat = bpy.data.materials.new("Floor")
floor_mat.diffuse_color = (0.72, 0.78, 0.84, 1.0)
floor.data.materials.append(floor_mat)

bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.15))
cube = bpy.context.active_object
cube.rotation_euler = (0.55, 0.28, 0.72)
cube.scale = (1.2, 1.05, 1.4)
mat = bpy.data.materials.new("Cube")
mat.diffuse_color = (0.20, 0.52, 0.86, 1.0)
cube.data.materials.append(mat)

scene.frame_set(1)
render_started = time.perf_counter()
if render_method == "opengl":
    bpy.ops.render.opengl(write_still=True, view_context=False)
else:
    bpy.ops.render.render(write_still=True)
output_png = Path(config["output_path"]).with_suffix(".png")
metadata_path = Path(config["metadata_output"])
metadata_path.write_text(json.dumps({
    "blender_version": bpy.app.version_string,
    "available_render_engines": available,
    "requested_engine": requested,
    "engine": engine,
    "actual_engine": scene.render.engine,
    "render_method": render_method,
    "samples": 1,
    "quality": config["quality"],
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "smoke_scene": config["smoke_scene"],
    "output_png_path": str(output_png),
    "output_exists": output_png.exists(),
    "output_file_size": output_png.stat().st_size if output_png.exists() else 0,
    "script_elapsed_seconds": round(time.perf_counter() - started_at, 3),
    "render_call_seconds": round(time.perf_counter() - render_started, 3),
    "settings": {
        "motion_blur": False,
        "ssr": False,
        "gtao": False,
        "bloom": False,
        "film_transparent": False,
        "world_color": [0.74, 0.82, 0.92],
    },
}, indent=2), encoding="utf-8")
'''


def _build_blender_material_smoke_script() -> str:
    return r'''
import json
import math
import time
import sys
from pathlib import Path

argv = sys.argv
if "--" not in argv:
    raise RuntimeError("Expected config path after --")
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

import bpy


def set_engine(scene, preferred):
    items = scene.render.bl_rna.properties["engine"].enum_items.keys()
    for name in preferred:
        if name in items:
            scene.render.engine = name
            return name
    raise RuntimeError(f"No supported engine from {preferred}; available={list(items)}")


def build_principled_material(name, base, *, transmission=0.0, alpha=1.0, roughness=0.3, metallic=0.0, stripe_scale=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND" if alpha < 0.999 or transmission > 0.01 else "OPAQUE"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE" if alpha < 0.999 or transmission > 0.01 else "OPAQUE"
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        if node.type not in {"OUTPUT_MATERIAL", "BSDF_PRINCIPLED"}:
            nodes.remove(node)
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = transmission
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.45 if transmission > 0.0 else 1.33
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.35 if transmission > 0.0 else 0.16
    if stripe_scale > 0.0:
        coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.inputs["Scale"].default_value = stripe_scale
        wave.inputs["Distortion"].default_value = 0.18
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.42
        ramp.color_ramp.elements[0].color = (base[0] * 0.75, base[1] * 0.75, base[2] * 0.75, 1.0)
        ramp.color_ramp.elements[1].position = 0.76
        ramp.color_ramp.elements[1].color = (min(1.0, base[0] + 0.12), min(1.0, base[1] + 0.12), min(1.0, base[2] + 0.12), 1.0)
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        links.new(wave.outputs["Color"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add_cube(location, scale, material, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.scale = scale
    obj.data.materials.append(material)
    return obj


def add_plane(location, scale, material, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.scale = scale
    obj.data.materials.append(material)
    return obj


def add_uv_sphere(location, scale, material):
    bpy.ops.mesh.primitive_uv_sphere_add(location=location, segments=10, ring_count=6)
    obj = bpy.context.active_object
    obj.scale = scale
    obj.data.materials.append(material)
    return obj


def add_cloud(location, scale, material):
    for ox, oy, oz in [(-0.7, 0.0, 0.0), (0.0, 0.14, 0.12), (0.7, -0.06, 0.0)]:
        add_uv_sphere((location[0] + ox, location[1] + oy, location[2] + oz), scale, material)


started_at = time.perf_counter()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = int(config["width"])
scene.render.resolution_y = int(config["height"])
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = config["output_path"]
scene.render.film_transparent = False
engine = set_engine(scene, ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"])
if engine != "BLENDER_WORKBENCH":
    eevee = scene.eevee
    samples = 1 if config["quality"] == "smoke" else 4
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = samples
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom = False
    if hasattr(eevee, "use_gtao"):
        eevee.use_gtao = False
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = False
    if hasattr(eevee, "use_ssr_refraction"):
        eevee.use_ssr_refraction = False
    if hasattr(eevee, "use_motion_blur"):
        eevee.use_motion_blur = False
else:
    samples = 1
    wb = scene.display.shading
    wb.light = "STUDIO"
    wb.color_type = "MATERIAL"
    wb.show_specular_highlight = True
    wb.show_object_outline = False
    if hasattr(scene.display, "render_aa"):
        scene.display.render_aa = "FXAA"

world = bpy.data.worlds.new("SmokeWorld")
world.use_nodes = True
scene.world = world
nt = world.node_tree
nodes = nt.nodes
links = nt.links
for node in list(nodes):
    nodes.remove(node)
output = nodes.new("ShaderNodeOutputWorld")
bg = nodes.new("ShaderNodeBackground")
grad = nodes.new("ShaderNodeTexGradient")
mapping = nodes.new("ShaderNodeMapping")
coord = nodes.new("ShaderNodeTexCoord")
ramp = nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.22
ramp.color_ramp.elements[0].color = (0.16, 0.28, 0.62, 1.0)
ramp.color_ramp.elements[1].position = 0.86
ramp.color_ramp.elements[1].color = (0.82, 0.93, 1.0, 1.0)
bg.inputs["Strength"].default_value = 0.72
links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
links.new(mapping.outputs["Vector"], grad.inputs["Vector"])
links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
links.new(ramp.outputs["Color"], bg.inputs["Color"])
links.new(bg.outputs["Background"], output.inputs["Surface"])

camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data)
camera.location = (4.9, -6.4, 3.6)
camera.rotation_euler = (1.02, 0.0, 0.62)
camera.data.lens = 28
scene.collection.objects.link(camera)
scene.camera = camera

sun_data = bpy.data.lights.new(name="Sun", type="SUN")
sun = bpy.data.objects.new("Sun", sun_data)
sun.rotation_euler = (0.92, 0.0, -0.76)
sun.data.energy = 2.2
scene.collection.objects.link(sun)

area_data = bpy.data.lights.new(name="Area", type="AREA")
area_data.energy = 900
area = bpy.data.objects.new("Area", area_data)
area.location = (0.0, -1.4, 4.0)
area.scale = (4.0, 4.0, 4.0)
scene.collection.objects.link(area)

floor_mat = build_principled_material("Floor", (0.44, 0.68, 0.94), roughness=0.12, metallic=0.06, stripe_scale=7.0)
water_mat = build_principled_material("Water", (0.14, 0.42, 0.92), transmission=0.0, alpha=0.72, roughness=0.10, stripe_scale=5.5)
glass_mat = build_principled_material("Glass", (0.74, 0.94, 1.00), transmission=0.0, alpha=0.30, roughness=0.08, stripe_scale=6.0)
plastic_mat = build_principled_material("Plastic", (0.42, 0.82, 0.95), transmission=0.0, alpha=0.76, roughness=0.14, stripe_scale=7.5)
inner_core_mat = build_principled_material("InnerCore", (0.12, 0.18, 0.30), alpha=0.60, roughness=0.24)
chrome_mat = build_principled_material("Chrome", (0.90, 0.94, 1.00), roughness=0.10, metallic=0.72, stripe_scale=12.0)
cloud_mat = build_principled_material("Cloud", (0.98, 0.99, 1.0), roughness=0.88, alpha=0.94)

add_plane((0.0, 0.0, 0.0), (4.2, 4.2, 1.0), floor_mat)
add_plane((0.0, -0.68, 0.01), (2.8, 0.36, 1.0), glass_mat)
add_plane((0.0, 1.15, 0.04), (2.4, 1.2, 1.0), water_mat)
add_plane((1.75, 0.28, 1.8), (0.82, 1.08, 1.0), glass_mat, rotation=(0.0, math.radians(16), 0.0))
add_cube((1.78, 0.28, 1.8), (0.03, 1.12, 1.02), chrome_mat)
add_cube((-0.15, -0.10, 1.15), (0.64, 0.38, 1.06), plastic_mat, rotation=(0.34, 0.14, 0.44))
add_cube((-0.15, -0.10, 1.15), (0.28, 0.14, 0.48), inner_core_mat, rotation=(0.34, 0.14, 0.44))
add_cube((0.12, 0.08, 1.60), (0.10, 0.02, 0.02), chrome_mat, rotation=(0.34, 0.14, 0.44))
for x, z in [(-0.8, 1.30), (0.0, 1.62), (0.8, 1.40)]:
    add_uv_sphere((x, 0.96, z), (0.04, 0.04, 0.04), chrome_mat)

add_cloud((-1.2, -2.5, 5.2), (0.82, 0.56, 0.34), cloud_mat)
add_cloud((1.6, -2.9, 5.5), (1.02, 0.66, 0.38), cloud_mat)

scene.frame_set(1)
render_started = time.perf_counter()
bpy.ops.render.render(write_still=True)
metadata_path = Path(config["metadata_output"])
metadata_path.write_text(json.dumps({
    "engine": engine,
    "samples": samples,
    "quality": config["quality"],
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "smoke_scene": config["smoke_scene"],
    "output_png_path": str(Path(config["output_path"]).with_suffix(".png")),
    "script_elapsed_seconds": round(time.perf_counter() - started_at, 3),
    "render_call_seconds": round(time.perf_counter() - render_started, 3),
    "settings": {
        "motion_blur": False,
        "ssr": False,
        "gtao": False,
        "bloom": False,
        "film_transparent": False,
        "shadow_method_alpha": "NONE",
        "transmission_mode": "fake_alpha_only",
    },
}, indent=2), encoding="utf-8")
'''


def _build_blender_material_batch_proof_script() -> str:
    return r'''
import json
import math
import time
import sys
from pathlib import Path

argv = sys.argv
if "--" not in argv:
    raise RuntimeError("Expected config path after --")
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

import bpy
from mathutils import Vector


def available_engines(scene):
    return list(scene.render.bl_rna.properties["engine"].enum_items.keys())


def choose_engine(scene):
    available = available_engines(scene)
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if candidate in available:
            scene.render.engine = candidate
            return candidate, available
    raise RuntimeError(f"No EEVEE engine available; available={available}")


def build_material(name, base, *, alpha=1.0, roughness=0.2, metallic=0.0, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND" if alpha < 0.999 else "OPAQUE"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE" if alpha < 0.999 else "OPAQUE"
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.28 if alpha < 0.999 else 0.12
    if "Emission Strength" in bsdf.inputs and emission > 0.0:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*base, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def build_floor_material(name):
    mat = build_material(name, (0.24, 0.40, 0.60), roughness=0.16, metallic=0.10)
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    bsdf = nodes["Principled BSDF"]
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    wave = nodes.new("ShaderNodeTexWave")
    ramp = nodes.new("ShaderNodeValToRGB")
    mix = nodes.new("ShaderNodeMixRGB")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 12.0
    wave.inputs["Distortion"].default_value = 1.2
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[0].color = (0.14, 0.24, 0.38, 1.0)
    ramp.color_ramp.elements[1].position = 0.58
    ramp.color_ramp.elements[1].color = (0.44, 0.64, 0.88, 1.0)
    mix.inputs["Fac"].default_value = 0.34
    mix.inputs["Color1"].default_value = (0.24, 0.40, 0.60, 1.0)
    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(wave.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.64
    return mat


def build_water_material(name):
    mat = build_material(name, (0.06, 0.24, 0.74), alpha=0.96, roughness=0.04)
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    bsdf = nodes["Principled BSDF"]
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise_a = nodes.new("ShaderNodeTexNoise")
    noise_b = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    mix = nodes.new("ShaderNodeMixRGB")
    noise_a.inputs["Scale"].default_value = 4.2
    noise_a.inputs["Detail"].default_value = 7.2
    noise_a.inputs["Roughness"].default_value = 0.48
    noise_b.inputs["Scale"].default_value = 18.0
    noise_b.inputs["Detail"].default_value = 6.0
    noise_b.inputs["Roughness"].default_value = 0.35
    ramp.color_ramp.elements[0].position = 0.26
    ramp.color_ramp.elements[0].color = (0.02, 0.10, 0.38, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (0.34, 0.78, 1.0, 1.0)
    mix.inputs["Fac"].default_value = 0.38
    mix.inputs["Color1"].default_value = (0.02, 0.16, 0.56, 1.0)
    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_a.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise_b.inputs["Vector"])
    links.new(noise_a.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["Color2"])
    links.new(noise_b.outputs["Fac"], mix.inputs["Fac"])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.94
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.50
    return mat


def build_glass_material(name):
    mat = build_material(name, (0.52, 0.82, 0.96), alpha=0.14, roughness=0.03)
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    bsdf = nodes["Principled BSDF"]
    fresnel = nodes.new("ShaderNodeLayerWeight")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.12
    ramp.color_ramp.elements[0].color = (0.30, 0.66, 0.88, 1.0)
    ramp.color_ramp.elements[1].position = 0.88
    ramp.color_ramp.elements[1].color = (0.94, 0.99, 1.0, 1.0)
    links.new(fresnel.outputs["Facing"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.92
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.46
    return mat


def build_shell_material(name, base, inner):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        if node.type not in {"OUTPUT_MATERIAL", "BSDF_PRINCIPLED"}:
            nodes.remove(node)
    output = nodes["Material Output"]
    bsdf = nodes["Principled BSDF"]
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    fresnel = nodes.new("ShaderNodeLayerWeight")
    mixrgb = nodes.new("ShaderNodeMixRGB")
    highlight_mix = nodes.new("ShaderNodeMixRGB")
    noise.inputs["Scale"].default_value = 6.5
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.35
    ramp.color_ramp.elements[0].position = 0.30
    ramp.color_ramp.elements[0].color = (*inner, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (*base, 1.0)
    mixrgb.inputs["Fac"].default_value = 0.42
    mixrgb.inputs["Color1"].default_value = (*base, 1.0)
    highlight_mix.inputs["Fac"].default_value = 0.30
    highlight_mix.inputs["Color2"].default_value = (0.96, 0.98, 1.0, 1.0)
    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mixrgb.inputs["Color2"])
    links.new(mixrgb.outputs["Color"], highlight_mix.inputs["Color1"])
    links.new(fresnel.outputs["Facing"], highlight_mix.inputs["Fac"])
    links.new(highlight_mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.14
    bsdf.inputs["Metallic"].default_value = 0.01
    bsdf.inputs["Specular IOR Level"].default_value = 1.0
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.62
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 0.72
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def ensure_collection(name):
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    return collection


def attach_to_collection(obj, collection):
    if obj.name in scene.collection.objects:
        scene.collection.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def add_plane(location, scale, material, rotation=(0.0, 0.0, 0.0), collection=None):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.scale = scale
    obj.data.materials.append(material)
    return attach_to_collection(obj, collection) if collection is not None else obj


def add_cube(location, scale, material, rotation=(0.0, 0.0, 0.0), collection=None):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.scale = scale
    obj.data.materials.append(material)
    return attach_to_collection(obj, collection) if collection is not None else obj


def add_sphere(location, scale, material, collection=None):
    bpy.ops.mesh.primitive_uv_sphere_add(location=location, segments=12, ring_count=8)
    obj = bpy.context.active_object
    obj.scale = scale
    obj.data.materials.append(material)
    return attach_to_collection(obj, collection) if collection is not None else obj


def add_curve(points, bevel_depth, material, collection=None):
    curve_data = bpy.data.curves.new("ProofCurve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = bevel_depth
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, point in zip(spline.bezier_points, points):
        bp.co = point
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new("CurveObj", curve_data)
    obj.data.materials.append(material)
    scene.collection.objects.link(obj)
    return attach_to_collection(obj, collection) if collection is not None else obj


def add_light(light_type, location, energy, collection=None, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(1.0, 1.0, 1.0)):
    light_data = bpy.data.lights.new(name=f"{light_type}Light", type=light_type)
    light_data.energy = energy
    if hasattr(light_data, "color"):
        light_data.color = color
    obj = bpy.data.objects.new(f"{light_type}LightObj", light_data)
    obj.location = location
    obj.rotation_euler = rotation
    obj.scale = scale
    scene.collection.objects.link(obj)
    return attach_to_collection(obj, collection) if collection is not None else obj


def set_smooth(obj):
    if hasattr(obj.data, "polygons"):
        for poly in obj.data.polygons:
            poly.use_smooth = True
    return obj


def add_wave_surface(center, width, depth, nx, ny, material, collection=None):
    mesh = bpy.data.meshes.new("WaveSurface")
    obj = bpy.data.objects.new("WaveSurfaceObj", mesh)
    verts = []
    faces = []
    for iy in range(ny):
        py = (iy / (ny - 1) - 0.5) * depth
        for ix in range(nx):
            px = (ix / (nx - 1) - 0.5) * width
            z = (
                math.sin(px * 2.4) * 0.05
                + math.cos(py * 3.6) * 0.035
                + math.sin((px + py) * 4.2) * 0.02
            )
            verts.append((center[0] + px, center[1] + py, center[2] + z))
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx + 1
            d = a + nx
            faces.append((a, b, c, d))
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj.data.materials.append(material)
    scene.collection.objects.link(obj)
    set_smooth(obj)
    return attach_to_collection(obj, collection) if collection is not None else obj


def add_cloud_cluster(center, top_mat, under_mat, scale, collection=None):
    cx, cy, cz = center
    puffs = [(-1.0, 0.0, 0.0), (-0.2, 0.12, 0.08), (0.6, -0.04, 0.0), (1.2, 0.06, 0.02)]
    for ox, oy, oz in puffs:
        add_sphere((cx + ox, cy + oy, cz + oz), scale, under_mat, collection=collection)
        add_sphere((cx + ox * 0.98, cy + oy, cz + oz + 0.14), (scale[0] * 0.92, scale[1] * 0.92, scale[2] * 0.82), top_mat, collection=collection)


def build_person(origin, shell_mat, core_mat, refl_mat, pose, collection=None):
    ox, oy, oz = origin
    arm_tilt, leg_tilt, lean = pose
    parts = []

    def cube(dx, dy, dz, sx, sy, sz, mat, rotation=(0.0, 0.0, 0.0), reflect=True):
        obj = add_cube((ox + dx, oy + dy, oz + dz), (sx, sy, sz), mat, rotation=rotation, collection=collection)
        parts.append(obj)
        if reflect:
            refl = add_cube((ox + dx, oy + dy, -0.96), (sx * 1.02, sy * 0.92, 0.012), refl_mat, rotation=(0.0, 0.0, rotation[2]), collection=collection)
            parts.append(refl)
        return obj

    def sphere(dx, dy, dz, sx, sy, sz, mat, reflect=True):
        obj = add_sphere((ox + dx, oy + dy, oz + dz), (sx, sy, sz), mat, collection=collection)
        parts.append(obj)
        if reflect:
            refl = add_cube((ox + dx, oy + dy, -0.955), (sx * 0.92, sy * 0.92, 0.012), refl_mat, collection=collection)
            parts.append(refl)
        return obj

    cube(0.0, 0.0, 0.78, 0.14, 0.12, 0.34, shell_mat, rotation=(lean, 0.0, 0.0))
    cube(0.0, 0.0, 0.79, 0.07, 0.05, 0.16, core_mat, rotation=(lean, 0.0, 0.0), reflect=False)
    cube(0.0, 0.0, 1.12, 0.17, 0.11, 0.18, shell_mat, rotation=(lean * 0.8, 0.0, 0.0))
    sphere(0.0, 0.0, 1.38, 0.10, 0.10, 0.10, shell_mat)
    cube(0.0, 0.07, 1.16, 0.11, 0.01, 0.04, refl_mat, rotation=(lean * 0.8, 0.0, 0.0), reflect=False)
    cube(-0.20, 0.0, 1.02, 0.05, 0.05, 0.24, shell_mat, rotation=(0.0, 0.0, arm_tilt))
    cube(0.20, 0.0, 1.02, 0.05, 0.05, 0.24, shell_mat, rotation=(0.0, 0.0, -arm_tilt))
    cube(-0.08, 0.0, 0.32, 0.05, 0.05, 0.28, shell_mat, rotation=(leg_tilt, 0.0, 0.0))
    cube(0.08, 0.0, 0.32, 0.05, 0.05, 0.28, shell_mat, rotation=(-leg_tilt, 0.0, 0.0))
    cube(-0.08, 0.04, 0.02, 0.08, 0.14, 0.02, shell_mat, reflect=False)
    cube(0.08, -0.04, 0.02, 0.08, 0.14, 0.02, shell_mat, reflect=False)
    parts.append(add_plane((ox, oy, -0.945), (0.26, 0.18, 1.0), refl_mat, collection=collection))
    return parts


started_at = time.perf_counter()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
engine, available = choose_engine(scene)
scene.render.resolution_x = int(config["width"])
scene.render.resolution_y = int(config["height"])
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.use_compositing = False
scene.render.use_sequencer = False
scene.display_settings.display_device = "sRGB"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

eevee = scene.eevee
if hasattr(eevee, "taa_render_samples"):
    eevee.taa_render_samples = 16
if hasattr(eevee, "taa_samples"):
    eevee.taa_samples = 16
if hasattr(eevee, "use_bloom"):
    eevee.use_bloom = False
if hasattr(eevee, "use_gtao"):
    eevee.use_gtao = True
if hasattr(eevee, "use_ssr"):
    eevee.use_ssr = False
if hasattr(eevee, "use_ssr_refraction"):
    eevee.use_ssr_refraction = False
if hasattr(eevee, "use_shadows"):
    eevee.use_shadows = True
if hasattr(eevee, "use_motion_blur"):
    eevee.use_motion_blur = False

world = bpy.data.worlds.new("ProofWorld")
world.use_nodes = True
scene.world = world
nt = world.node_tree
nodes = nt.nodes
links = nt.links
for node in list(nodes):
    nodes.remove(node)
output = nodes.new("ShaderNodeOutputWorld")
bg = nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0.03, 0.04, 0.05, 1.0)
bg.inputs["Strength"].default_value = 0.10
links.new(bg.outputs["Background"], output.inputs["Surface"])

camera_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", camera_data)
camera.data.lens = 34
scene.collection.objects.link(camera)
scene.camera = camera


def point_camera(cam, location, target):
    cam.location = location
    direction = Vector(target) - Vector(location)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

dancer_shells = {
    "cyan": build_shell_material("CyanShell", (0.40, 0.92, 1.0), (0.10, 0.28, 0.42)),
    "magenta": build_shell_material("MagentaShell", (0.92, 0.58, 0.92), (0.26, 0.10, 0.28)),
    "yellow": build_shell_material("YellowShell", (0.98, 0.88, 0.54), (0.28, 0.22, 0.08)),
    "green": build_shell_material("GreenShell", (0.64, 0.96, 0.60), (0.10, 0.22, 0.10)),
}
dancer_cores = {
    "cyan": build_material("CoreCyan", (0.04, 0.12, 0.20), alpha=0.92, roughness=0.24),
    "magenta": build_material("CoreMagenta", (0.14, 0.05, 0.18), alpha=0.92, roughness=0.24),
    "yellow": build_material("CoreYellow", (0.16, 0.14, 0.05), alpha=0.92, roughness=0.24),
    "green": build_material("CoreGreen", (0.06, 0.12, 0.05), alpha=0.92, roughness=0.24),
}
dancer_reflections = {
    "cyan": build_material("ReflectionCyan", (0.18, 0.60, 0.92), alpha=0.58, roughness=0.10),
    "magenta": build_material("ReflectionMagenta", (0.72, 0.34, 0.70), alpha=0.58, roughness=0.10),
    "yellow": build_material("ReflectionYellow", (0.76, 0.66, 0.24), alpha=0.54, roughness=0.10),
    "green": build_material("ReflectionGreen", (0.34, 0.68, 0.28), alpha=0.54, roughness=0.10),
}


def build_dancer_showcase(collection):
    center = -10.0
    floor_mat = build_material("DancerFloor", (0.05, 0.04, 0.16), roughness=0.04, metallic=0.18)
    stripe_mat = build_material("DancerStripe", (0.28, 0.20, 0.52), alpha=0.78, roughness=0.08)
    wall_mat = build_material("DancerWall", (0.08, 0.10, 0.22), roughness=0.88)
    glass_mat = build_glass_material("DancerGlass")
    chrome_mat = build_material("DancerChrome", (0.96, 0.97, 1.0), roughness=0.06, metallic=0.72)

    add_plane((center, 0.0, -1.0), (2.8, 2.2, 1.0), floor_mat, collection=collection)
    add_plane((center, -0.55, -0.995), (2.0, 0.10, 1.0), stripe_mat, collection=collection)
    add_plane((center, 2.0, 1.2), (4.0, 1.4, 1.0), wall_mat, rotation=(math.radians(90), 0.0, 0.0), collection=collection)
    add_plane((center - 1.95, 0.05, 0.72), (0.10, 0.92, 1.0), glass_mat, rotation=(0.0, math.radians(-10), 0.0), collection=collection)
    add_plane((center + 1.95, 0.05, 0.72), (0.10, 0.92, 1.0), glass_mat, rotation=(0.0, math.radians(10), 0.0), collection=collection)
    add_cube((center - 2.03, 0.05, 0.78), (0.02, 0.98, 0.82), chrome_mat, collection=collection)
    add_cube((center + 2.03, 0.05, 0.78), (0.02, 0.98, 0.82), chrome_mat, collection=collection)
    area_light = add_light("AREA", (center, -1.2, 3.2), 260, collection=collection, scale=(4.0, 4.0, 4.0), color=(0.95, 0.97, 1.0))
    sun_light = add_light("SUN", (center, -3.0, 5.0), 1.2, collection=collection, rotation=(1.02, 0.0, -0.40), color=(0.90, 0.94, 1.0))

    rigs = []
    glints = []
    people = [
        (center - 1.08, -0.14, "cyan", (0.42, 0.18, 0.08)),
        (center - 0.28, 0.12, "magenta", (0.76, -0.12, -0.05)),
        (center + 0.56, 0.04, "yellow", (0.34, 0.16, 0.03)),
        (center + 1.28, -0.10, "green", (0.58, -0.16, -0.06)),
    ]
    for x, y, key, pose in people:
        rigs.append(build_person((x, y, -0.98), dancer_shells[key], dancer_cores[key], dancer_reflections[key], pose, collection=collection))
        glints.append(add_plane((x, y + 0.10, 1.38), (0.05, 0.015, 1.0), chrome_mat, collection=collection))
        glints.append(add_plane((x, y + 0.06, 1.12), (0.08, 0.015, 1.0), chrome_mat, collection=collection))
    return {"center": center, "rigs": rigs, "glints": glints, "lights": [area_light, sun_light]}


def build_water_showcase(collection):
    center = 0.0
    deck_mat = build_material("WaterDeck", (0.05, 0.08, 0.14), roughness=0.18)
    basin_mat = build_material("WaterBasin", (0.14, 0.20, 0.34), roughness=0.24)
    water_mat = build_water_material("WaterSurface")
    crest_mat = build_material("WaterCrest", (0.74, 0.96, 1.0), alpha=0.88, roughness=0.04, emission=0.14)
    foam_mat = build_material("WaterFoam", (0.98, 0.99, 1.0), alpha=0.98, roughness=0.04, emission=0.08)
    reflect_cyan = build_material("WaterReflectCyan", (0.22, 0.78, 1.0), alpha=0.42, roughness=0.10)
    reflect_magenta = build_material("WaterReflectMagenta", (0.80, 0.38, 0.86), alpha=0.38, roughness=0.10)
    sky_mat = build_material("WaterSky", (0.36, 0.68, 0.98), roughness=1.0)

    add_plane((center, 0.0, -1.08), (3.8, 3.0, 1.0), deck_mat, collection=collection)
    add_cube((center, 0.12, -0.88), (2.90, 2.05, 0.22), basin_mat, collection=collection)
    wave_surface = add_wave_surface((center, 0.38, -0.66), 5.2, 3.4, 50, 34, water_mat, collection=collection)
    add_cube((center, 2.20, -0.58), (2.72, 0.10, 0.20), basin_mat, collection=collection)
    add_cube((center, -1.34, -0.58), (2.72, 0.10, 0.20), basin_mat, collection=collection)
    add_cube((center - 2.64, 0.42, -0.58), (0.10, 1.74, 0.20), basin_mat, collection=collection)
    add_cube((center + 2.64, 0.42, -0.58), (0.10, 1.74, 0.20), basin_mat, collection=collection)
    add_cube((center, 3.10, 1.50), (4.2, 0.12, 1.9), sky_mat, collection=collection)
    for x, y, z in [(-1.70, -0.20, -0.48), (1.55, 0.06, -0.50)]:
        add_cube((center + x, y, z), (0.16, 0.16, 0.64), reflect_cyan if x < 0 else reflect_magenta, collection=collection)
    foam_patches = []
    for x, scale_y in [(-1.0, 0.16), (-0.18, 0.14), (0.58, 0.18), (1.28, 0.14)]:
        foam_patches.append(add_plane((center + x, 0.36, -0.56), (0.44, scale_y, 1.0), crest_mat, collection=collection))
    splash_curves = []
    for points in [
        [(center - 0.65, 0.30, -0.58), (center - 0.38, 1.02, -0.10), (center - 0.10, 1.34, 0.12)],
        [(center + 0.18, 0.26, -0.58), (center + 0.02, 1.14, -0.08), (center - 0.12, 1.52, 0.14)],
        [(center + 0.62, 0.36, -0.56), (center + 0.86, 1.12, -0.08), (center + 1.10, 1.42, 0.10)],
    ]:
        splash_curves.append(add_curve(points, 0.030, crest_mat, collection=collection))
    droplets = []
    for x, y, z in [(-0.42, 1.18, 0.18), (-0.16, 1.34, 0.26), (0.10, 1.48, 0.24), (0.34, 1.30, 0.18), (0.58, 1.08, 0.10), (0.84, 0.92, 0.06), (-0.02, 1.12, 0.16), (0.22, 1.20, 0.14)]:
        droplets.append(add_sphere((center + x, y, z), (0.035, 0.035, 0.035), foam_mat, collection=collection))
    foam_lines = []
    for x, y in [(-0.26, 0.98), (0.04, 1.10), (0.32, 1.04), (0.60, 0.92)]:
        foam_lines.append(add_plane((center + x, y, -0.56), (0.20, 0.05, 1.0), foam_mat, collection=collection))
    area_light = add_light("AREA", (center, -1.7, 3.2), 240, collection=collection, scale=(4.8, 4.8, 4.8), color=(0.92, 0.97, 1.0))
    sun_light = add_light("SUN", (center, -2.6, 5.2), 1.12, collection=collection, rotation=(1.00, 0.0, -0.24), color=(0.90, 0.96, 1.0))
    return {
        "center": center,
        "wave": wave_surface,
        "foam_patches": foam_patches,
        "splash_curves": splash_curves,
        "droplets": droplets,
        "foam_lines": foam_lines,
        "lights": [area_light, sun_light],
    }


def build_cloud_showcase(collection):
    center = 10.0
    sky_mat = build_material("CloudSky", (0.30, 0.58, 0.90), roughness=1.0, emission=0.0)
    top_mat = build_material("CloudTop", (0.90, 0.94, 0.98), alpha=1.0, roughness=0.94)
    under_mat = build_material("CloudUnder", (0.68, 0.76, 0.88), alpha=1.0, roughness=0.96)
    sun_mat = build_material("CloudSun", (0.98, 0.96, 0.88), alpha=1.0, roughness=0.14, emission=0.05)
    add_cube((center, 2.20, 2.55), (8.0, 0.08, 3.6), sky_mat, collection=collection)
    add_plane((center + 2.90, 2.12, 3.78), (0.54, 0.54, 1.0), sun_mat, rotation=(math.radians(90), 0.0, 0.0), collection=collection)
    cloud_objects = []
    for pos, scale in [
        ((center - 2.5, 0.78, 2.58), (1.02, 0.70, 0.50)),
        ((center - 0.1, 0.92, 2.74), (1.36, 0.86, 0.62)),
        ((center + 2.1, 0.72, 2.56), (1.14, 0.74, 0.54)),
        ((center - 1.5, 0.18, 2.08), (0.76, 0.50, 0.36)),
        ((center + 1.2, 0.16, 2.10), (0.74, 0.48, 0.34)),
    ]:
        before = set(obj.name for obj in collection.objects)
        add_cloud_cluster(pos, top_mat, under_mat, scale, collection=collection)
        cloud_objects.extend([obj for obj in collection.objects if obj.name not in before])
    for pos, scale in [
        ((center - 3.2, 0.08, 1.92), (0.62, 0.42, 0.30)),
        ((center - 0.8, -0.06, 1.86), (0.78, 0.50, 0.36)),
        ((center + 2.8, 0.02, 1.88), (0.68, 0.44, 0.32)),
    ]:
        before = set(obj.name for obj in collection.objects)
        add_cloud_cluster(pos, top_mat, under_mat, scale, collection=collection)
        cloud_objects.extend([obj for obj in collection.objects if obj.name not in before])
    sun_light = add_light("SUN", (center, -2.2, 5.6), 0.42, collection=collection, rotation=(0.82, 0.0, -0.04), color=(1.0, 0.98, 0.95))
    area_light = add_light("AREA", (center, -0.8, 3.4), 24, collection=collection, scale=(6.2, 6.2, 6.2), color=(0.90, 0.95, 1.0))
    return {"center": center, "clouds": cloud_objects, "lights": [sun_light, area_light]}


def build_glass_showcase(collection):
    center = 20.0
    floor_mat = build_material("GlassFloor", (0.05, 0.10, 0.18), roughness=0.05, metallic=0.22)
    wall_mat = build_material("GlassBack", (0.04, 0.08, 0.16), roughness=0.88)
    void_mat = build_material("GlassVoid", (0.01, 0.02, 0.05), roughness=0.98)
    glass_mat = build_glass_material("PavilionGlass")
    chrome_mat = build_material("PavilionChrome", (0.96, 0.98, 1.0), roughness=0.05, metallic=0.78)
    glow_mat = build_material("PavilionGlow", (0.60, 0.88, 0.98), alpha=0.52, roughness=0.06, emission=0.02)
    reflect_mat = build_material("GlassReflect", (0.36, 0.74, 0.96), alpha=0.34, roughness=0.08)
    planter_mat = build_material("GlassPlanter", (0.10, 0.26, 0.18), roughness=0.40)
    leaf_mat = build_material("GlassLeaves", (0.34, 0.74, 0.44), alpha=0.92, roughness=0.32)
    skyglass_mat = build_material("GlassSky", (0.32, 0.64, 0.96), roughness=1.0)

    add_plane((center, 0.0, -1.0), (3.1, 2.6, 1.0), floor_mat, collection=collection)
    add_cube((center, 2.95, 1.6), (3.8, 0.10, 2.2), skyglass_mat, collection=collection)
    add_cube((center, 2.42, 1.24), (3.2, 0.10, 1.9), wall_mat, collection=collection)
    add_cube((center - 1.36, 0.28, 0.84), (0.10, 1.32, 1.04), glass_mat, rotation=(0.0, math.radians(-12), 0.0), collection=collection)
    add_cube((center + 1.36, 0.28, 0.84), (0.10, 1.32, 1.04), glass_mat, rotation=(0.0, math.radians(12), 0.0), collection=collection)
    add_cube((center - 0.86, 0.98, 1.68), (0.80, 0.04, 0.04), chrome_mat, collection=collection)
    add_cube((center + 0.86, 0.98, 1.68), (0.80, 0.04, 0.04), chrome_mat, collection=collection)
    add_cube((center, 0.98, 1.72), (1.86, 0.04, 0.04), chrome_mat, collection=collection)
    add_cube((center, 0.78, 0.62), (0.44, 0.08, 1.06), void_mat, collection=collection)
    add_cube((center, -0.10, 0.38), (0.36, 0.08, 0.64), glow_mat, collection=collection)
    add_cube((center, -0.10, 0.38), (0.04, 0.12, 0.78), chrome_mat, collection=collection)
    add_plane((center, -0.38, -0.992), (1.85, 0.10, 1.0), reflect_mat, collection=collection)
    for x, ang in [(-0.94, 26), (0.0, 0), (0.94, -26)]:
        add_cube((center + x, 0.10, -0.960), (0.26, 0.92, 0.04), reflect_mat, rotation=(0.0, math.radians(ang), 0.0), collection=collection)
        add_cube((center + x, 0.10, -0.988), (0.02, 1.04, 0.02), chrome_mat, collection=collection)
    add_plane((center - 1.04, 0.18, 0.82), (0.08, 0.96, 1.0), glass_mat, rotation=(0.0, math.radians(-18), 0.0), collection=collection)
    add_plane((center + 1.04, 0.18, 0.82), (0.08, 0.96, 1.0), glass_mat, rotation=(0.0, math.radians(18), 0.0), collection=collection)
    add_plane((center - 0.62, 0.18, 1.06), (0.03, 0.82, 1.0), chrome_mat, rotation=(0.0, math.radians(-10), 0.0), collection=collection)
    add_plane((center + 0.62, 0.18, 1.06), (0.03, 0.82, 1.0), chrome_mat, rotation=(0.0, math.radians(10), 0.0), collection=collection)
    add_cube((center - 0.82, -0.18, -0.84), (0.34, 0.18, 0.12), planter_mat, collection=collection)
    for off in [(-0.94, -0.10, -0.50), (-0.82, -0.12, -0.34), (-0.70, -0.08, -0.54)]:
        add_plane((center + off[0], off[1], off[2]), (0.14, 0.32, 1.0), leaf_mat, rotation=(math.radians(74), 0.0, math.radians(18)), collection=collection)
    add_cube((center - 2.10, -0.30, 0.92), (0.16, 1.30, 1.40), wall_mat, rotation=(0.0, math.radians(-16), 0.0), collection=collection)
    add_plane((center, -0.92, -0.995), (2.20, 0.18, 1.0), void_mat, collection=collection)
    area_light = add_light("AREA", (center, -1.3, 3.5), 300, collection=collection, scale=(4.4, 4.4, 4.4), color=(0.92, 0.98, 1.0))
    sun_light = add_light("SUN", (center, -2.6, 5.2), 1.18, collection=collection, rotation=(0.96, 0.0, -0.28), color=(0.95, 0.98, 1.0))
    return {"center": center, "lights": [area_light, sun_light]}


dancer_col = ensure_collection("DancerScene")
water_col = ensure_collection("WaterScene")
cloud_col = ensure_collection("CloudScene")
glass_col = ensure_collection("GlassScene")

dancer_scene = build_dancer_showcase(dancer_col)
water_scene = build_water_showcase(water_col)
cloud_scene = build_cloud_showcase(cloud_col)
glass_scene = build_glass_showcase(glass_col)

dancer_center = dancer_scene["center"]
water_center = water_scene["center"]
cloud_center = cloud_scene["center"]
glass_center = glass_scene["center"]

scene_collections = {
    "dancers_wide": dancer_col,
    "dancers_close": dancer_col,
    "reflection_close": dancer_col,
    "orbit_angle": dancer_col,
    "water_wide": water_col,
    "water_close": water_col,
    "clouds_wide": cloud_col,
    "clouds_close": cloud_col,
    "glass_wide": glass_col,
    "glass_close": glass_col,
}

views = [
    ("dancers_wide", (dancer_center, -3.0, 2.0), (dancer_center, 0.0, 0.8)),
    ("dancers_close", (dancer_center + 1.2, -1.75, 1.28), (dancer_center + 0.4, 0.0, 1.0)),
    ("reflection_close", (dancer_center - 0.7, -1.25, 0.66), (dancer_center, 0.08, 0.54)),
    ("water_wide", (water_center, -2.3, 1.3), (water_center, 0.5, -0.35)),
    ("water_close", (water_center + 0.02, -0.72, 0.48), (water_center, 0.95, -0.42)),
    ("clouds_wide", (cloud_center, -2.65, 2.18), (cloud_center, 0.74, 2.40)),
    ("clouds_close", (cloud_center + 0.08, -1.92, 2.06), (cloud_center + 0.04, 0.74, 2.34)),
    ("glass_wide", (glass_center, -2.20, 1.46), (glass_center, 0.45, 0.82)),
    ("glass_close", (glass_center - 0.66, -2.18, 0.62), (glass_center, 0.56, 0.52)),
    ("orbit_angle", (dancer_center - 2.3, -2.2, 1.85), (dancer_center + 0.1, 0.0, 0.9)),
]

family_collections = {
    "clouds": cloud_col,
    "glass": glass_col,
    "dancers": dancer_col,
    "water": water_col,
}


def set_active_collection(active_collection):
    for collection in (dancer_col, water_col, cloud_col, glass_col):
        visible = collection == active_collection
        collection.hide_viewport = not visible
        collection.hide_render = not visible


def animate_dancers(t, features):
    beat = float(features.get("beat", 0.0))
    bass = float(features.get("bass", 0.0))
    for idx, rig in enumerate(dancer_scene["rigs"]):
        phase = t * 1.2 + idx * 0.7
        swing = math.sin(phase * 2.0) * (0.18 + beat * 0.12)
        bounce = abs(math.sin(phase * 2.4)) * (0.08 + bass * 0.12)
        for obj in rig:
            base = obj.get("_base_loc")
            if base:
                obj.location = Vector(base)
            base_rot = obj.get("_base_rot")
            if base_rot:
                obj.rotation_euler = Vector(base_rot)
            if obj.name.endswith("_shadow") or "Plane" in obj.name:
                continue
            obj.location.z += bounce * 0.04
            obj.rotation_euler.z += swing * 0.06
    for idx, glint in enumerate(dancer_scene["glints"]):
        base = glint.get("_base_loc")
        if base:
            glint.location = Vector(base)
        glint.location.y += math.sin(t * 3.0 + idx) * 0.04


def animate_water(t, features):
    mids = float(features.get("mids", 0.0))
    highs = float(features.get("highs", 0.0))
    for idx, patch in enumerate(water_scene["foam_patches"]):
        base_scale = patch.get("_base_scale")
        if base_scale:
            patch.scale = Vector(base_scale)
        patch.scale.y = 0.12 + math.sin(t * 1.8 + idx * 0.7) * 0.03 + mids * 0.04
    for idx, droplet in enumerate(water_scene["droplets"]):
        base = droplet.get("_base_loc")
        if base:
            droplet.location = Vector(base)
        droplet.location.z += abs(math.sin(t * 2.2 + idx * 0.8)) * (0.03 + highs * 0.05)
    wave_base = water_scene["wave"].get("_base_rot")
    if wave_base:
        water_scene["wave"].rotation_euler = Vector(wave_base)
    water_scene["wave"].rotation_euler.z += math.sin(t * 0.45) * 0.02


def animate_clouds(t, features):
    energy = float(features.get("energy", 0.0))
    for idx, cloud in enumerate(cloud_scene["clouds"]):
        base = cloud.get("_base_loc")
        if base:
            cloud.location = Vector(base)
        drift = math.sin(t * 0.28 + idx * 0.31) * (0.05 + energy * 0.03)
        cloud.location.x += drift * 0.05
        cloud.location.y += math.cos(t * 0.22 + idx * 0.17) * 0.015


def animate_glass(t, features):
    highs = float(features.get("highs", 0.0))
    for idx, obj in enumerate(glass_col.objects):
        if "PavilionGlow" in obj.name or "GlassReflect" in obj.name:
            base = obj.get("_base_loc")
            if base:
                obj.location = Vector(base)
            obj.location.y += math.sin(t * 0.9 + idx * 0.5) * (0.02 + highs * 0.015)


def features_for_frame(index):
    feature_arrays = config.get("features", {})
    values = {}
    for key in ("beat", "bass", "mids", "highs", "energy", "onset"):
        arr = feature_arrays.get(key, [])
        values[key] = float(arr[min(index, len(arr) - 1)]) if arr else 0.0
    return values


def city_sequence():
    sequence = config.get("city_sequence", [])
    if sequence:
        return sequence
    return [
        {"scene": "clouds", "shot": "drift", "start": 0.0, "end": 6.0},
        {"scene": "glass", "shot": "dolly", "start": 6.0, "end": 12.0},
        {"scene": "dancers", "shot": "orbit", "start": 12.0, "end": 18.0},
        {"scene": "water", "shot": "low_pass", "start": 18.0, "end": 24.0},
    ]


def city_segment_for_time(t):
    sequence = city_sequence()
    for segment in sequence:
        if segment["start"] <= t < segment["end"]:
            return segment
    return sequence[-1]


def city_camera(scene_name, local_t, segment_duration, features, shot):
    bass = float(features.get("bass", 0.0))
    energy = float(features.get("energy", 0.0))
    if scene_name == "clouds":
        local = local_t
        if shot == "crane":
            return (
                (cloud_center - 0.9 + local * 0.08, -3.4 + local * 0.04, 1.75 + local * 0.06),
                (cloud_center + 0.20, 0.62, 2.32),
            )
        return (
            (cloud_center + math.sin(local * 0.24) * 0.55, -2.9 + local * 0.06, 2.18 + math.sin(local * 0.18) * 0.12),
            (cloud_center + 0.10, 0.70 + math.sin(local * 0.22) * 0.10, 2.38),
        )
    if scene_name == "glass":
        local = local_t
        if shot == "side_reveal":
            return (
                (glass_center - 2.2 + local * 0.12, -1.9 + math.sin(local * 0.4) * 0.10, 0.86 + energy * 0.05),
                (glass_center + 0.18, 0.58, 0.86),
            )
        return (
            (glass_center - 1.10 + local * 0.18, -2.55 + local * 0.08, 1.08 + math.sin(local * 0.35) * 0.08),
            (glass_center + 0.20, 0.48, 0.80),
        )
    if scene_name == "dancers":
        local = local_t
        if shot == "close_orbit":
            return (
                (dancer_center + math.sin(local * 0.42) * 0.9, -1.65 + math.cos(local * 0.36) * 0.18, 1.22 + bass * 0.10),
                (dancer_center + 0.02, 0.00, 1.04),
            )
        radius = 1.55 + energy * 0.08
        angle = -0.95 + local * 0.24
        return (
            (dancer_center + math.sin(angle) * radius, -2.25 + math.cos(angle) * 0.32, 1.34 + bass * 0.12),
            (dancer_center + 0.05, 0.02, 0.98),
        )
    local = local_t
    if shot == "basin_arc":
        return (
            (water_center - 1.22 + math.sin(local * 0.28) * 0.42, -1.18 + local * 0.06, 0.96 + bass * 0.06),
            (water_center + 0.12, 0.72, -0.28),
        )
    return (
        (water_center - 0.62 + local * 0.22, -1.48 + math.sin(local * 0.25) * 0.08, 0.82 + bass * 0.08),
        (water_center + 0.10, 0.82, -0.34),
    )


def render_city_preview():
    frame_count = int(config["frame_count"])
    fps = float(config["fps"])
    frame_output_pattern = Path(config["frame_output_pattern"])
    frame_output_pattern.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(config["debug_frames_dir"]) if config.get("debug_frames_dir") else None
    debug_map = {int(entry["frame"]): entry["filename"] for entry in config.get("debug_frames", [])}
    saved_rows = []

    scene.frame_set(1)
    set_active_collection(cloud_col)
    point_camera(camera, (cloud_center, -3.0, 2.1), (cloud_center, 0.7, 2.35))
    warmup_path = Path(config["warmup_output"]).with_suffix("")
    warmup_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(warmup_path)
    warmup_started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    warmup_seconds = round(time.perf_counter() - warmup_started, 3)

    for collection in (dancer_col, water_col, cloud_col, glass_col):
        for obj in collection.objects:
            obj["_base_loc"] = tuple(obj.location)
            obj["_base_rot"] = tuple(obj.rotation_euler)
            obj["_base_scale"] = tuple(obj.scale)

    for frame_number in range(1, frame_count + 1):
        t = (frame_number - 1) / fps
        features = features_for_frame(frame_number - 1)
        segment = city_segment_for_time(t)
        scene_name = segment["scene"]
        local_t = t - float(segment["start"])
        segment_duration = max(0.001, float(segment["end"]) - float(segment["start"]))
        shot = segment.get("shot", "default")
        if scene_name == "clouds":
            animate_clouds(t, features)
        elif scene_name == "glass":
            animate_glass(t, features)
        elif scene_name == "dancers":
            animate_dancers(t, features)
        else:
            animate_water(t, features)
        set_active_collection(family_collections[scene_name])
        location, target = city_camera(scene_name, local_t, segment_duration, features, shot)
        point_camera(camera, location, target)
        frame_path = frame_output_pattern.parent / f"frame_{frame_number:04d}.png"
        scene.render.filepath = str(frame_path.with_suffix(""))
        render_started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        elapsed = round(time.perf_counter() - render_started, 3)
        saved_rows.append({
            "frame": frame_number,
            "time": round(t, 3),
            "scene": scene_name,
            "shot": shot,
            "render_call_seconds": elapsed,
            "png_path": str(frame_path),
        })
        if debug_dir is not None and frame_number in debug_map:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / debug_map[frame_number]
            scene.render.filepath = str(debug_path.with_suffix(""))
            bpy.ops.render.render(write_still=True)

    metadata_path = Path(config["metadata_output"])
    metadata_path.write_text(json.dumps({
        "blender_version": bpy.app.version_string,
        "available_render_engines": available,
        "requested_engine": "eevee",
        "actual_engine": engine,
        "width": scene.render.resolution_x,
        "height": scene.render.resolution_y,
        "warmup_render_seconds": warmup_seconds,
        "total_script_seconds": round(time.perf_counter() - started_at, 3),
        "saved_frames": saved_rows,
        "warmup_output": str(Path(config["warmup_output"]).with_suffix(".png")),
        "output_png_list": [row["png_path"] for row in saved_rows],
        "city_preview": True,
    }, indent=2), encoding="utf-8")
    return


if config.get("city_preview"):
    render_city_preview()
    raise SystemExit(0)

scene.frame_set(1)
set_active_collection(dancer_col)
point_camera(camera, (dancer_center, -3.6, 2.1), (dancer_center, 0.0, 0.8))
Path(config["warmup_output"]).parent.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(Path(config["warmup_output"]).with_suffix(""))
warmup_started = time.perf_counter()
bpy.ops.render.render(write_still=True)
warmup_seconds = round(time.perf_counter() - warmup_started, 3)

saved_rows = []
debug_dir = Path(config["debug_frames_dir"])
debug_dir.mkdir(parents=True, exist_ok=True)
for label, location, target in views:
    active_collection = scene_collections[label]
    set_active_collection(active_collection)
    point_camera(camera, location, target)
    target = debug_dir / f"{label}.png"
    scene.render.filepath = str(target.with_suffix(""))
    render_started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = round(time.perf_counter() - render_started, 3)
    saved_rows.append({
        "label": label,
        "render_call_seconds": elapsed,
        "png_path": str(target),
        "png_exists": target.exists(),
        "png_size": target.stat().st_size if target.exists() else 0,
    })

metadata_path = Path(config["metadata_output"])
metadata_path.write_text(json.dumps({
    "blender_version": bpy.app.version_string,
    "available_render_engines": available,
    "requested_engine": "eevee",
    "actual_engine": engine,
    "width": scene.render.resolution_x,
    "height": scene.render.resolution_y,
    "warmup_render_seconds": warmup_seconds,
    "total_script_seconds": round(time.perf_counter() - started_at, 3),
    "saved_frames": saved_rows,
    "warmup_output": str(Path(config["warmup_output"]).with_suffix(".png")),
    "output_png_list": [row["png_path"] for row in saved_rows],
    "settings": {
        "samples": 16,
        "motion_blur": False,
        "ssr": False,
        "gtao": True,
        "bloom": False,
        "shadows": True,
        "alpha_mode": "blend",
        "refraction": False,
    },
}, indent=2), encoding="utf-8")
'''


def _run_blender_process(
    *,
    blender: str,
    script_text: str,
    payload: dict[str, object],
    artifact_dir: Path,
    timeout_seconds: int,
) -> tuple[int, Path, Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    script_path = artifact_dir / "blender_driver.py"
    config_path = artifact_dir / "blender_config.json"
    stdout_path = artifact_dir / "blender_stdout.log"
    stderr_path = artifact_dir / "blender_stderr.log"
    run_info_path = artifact_dir / "blender_run_info.json"
    metadata_path = artifact_dir / "blender_metadata.json"

    script_path.write_text(script_text, encoding="utf-8")
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        str(config_path),
    ]
    started_at = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        run_info_path.write_text(
            json.dumps(
                {
                    "command": command,
                    "timeout_seconds": timeout_seconds,
                    "timed_out": True,
                    "payload_render_settings": payload.get("render_settings", {}),
                    "smoke_scene": payload.get("smoke_scene"),
                    "quality": payload.get("quality"),
                    "output_dir_listing": sorted(path.name for path in artifact_dir.iterdir()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Blender timed out after {timeout_seconds}s. Logs: {stdout_path} and {stderr_path}. Script: {script_path}"
        ) from exc

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    subprocess_elapsed = round(time.time() - started_at, 3)
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"metadata_error": "invalid_json"}
    run_info_path.write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "timed_out": False,
                "subprocess_elapsed_seconds": subprocess_elapsed,
                "blender_script_elapsed_seconds": metadata.get("script_elapsed_seconds"),
                "render_call_seconds": metadata.get("render_call_seconds"),
                "estimated_startup_overhead_seconds": round(
                    max(0.0, subprocess_elapsed - float(metadata.get("script_elapsed_seconds", 0.0))),
                    3,
                ),
                "engine_used": metadata.get("engine", payload.get("render_settings", {}).get("engine")),
                "samples": metadata.get("samples", payload.get("render_settings", {}).get("samples")),
                "resolution": metadata.get("resolution", [payload.get("width"), payload.get("height")]),
                "smoke_scene": payload.get("smoke_scene"),
                "quality": payload.get("quality"),
                "output_png_path": metadata.get("output_png_path"),
                "payload_render_settings": payload.get("render_settings", {}),
                "metadata": metadata,
                "output_dir_listing": sorted(path.name for path in artifact_dir.iterdir()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return completed.returncode, script_path, stdout_path, stderr_path


def _build_blender_city_sequence(duration: float) -> list[dict[str, object]]:
    if duration <= 30.0:
        return [
            {"scene": "clouds", "shot": "drift", "start": 0.0, "end": min(duration, 6.0)},
            {"scene": "glass", "shot": "dolly", "start": min(duration, 6.0), "end": min(duration, 12.0)},
            {"scene": "dancers", "shot": "orbit", "start": min(duration, 12.0), "end": min(duration, 18.0)},
            {"scene": "water", "shot": "low_pass", "start": min(duration, 18.0), "end": duration},
        ]

    fractions = [0.14, 0.28, 0.45, 0.60, 0.75, 0.86, 0.94, 1.0]
    ends = [round(duration * value, 3) for value in fractions]
    starts = [0.0] + ends[:-1]
    scenes = [
        ("clouds", "drift"),
        ("glass", "dolly"),
        ("dancers", "orbit"),
        ("water", "low_pass"),
        ("glass", "side_reveal"),
        ("clouds", "crane"),
        ("dancers", "close_orbit"),
        ("water", "basin_arc"),
    ]
    sequence: list[dict[str, object]] = []
    for (scene, shot), start, end in zip(scenes, starts, ends):
        if end - start <= 0.01:
            continue
        sequence.append({"scene": scene, "shot": shot, "start": start, "end": end})
    return sequence


def _city_debug_entries(sequence: list[dict[str, object]], duration: float, fps: int) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for segment in sequence:
        target = min(duration, (float(segment["start"]) + float(segment["end"])) * 0.5)
        frame_index = max(0, round(target * fps))
        scene = str(segment["scene"])
        shot = str(segment.get("shot", "shot"))
        entries.append(
            {
                "frame": frame_index + 1,
                "filename": f"t_{target:05.2f}s__{scene}_{shot}.png",
            }
        )
    return entries


def _build_blender_script() -> str:
    return r'''
import json
import math
import sys
from pathlib import Path

argv = sys.argv
if "--" not in argv:
    raise RuntimeError("Expected config path after --")
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

import bpy
from mathutils import Vector


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    return scene


def make_world(scene):
    world = bpy.data.worlds.new("ProofWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    grad = nodes.new("ShaderNodeTexGradient")
    mapping = nodes.new("ShaderNodeMapping")
    texcoord = nodes.new("ShaderNodeTexCoord")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (0.04, 0.14, 0.34, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (0.72, 0.90, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 0.9
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], grad.inputs["Vector"])
    links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])


def ensure_collection(name):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_obj(collection, obj):
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    if obj.name in bpy.context.scene.collection.objects:
        if collection != bpy.context.scene.collection:
            bpy.context.scene.collection.objects.unlink(obj)


def build_material(name, base, *, transmission=0.0, roughness=0.3, metallic=0.0, alpha=1.0, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND" if alpha < 0.999 or transmission > 0.01 else "OPAQUE"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "HASHED"
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        if node.type not in {"OUTPUT_MATERIAL", "BSDF_PRINCIPLED"}:
            nodes.remove(node)
    principled = nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (*base, 1.0)
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
    if "Transmission Weight" in principled.inputs:
        principled.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in principled.inputs:
        principled.inputs["Transmission"].default_value = transmission
    if "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = alpha
    if "IOR" in principled.inputs:
        principled.inputs["IOR"].default_value = 1.45 if transmission > 0.0 else 1.33
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 1.0
    if "Coat Weight" in principled.inputs:
        principled.inputs["Coat Weight"].default_value = 0.35 if transmission > 0.0 else 0.12
    if emission > 0.0:
        if "Emission Color" in principled.inputs:
            principled.inputs["Emission Color"].default_value = (*base, 1.0)
        if "Emission Strength" in principled.inputs:
            principled.inputs["Emission Strength"].default_value = emission
    if config.get("proof_stills"):
        return mat
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 10.0
    noise.inputs["Detail"].default_value = 4.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (base[0] * 0.7, base[1] * 0.7, base[2] * 0.7, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (min(1.0, base[0] * 1.1 + 0.08), min(1.0, base[1] * 1.1 + 0.08), min(1.0, base[2] * 1.1 + 0.08), 1.0)
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.04
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return mat


def add_plane(collection, name, location, scale, material):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    link_obj(collection, obj)
    return obj


def add_cube(collection, name, location, scale, material, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    link_obj(collection, obj)
    return obj


def add_uv_sphere(collection, name, location, scale, material):
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=location,
        segments=8 if config.get("proof_stills") else 20,
        ring_count=6 if config.get("proof_stills") else 12,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    link_obj(collection, obj)
    return obj


def add_cylinder(collection, name, location, scale, material, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cylinder_add(
        location=location,
        rotation=rotation,
        vertices=8 if config.get("proof_stills") else 18,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    link_obj(collection, obj)
    return obj


def build_person(collection, name, x, y, color, plastic_mat):
    root = bpy.data.objects.new(name, None)
    link_obj(collection, root)
    parts = {}
    torso = add_cube(collection, f"{name}_torso", (x, y, 1.2), (0.22, 0.16, 0.34), plastic_mat)
    chest = add_cube(collection, f"{name}_chest", (x, y, 1.55), (0.24, 0.14, 0.2), plastic_mat)
    head = add_uv_sphere(collection, f"{name}_head", (x, y, 1.92), (0.12, 0.12, 0.12), plastic_mat)
    hip = add_cube(collection, f"{name}_hip", (x, y, 0.9), (0.2, 0.14, 0.12), plastic_mat)
    l_arm = add_cube(collection, f"{name}_l_arm", (x - 0.3, y, 1.42), (0.08, 0.08, 0.28), plastic_mat, rotation=(0.0, 0.0, 0.4))
    r_arm = add_cube(collection, f"{name}_r_arm", (x + 0.3, y, 1.42), (0.08, 0.08, 0.28), plastic_mat, rotation=(0.0, 0.0, -0.4))
    l_leg = add_cube(collection, f"{name}_l_leg", (x - 0.1, y, 0.42), (0.08, 0.08, 0.34), plastic_mat, rotation=(0.0, 0.0, 0.1))
    r_leg = add_cube(collection, f"{name}_r_leg", (x + 0.1, y, 0.42), (0.08, 0.08, 0.34), plastic_mat, rotation=(0.0, 0.0, -0.1))
    l_foot = add_cube(collection, f"{name}_l_foot", (x - 0.1, y + 0.05, 0.05), (0.1, 0.16, 0.04), plastic_mat)
    r_foot = add_cube(collection, f"{name}_r_foot", (x + 0.1, y - 0.05, 0.05), (0.1, 0.16, 0.04), plastic_mat)
    shadow = add_plane(collection, f"{name}_shadow", (x, y, 0.01), (0.42, 0.24, 1.0), SHADOW_MAT)
    for obj in (torso, chest, head, hip, l_arm, r_arm, l_leg, r_leg, l_foot, r_foot, shadow):
        obj.parent = root
    parts.update({
        "root": root,
        "torso": torso,
        "chest": chest,
        "head": head,
        "hip": hip,
        "l_arm": l_arm,
        "r_arm": r_arm,
        "l_leg": l_leg,
        "r_leg": r_leg,
        "l_foot": l_foot,
        "r_foot": r_foot,
        "shadow": shadow,
    })
    return parts


def create_scene():
    scene = clear_scene()
    make_world(scene)
    scene.render.engine = "BLENDER_EEVEE"
    eevee = scene.eevee
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 1 if config.get("proof_stills") else 24
    if hasattr(eevee, "use_gtao"):
        eevee.use_gtao = False if config.get("proof_stills") else True
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom = False
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = False if config.get("proof_stills") else True
    if hasattr(eevee, "use_ssr_refraction"):
        eevee.use_ssr_refraction = False if config.get("proof_stills") else True
    if hasattr(eevee, "use_motion_blur"):
        eevee.use_motion_blur = False
    scene.render.resolution_x = config["width"]
    scene.render.resolution_y = config["height"]
    scene.render.resolution_percentage = 100
    scene.render.fps = config["fps"]
    scene.frame_start = 1
    scene.frame_end = config["frame_count"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = config["frame_output_pattern"]
    scene.render.film_transparent = False
    scene.camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(scene.camera)
    scene.camera.data.lens = 28
    scene.camera.location = (0.0, -10.0, 3.6)
    target = bpy.data.objects.new("CameraTarget", None)
    scene.collection.objects.link(target)
    constraint = scene.camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    sun_data = bpy.data.lights.new(name="Sun", type="SUN")
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.9, 0.0, -0.8)
    sun.data.energy = 2.1
    area_data = bpy.data.lights.new(name="Area", type="AREA")
    area_data.energy = 2500
    area = bpy.data.objects.new("Area", area_data)
    area.location = (0.0, -2.0, 5.6)
    area.scale = (6.0, 6.0, 6.0)
    scene.collection.objects.link(area)
    return scene, target


def set_obj_visible(obj, visible):
    obj.hide_viewport = not visible
    obj.hide_render = not visible


scene, camera_target = create_scene()
root = scene.collection

PLASTIC_CYAN = build_material("PlasticCyan", (0.28, 0.78, 0.92), transmission=0.65, roughness=0.12, alpha=0.78)
PLASTIC_MAGENTA = build_material("PlasticMagenta", (0.88, 0.46, 0.78), transmission=0.62, roughness=0.14, alpha=0.78)
PLASTIC_YELLOW = build_material("PlasticYellow", (0.92, 0.84, 0.44), transmission=0.48, roughness=0.16, alpha=0.8)
PLASTIC_GREEN = build_material("PlasticGreen", (0.42, 0.82, 0.54), transmission=0.54, roughness=0.14, alpha=0.8)
GLASS_MAT = build_material("GlassMat", (0.62, 0.9, 1.0), transmission=0.9, roughness=0.03, alpha=0.22)
FLOOR_MAT = build_material("FloorMat", (0.38, 0.6, 0.9), roughness=0.06, metallic=0.08)
WATER_MAT = build_material("WaterMat", (0.08, 0.32, 0.86), transmission=0.25, roughness=0.04, alpha=0.94)
CHROME_MAT = build_material("ChromeMat", (0.86, 0.92, 1.0), roughness=0.08, metallic=0.65)
GRASS_MAT = build_material("GrassMat", (0.24, 0.72, 0.34), roughness=0.7, metallic=0.0)
CLOUD_MAT = build_material("CloudMat", (0.98, 0.99, 1.0), transmission=0.0, roughness=0.9, alpha=0.94)
SPACE_MAT = build_material("SpaceMetal", (0.12, 0.16, 0.32), roughness=0.22, metallic=0.28)
SHADOW_MAT = build_material("ShadowMat", (0.06, 0.12, 0.2), roughness=1.0, metallic=0.0, alpha=0.34)

sky_col = ensure_collection("SkyZone")
people_col = ensure_collection("PeopleZone")
water_col = ensure_collection("WaterZone")
glass_col = ensure_collection("GlassZone")
space_col = ensure_collection("SpaceZone")

zones = {
    "blue_sky_aero_terminal": Vector((-12.0, 0.0, 0.0)),
    "dancing_polygon_crowd": Vector((-4.0, 0.0, 0.0)),
    "splash_fountain_court": Vector((4.0, 0.0, 0.0)),
    "glass_greenhouse": Vector((12.0, 0.0, 0.0)),
    "glass_elevator_shaft": Vector((20.0, 0.0, 0.0)),
    "orbital_garden_station": Vector((28.0, 0.0, 0.0)),
    "ocean_glass_terrace": Vector((36.0, 0.0, 0.0)),
}

# Sky / terminal zone
sky_floor = add_plane(sky_col, "SkyFloor", (-12.0, 0.0, 0.0), (6.0, 6.0, 1.0), FLOOR_MAT)
for dx in (-3.8, -1.2, 1.4, 4.2):
    add_cube(sky_col, f"SkyColumn_{dx}", (-12.0 + dx, 0.6, 1.4), (0.14, 0.14, 1.4), CHROME_MAT)
for dx in (-4.2, 0.0, 4.2):
    add_plane(sky_col, f"SkyRoof_{dx}", (-12.0 + dx, -0.4, 3.7), (1.9, 1.2, 1.0), GLASS_MAT)
add_cube(sky_col, "SkyShuttle", (-15.0, 1.8, 0.34), (0.8, 0.4, 0.28), PLASTIC_CYAN)

# People zone
people_floor = add_plane(people_col, "PeopleFloor", (-4.0, 0.0, 0.0), (5.0, 5.0, 1.0), FLOOR_MAT)
people_bar = add_cube(people_col, "PeopleBackdrop", (-4.0, 2.6, 1.4), (4.2, 0.18, 1.4), GLASS_MAT)
people_rigs = []
person_specs = [
    (-5.6, -0.5, PLASTIC_MAGENTA),
    (-4.7, 0.7, PLASTIC_YELLOW),
    (-3.9, -0.2, PLASTIC_CYAN),
    (-3.0, 0.9, PLASTIC_GREEN),
    (-2.2, -0.7, PLASTIC_CYAN),
]
for idx, (px, py, mat) in enumerate(person_specs):
    people_rigs.append(build_person(people_col, f"Dancer{idx}", px, py, (1, 1, 1), mat))

# Water zone
water_floor = add_plane(water_col, "WaterDeck", (4.0, 0.0, 0.0), (5.8, 5.8, 1.0), FLOOR_MAT)
water_plane = add_plane(water_col, "WaterPlane", (4.0, 0.0, 0.05), (4.6, 3.4, 1.0), WATER_MAT)
subdiv = water_plane.modifiers.new("Subsurf", type="SUBSURF")
subdiv.levels = 1 if config.get("proof_stills") else 3
wave = water_plane.modifiers.new("Wave", type="WAVE")
wave.height = 0.09
wave.width = 1.4
wave.speed = 0.3
wave.narrowness = 1.1
jets = []
for idx, offset in enumerate((-1.8, -0.9, 0.0, 0.9, 1.8)):
    jets.append(add_cylinder(water_col, f"Jet{idx}", (4.0 + offset, 0.0, 1.0), (0.06, 0.06, 1.0), WATER_MAT))
for idx in range(4 if config.get("proof_stills") else 12):
    add_uv_sphere(water_col, f"Droplet{idx}", (4.0 + ((idx % 4) - 1.5) * 0.6, (idx // 4 - 1) * 0.5, 1.2 + (idx % 3) * 0.2), (0.06, 0.06, 0.06), CHROME_MAT)

# Glass zone
glass_floor = add_plane(glass_col, "GlassFloor", (12.0, 0.0, 0.0), (5.4, 5.4, 1.0), FLOOR_MAT)
for dx in (-2.2, 0.0, 2.2):
    add_cube(glass_col, f"Planter_{dx}", (12.0 + dx, 1.2, 0.44), (0.36, 0.36, 0.44), GRASS_MAT)
    add_uv_sphere(glass_col, f"Plant_{dx}", (12.0 + dx, 1.2, 1.36), (0.44, 0.44, 0.6), GRASS_MAT)
for side in (-1, 1):
    add_plane(glass_col, f"GlassWall_{side}", (12.0 + side * 2.8, 0.2, 2.0), (0.2, 2.4, 1.0), GLASS_MAT)
    add_cube(glass_col, f"GlassFrame_{side}", (12.0 + side * 2.8, 0.2, 2.0), (0.06, 2.46, 2.2), CHROME_MAT)
add_plane(glass_col, "GlassRoof", (12.0, -0.2, 3.6), (3.6, 1.8, 1.0), GLASS_MAT)

# Elevator zone
elevator_floor = add_plane(glass_col, "ElevatorFloor", (20.0, 0.0, 0.0), (4.4, 4.8, 1.0), FLOOR_MAT)
for side in (-1, 1):
    add_plane(glass_col, f"ElevatorWall_{side}", (20.0 + side * 1.9, 0.0, 2.0), (0.16, 1.6, 1.0), GLASS_MAT)
    add_cube(glass_col, f"ElevatorRail_{side}", (20.0 + side * 1.94, 0.0, 2.0), (0.04, 0.04, 2.2), CHROME_MAT)
elevator_cab = add_cube(glass_col, "ElevatorCab", (20.0, -0.6, 1.1), (0.9, 0.9, 1.0), GLASS_MAT)

# Space zone
space_floor = add_plane(space_col, "SpaceDeck", (28.0, 0.0, 0.0), (5.4, 5.4, 1.0), SPACE_MAT)
planet = add_uv_sphere(space_col, "Planet", (30.6, 2.2, 3.0), (1.2, 1.2, 1.2), GLASS_MAT)
ring = add_torus = bpy.ops.mesh.primitive_torus_add
bpy.ops.mesh.primitive_torus_add(location=(30.6, 2.2, 3.0), major_radius=1.8, minor_radius=0.06)
planet_ring = bpy.context.active_object
planet_ring.data.materials.append(CHROME_MAT)
link_obj(space_col, planet_ring)
for dx in (-1.8, 0.0, 1.8):
    add_cube(space_col, f"SpaceTower_{dx}", (28.0 + dx, 0.8, 1.2), (0.24, 0.24, 1.2), CHROME_MAT)

# Ocean terrace zone
ocean_deck = add_plane(water_col, "OceanDeck", (36.0, 0.0, 0.0), (5.8, 5.8, 1.0), FLOOR_MAT)
ocean = add_plane(water_col, "OceanPlane", (36.0, 3.0, -0.05), (7.2, 6.0, 1.0), WATER_MAT)
ocean_wave = ocean.modifiers.new("OceanWave", type="WAVE")
ocean_wave.height = 0.06
ocean_wave.width = 2.4
ocean_wave.speed = 0.22
for side in (-1, 1):
    add_plane(water_col, f"OceanRail_{side}", (36.0 + side * 2.8, 0.8, 1.0), (0.12, 2.0, 1.0), GLASS_MAT)

# Clouds
cloud_objects = []
cloud_specs = [
    (-12.0, -2.8, 5.4, 1.2, 0.8, 0.5),
    (-9.8, -3.3, 5.1, 0.9, 0.6, 0.42),
    (11.8, -3.2, 5.7, 1.3, 0.8, 0.5),
    (35.8, -3.0, 5.4, 1.4, 0.9, 0.56),
]
if config.get("proof_stills"):
    cloud_specs = cloud_specs[:3]
for idx, (cx, cy, cz, sx, sy, sz) in enumerate(cloud_specs):
    cloud = add_uv_sphere(root, f"Cloud_{idx}", (cx, cy, cz), (sx, sy, sz), CLOUD_MAT)
    cloud_objects.append(cloud)
    puff = add_uv_sphere(root, f"CloudPuff_{idx}", (cx + 0.9, cy + 0.1, cz + 0.1), (sx * 0.7, sy * 0.7, sz * 0.7), CLOUD_MAT)
    cloud_objects.append(puff)

for collection in (sky_col, people_col, water_col, glass_col, space_col):
    for obj in collection.objects:
        set_obj_visible(obj, False)

timeline = config["timeline"]
segments = []
for seg in timeline:
    segments.append({
        "name": seg["name"],
        "start": seg["start"],
        "end": seg["end"],
        "shot_role": seg["shot_role"],
    })

beat = config["features"]["beat"]
bass = config["features"]["bass"]
highs = config["features"]["highs"]
energy = config["features"]["energy"]
onset = config["features"]["onset"]
debug_frames = config["debug_frames"]


def segment_at(t):
    for seg in segments:
        if seg["start"] <= t <= seg["end"]:
            return seg
    return segments[-1]


def family_objects(name):
    if name in {"blue_sky_aero_terminal"}:
        return list(sky_col.objects)
    if name in {"dancing_polygon_crowd"}:
        return list(people_col.objects)
    if name in {"splash_fountain_court", "ocean_glass_terrace"}:
        return list(water_col.objects)
    if name in {"glass_greenhouse", "glass_elevator_shaft"}:
        return list(glass_col.objects)
    if name in {"orbital_garden_station"}:
        return list(space_col.objects)
    return []


def set_zone_visibility(active_name):
    active = set(family_objects(active_name))
    for collection in (sky_col, people_col, water_col, glass_col, space_col):
        for obj in collection.objects:
            set_obj_visible(obj, obj in active)
    for obj in cloud_objects:
        visible = active_name in {"blue_sky_aero_terminal", "glass_greenhouse", "ocean_glass_terrace"}
        set_obj_visible(obj, visible)


def update_people(frame_idx, t):
    b = beat[frame_idx]
    on = onset[frame_idx]
    for idx, rig in enumerate(people_rigs):
        phase = t * 2.2 + idx * 0.7
        rig["root"].location.z = 0.08 + max(0.0, math.sin(phase)) * 0.12 + b * 0.08
        rig["root"].rotation_euler.z = math.sin(t * 0.7 + idx) * 0.12
        rig["l_arm"].rotation_euler.y = 0.6 + math.sin(phase) * 0.6 + on * 0.3
        rig["r_arm"].rotation_euler.y = -0.6 - math.sin(phase + 0.8) * 0.6 - on * 0.3
        rig["l_leg"].rotation_euler.y = math.sin(phase) * 0.42
        rig["r_leg"].rotation_euler.y = -math.sin(phase) * 0.42
        rig["shadow"].scale.x = 0.42 + max(0.0, math.sin(phase)) * 0.08


def update_water(frame_idx, t):
    b = bass[frame_idx]
    for idx, jet in enumerate(jets):
        jet.scale.z = 0.8 + b * 0.8 + max(0.0, math.sin(t * 1.6 + idx * 0.4)) * 0.7
        jet.location.z = 0.3 + jet.scale.z * 0.5
    ocean.rotation_euler.z = math.sin(t * 0.08) * 0.04


def update_clouds(frame_idx, t):
    for idx, obj in enumerate(cloud_objects):
        obj.location.x += math.sin(t * 0.08 + idx) * 0.002
        obj.location.z += math.sin(t * 0.11 + idx * 0.7) * 0.001


def camera_state(segment, local_t, frame_idx):
    base = zones.get(segment["name"], Vector((0.0, 0.0, 0.0)))
    e = energy[frame_idx]
    if segment["name"] == "dancing_polygon_crowd":
        angle = -1.0 + local_t * 2.0
        radius = 5.4 - e * 0.5
        cam = base + Vector((math.sin(angle) * radius, -math.cos(angle) * radius - 2.0, 2.6 + math.sin(local_t * math.pi) * 0.5))
        target = base + Vector((0.0, 0.0, 1.2))
        return cam, target
    if segment["name"] in {"splash_fountain_court", "ocean_glass_terrace"}:
        angle = -0.8 + local_t * 1.6
        radius = 6.0
        cam = base + Vector((math.sin(angle) * radius, -math.cos(angle) * radius - 2.6, 2.0 + local_t * 0.9))
        target = base + Vector((0.0, 0.8 if segment["name"] == "ocean_glass_terrace" else 0.0, 0.9))
        return cam, target
    if segment["name"] in {"glass_greenhouse", "glass_elevator_shaft"}:
        angle = -0.7 + local_t * 1.4
        radius = 5.8
        cam = base + Vector((math.sin(angle) * radius, -math.cos(angle) * radius - 2.2, 2.4 + local_t * 1.1))
        target = base + Vector((0.0, 0.2, 1.7))
        return cam, target
    if segment["name"] == "orbital_garden_station":
        angle = -1.2 + local_t * 2.2
        radius = 6.8
        cam = base + Vector((math.sin(angle) * radius, -math.cos(angle) * radius - 1.4, 3.1 + math.sin(local_t * math.pi) * 0.8))
        target = base + Vector((1.6, 1.0, 2.4))
        return cam, target
    angle = -0.9 + local_t * 1.8
    radius = 6.2
    cam = base + Vector((math.sin(angle) * radius, -math.cos(angle) * radius - 2.4, 2.6 + math.sin(local_t * math.pi) * 0.6))
    target = base + Vector((0.0, 0.0, 1.6))
    return cam, target


for frame in range(scene.frame_start, scene.frame_end + 1):
    if config.get("proof_stills"):
        break
    frame_idx = frame - 1
    t = frame_idx / config["fps"]
    segment = segment_at(t)
    local_t = 0.0
    if segment["end"] > segment["start"]:
        local_t = max(0.0, min(1.0, (t - segment["start"]) / (segment["end"] - segment["start"])))
    set_zone_visibility(segment["name"])
    update_people(frame_idx, t)
    update_water(frame_idx, t)
    update_clouds(frame_idx, t)
    cam, target = camera_state(segment, local_t, frame_idx)
    scene.camera.location = cam
    camera_target.location = target
    scene.frame_set(frame)
    scene.camera.keyframe_insert(data_path="location", frame=frame)
    camera_target.keyframe_insert(data_path="location", frame=frame)

debug_dir = Path(config["debug_frames_dir"]) if config["debug_frames_dir"] else None
if config.get("proof_stills"):
    if debug_dir is None:
        raise RuntimeError("proof_stills requires debug_frames_dir")
    debug_dir.mkdir(parents=True, exist_ok=True)
    for entry in debug_frames:
        frame = int(entry["frame"])
        name = entry["filename"]
        frame_idx = max(0, min(len(beat) - 1, frame - 1))
        t = frame_idx / config["fps"]
        segment = segment_at(t)
        local_t = 0.0
        if segment["end"] > segment["start"]:
            local_t = max(0.0, min(1.0, (t - segment["start"]) / (segment["end"] - segment["start"])))
        set_zone_visibility(segment["name"])
        update_people(frame_idx, t)
        update_water(frame_idx, t)
        update_clouds(frame_idx, t)
        cam, target = camera_state(segment, local_t, frame_idx)
        scene.camera.location = cam
        camera_target.location = target
        scene.frame_set(frame)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(debug_dir / Path(name).stem)
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for entry in debug_frames:
            frame = int(entry["frame"])
            name = entry["filename"]
            scene.frame_set(frame)
            scene.render.image_settings.file_format = "PNG"
            scene.render.filepath = str(debug_dir / Path(name).stem)
            bpy.ops.render.render(write_still=True)
'''


def render_video_blender(
    input_path: Path,
    output_path: Path,
    duration: float | None,
    width: int,
    height: int,
    fps: int,
    seed: int,
    preset: str,
    crf: int,
    audio_bitrate: str,
    analysis_backend: str,
    render_profile: str | None = None,
    aesthetic: str = "frutiger_cyber",
    scene_grammar: str = "worlds",
    debug_frames_dir: Path | None = None,
    debug_labels: bool = False,
    blender_proof_stills: bool = False,
    blender_smoke_scene: str | None = None,
    blender_quality: str = "proof",
    blender_diagnostic_engine: str | None = None,
) -> None:
    del seed, crf, audio_bitrate, debug_labels
    if preset == "city_promise":
        preset = "lofi"
    if preset not in {"worlds_material_proof", "lofi"}:
        raise RuntimeError(
            "Experimental Blender backend currently supports only --preset worlds_material_proof and --preset lofi."
        )
    if scene_grammar != "worlds":
        raise RuntimeError(
            "Experimental Blender backend currently supports only --scene-grammar worlds."
        )
    blender = _assert_blender()
    if (blender_proof_stills or blender_smoke_scene is not None) and debug_frames_dir is None:
        raise RuntimeError("Blender still/smoke modes require --debug-frames so PNGs, scripts, and logs have a destination directory.")
    if blender_smoke_scene is not None:
        assert debug_frames_dir is not None
        debug_frames_dir.mkdir(parents=True, exist_ok=True)
        if blender_proof_stills and blender_smoke_scene == "materials":
            payload = {
                "width": width,
                "height": height,
                "fps": fps,
                "debug_frames_dir": str(debug_frames_dir.resolve()),
                "metadata_output": str((debug_frames_dir.resolve() / "blender_metadata.json")),
                "warmup_output": str((debug_frames_dir.resolve() / "_warmup" / "warmup")),
                "render_settings": {
                    "engine": "eevee",
                    "quality": "proof",
                    "samples": 1,
                    "motion_blur": False,
                    "ssr": False,
                    "gtao": False,
                    "bloom": False,
                    "alpha_mode": "blend",
                    "shadow_method_alpha": "NONE",
                    "persistent_process": True,
                },
            }
            returncode, script_path, stdout_path, stderr_path = _run_blender_process(
                blender=blender,
                script_text=_build_blender_material_batch_proof_script(),
                payload=payload,
                artifact_dir=debug_frames_dir,
                timeout_seconds=420,
            )
            rendered = sorted(path for path in debug_frames_dir.glob("*.png") if not path.name.startswith("_warmup"))
            if returncode != 0:
                raise RuntimeError(
                    f"Blender material batch proof exited with return code {returncode}. "
                    f"Script: {script_path} Logs: {stdout_path}, {stderr_path}. "
                    f"Output dir contents: {[path.name for path in debug_frames_dir.iterdir()]}"
                )
            if len(rendered) < 6:
                raise RuntimeError(
                    f"Blender material batch proof produced {len(rendered)} PNGs, expected at least 6. "
                    f"Script: {script_path} Logs: {stdout_path}, {stderr_path}. "
                    f"Output dir contents: {[path.name for path in debug_frames_dir.iterdir()]}"
                )
            return
        smoke_output = debug_frames_dir.resolve() / f"{blender_smoke_scene}_smoke.png"
        chosen_quality = "smoke" if blender_quality == "proof" else blender_quality
        requested_engine = blender_diagnostic_engine or ("workbench" if blender_smoke_scene == "cube" else "eevee")
        payload = {
            "width": width,
            "height": height,
            "fps": fps,
            "output_path": str(smoke_output.with_suffix("")),
            "smoke_scene": blender_smoke_scene,
            "quality": chosen_quality,
            "diagnostic_engine": requested_engine,
            "metadata_output": str((debug_frames_dir.resolve() / "blender_metadata.json")),
            "render_settings": {
                "engine": requested_engine,
                "quality": chosen_quality,
                "samples": 1,
                "motion_blur": False,
                "ssr": False,
                "gtao": False,
                "bloom": False,
                "alpha_mode": "blend",
                "shadow_method_alpha": "NONE",
            },
        }
        script_text = _build_blender_cube_smoke_script() if blender_smoke_scene == "cube" else _build_blender_material_smoke_script()
        returncode, script_path, stdout_path, stderr_path = _run_blender_process(
            blender=blender,
            script_text=script_text,
            payload=payload,
            artifact_dir=debug_frames_dir,
            timeout_seconds=300 if blender_smoke_scene == "cube" else 300,
        )
        rendered = sorted(debug_frames_dir.glob("*.png"))
        if returncode != 0:
            raise RuntimeError(
                f"Blender smoke scene '{blender_smoke_scene}' exited with return code {returncode}. "
                f"Script: {script_path} Logs: {stdout_path}, {stderr_path}. "
                f"Output dir contents: {[path.name for path in debug_frames_dir.iterdir()]}"
            )
        if not rendered:
            raise RuntimeError(
                f"Blender smoke scene '{blender_smoke_scene}' exited cleanly but produced no PNGs. "
                f"Script: {script_path} Logs: {stdout_path}, {stderr_path}. "
                f"Output dir contents: {[path.name for path in debug_frames_dir.iterdir()]}"
            )
        return
    if blender_proof_stills and debug_frames_dir is None:
        raise RuntimeError("Blender proof stills require --debug-frames so PNGs have a destination directory.")
    analysis = analyze_audio(input_path, fps=fps, duration_limit=duration, backend=analysis_backend)
    blender_city_preview = preset == "lofi"
    timeline = build_timeline(analysis.duration, preset=preset, scene_grammar=scene_grammar)
    city_sequence = _build_blender_city_sequence(analysis.duration) if blender_city_preview else []
    if blender_city_preview:
        debug_entries = _city_debug_entries(city_sequence, analysis.duration, fps)
        debug_targets = [max(0.0, (int(entry["frame"]) - 1) / fps) for entry in debug_entries]
    else:
        debug_targets = debug_targets_for_timeline(analysis.duration, preset, timeline)
        debug_entries = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if debug_frames_dir is not None:
        debug_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count = len(analysis.frame_times)
    debug_entries = list(debug_entries)
    proof_frames = [1, 12, 24, 36, 48, 60] if blender_proof_stills else []
    targets = debug_targets
    if blender_proof_stills:
        targets = [min(analysis.duration, max(0.0, (frame - 1) / fps)) for frame in proof_frames if frame <= frame_count]
    if not blender_city_preview or blender_proof_stills:
        debug_entries = []
        for target in targets:
            frame_index = min(frame_count - 1, max(0, round(target * fps)))
            segment_name = "none"
            for segment in timeline:
                if segment.start <= target <= segment.end:
                    segment_name = segment.name
                    break
            debug_entries.append(
                {
                    "frame": frame_index + 1,
                    "filename": f"t_{target:05.2f}s__{segment_name}.png",
                }
            )

    payload = {
        "audio_path": str(input_path),
        "output_path": str(output_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": analysis.duration,
        "render_profile": render_profile or "final",
        "aesthetic": aesthetic,
        "proof_stills": blender_proof_stills,
        "proof_frame_numbers": [entry["frame"] for entry in debug_entries],
        "debug_frames_dir": str(debug_frames_dir.resolve()) if debug_frames_dir is not None else None,
        "debug_frames": debug_entries,
        "metadata_output": str((debug_frames_dir / "blender_metadata.json").resolve()) if debug_frames_dir is not None else str((output_path.parent / f"{output_path.stem}_blender_metadata.json").resolve()),
        "warmup_output": str(((debug_frames_dir or output_path.parent) / "_warmup" / "warmup").resolve()),
        "city_preview": blender_city_preview,
        "city_sequence": city_sequence,
        "timeline": [
            {
                "name": segment.name,
                "start": segment.start,
                "end": segment.end,
                "shot_role": segment.shot_role,
            }
            for segment in timeline
        ],
        "features": {
            "beat": analysis.beat.tolist(),
            "bass": analysis.bass.tolist(),
            "mids": analysis.mids.tolist(),
            "highs": analysis.highs.tolist(),
            "energy": analysis.energy.tolist(),
            "onset": analysis.onset.tolist(),
        },
    }

    if blender_city_preview:
        ffmpeg = _assert_ffmpeg()
        with tempfile.TemporaryDirectory(prefix="citypromise_blender_city_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            frame_dir = tmp_root / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            payload["frame_output_pattern"] = str((frame_dir / "frame_####").resolve())
            artifact_dir = debug_frames_dir if debug_frames_dir is not None else (output_path.parent / f"{output_path.stem}_blender_artifacts")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            city_timeout_seconds = max(2400, int(frame_count * 12 + 600))
            returncode, script_path, stdout_path, stderr_path = _run_blender_process(
                blender=blender,
                script_text=_build_blender_material_batch_proof_script(),
                payload=payload,
                artifact_dir=artifact_dir,
                timeout_seconds=city_timeout_seconds,
            )
            first_frame = frame_dir / "frame_0001.png"
            if returncode != 0:
                raise RuntimeError(
                    f"Blender city preview exited with return code {returncode}. "
                    f"Script: {script_path} Logs: {stdout_path}, {stderr_path}."
                )
            if not first_frame.exists():
                raise RuntimeError(
                    f"Blender city preview exited cleanly but produced no frame images. "
                    f"Script: {script_path} Logs: {stdout_path}, {stderr_path}."
                )
            encode_command = [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame_%04d.png"),
                "-i",
                str(input_path),
                "-t",
                str(analysis.duration),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_path),
            ]
            subprocess.run(encode_command, check=True)
            return

    ffmpeg = _assert_ffmpeg() if not blender_proof_stills else ""
    with tempfile.TemporaryDirectory(prefix="citypromise_blender_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        config_path = tmp_root / "config.json"
        script_path = tmp_root / "driver.py"
        frame_dir = tmp_root / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path.write_text(_build_blender_script(), encoding="utf-8")
        payload["frame_output_pattern"] = str(frame_dir / "frame_####")
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        command = [
            blender,
            "--background",
            "--factory-startup",
            "--python",
            str(script_path),
            "--",
            str(config_path),
        ]
        try:
            subprocess.run(command, check=True, timeout=180 if blender_proof_stills else 1800)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Blender backend timed out after {int(exc.timeout)}s before producing the requested proof output. "
                f"Any partial PNGs would be under {debug_frames_dir if debug_frames_dir is not None else frame_dir}."
            ) from exc
        if blender_proof_stills:
            rendered = sorted((debug_frames_dir or frame_dir).glob("*.png")) if (debug_frames_dir or frame_dir).exists() else []
            if not rendered:
                raise RuntimeError(
                    "Blender proof-stills mode exited without producing PNG frames. "
                    "Check Blender stdout/stderr above for the underlying scene-script failure."
                )
            return
        first_frame = frame_dir / "frame_0001.png"
        if not first_frame.exists():
            raise RuntimeError(
                "Blender proof backend did not produce any frame images. "
                "Check Blender stdout/stderr above for the underlying scene-script failure."
            )
        encode_command = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-i",
            str(input_path),
            "-t",
            str(analysis.duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
        subprocess.run(encode_command, check=True)
