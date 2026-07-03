import argparse
import json
import time
from pathlib import Path

import bpy


def _available_engines(scene):
    return list(scene.render.bl_rna.properties["engine"].enum_items.keys())


def _choose_engine(scene, requested: str) -> tuple[str | None, str]:
    available = _available_engines(scene)
    mapping = {
        "workbench": ["BLENDER_WORKBENCH"],
        "eevee": ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
        "opengl": ["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
    }
    for candidate in mapping.get(requested, []):
        if candidate in available:
            scene.render.engine = candidate
            return candidate, "opengl" if requested == "opengl" else "render"
    return None, "opengl" if requested == "opengl" else "render"


def _gpu_info() -> dict[str, object]:
    info: dict[str, object] = {}
    try:
        import gpu  # type: ignore

        platform = gpu.platform
        info["gpu_vendor"] = platform.vendor_get()
        info["gpu_renderer"] = platform.renderer_get()
        info["gpu_version"] = platform.version_get()
        info["gpu_backend_type"] = platform.backend_type_get()
        info["gpu_device_type"] = platform.device_type_get()
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["gpu_info_error"] = repr(exc)
    try:
        prefs = bpy.context.preferences.system
        for attr in (
            "gpu_backend",
            "viewport_aa",
            "use_overlay_smooth_wire",
            "use_edit_mode_smooth_wire",
        ):
            if hasattr(prefs, attr):
                info[f"prefs_{attr}"] = getattr(prefs, attr)
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["prefs_error"] = repr(exc)
    return info


def _setup_scene(width: int, height: int, requested_engine: str) -> tuple[dict[str, object], str | None, str]:
    started = time.perf_counter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    available = _available_engines(scene)
    actual_engine, render_method = _choose_engine(scene, requested_engine)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
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

    if actual_engine == "BLENDER_EEVEE":
        eevee = scene.eevee
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = 1
        if hasattr(eevee, "taa_samples"):
            eevee.taa_samples = 1
        if hasattr(eevee, "use_bloom"):
            eevee.use_bloom = False
        if hasattr(eevee, "use_gtao"):
            eevee.use_gtao = False
        if hasattr(eevee, "use_ssr"):
            eevee.use_ssr = False
        if hasattr(eevee, "use_ssr_refraction"):
            eevee.use_ssr_refraction = False
        if hasattr(eevee, "use_shadows"):
            eevee.use_shadows = False
        if hasattr(eevee, "shadow_pool_size"):
            eevee.shadow_pool_size = "256"
        if hasattr(eevee, "use_motion_blur"):
            eevee.use_motion_blur = False
        if hasattr(eevee, "volumetric_tile_size"):
            eevee.volumetric_tile_size = "8"

    world = bpy.data.worlds.new("DiagWorld")
    scene.world = world
    world.color = (0.70, 0.78, 0.88)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    camera.location = (6.4, -7.8, 4.2)
    camera.rotation_euler = (1.08, 0.0, 0.66)
    camera.data.lens = 34
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    light = bpy.data.objects.new("Sun", light_data)
    light.rotation_euler = (0.92, 0.0, -0.74)
    light.data.energy = 1.5
    scene.collection.objects.link(light)

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, -1.1))
    floor = bpy.context.active_object
    floor_mat = bpy.data.materials.new("Floor")
    floor_mat.diffuse_color = (0.72, 0.76, 0.80, 1.0)
    floor.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.1))
    cube = bpy.context.active_object
    cube.rotation_euler = (0.58, 0.18, 0.72)
    cube.scale = (1.4, 1.1, 1.5)
    cube_mat = bpy.data.materials.new("Cube")
    cube_mat.diffuse_color = (0.24, 0.50, 0.82, 1.0)
    cube.data.materials.append(cube_mat)

    if actual_engine == "BLENDER_WORKBENCH":
        wb = scene.display.shading
        wb.light = "STUDIO"
        wb.color_type = "MATERIAL"
        wb.show_specular_highlight = False
        wb.show_object_outline = False
        if hasattr(scene.display, "render_aa"):
            scene.display.render_aa = "OFF"

    setup = {
        "blender_version": bpy.app.version_string,
        "available_render_engines": available,
        "requested_engine": requested_engine,
        "actual_engine": actual_engine,
        "render_method": render_method,
        "setup_seconds": round(time.perf_counter() - started, 3),
        "width": width,
        "height": height,
        "gpu_info": _gpu_info(),
    }
    return setup, actual_engine, render_method


def _build_material_scene() -> None:
    scene = bpy.context.scene
    world = scene.world
    if world is not None:
        world.use_nodes = False
        world.color = (0.70, 0.80, 0.92)

    camera = scene.camera
    assert camera is not None
    camera.location = (5.8, -7.1, 4.3)
    camera.rotation_euler = (1.02, 0.0, 0.66)
    camera.data.lens = 30

    for obj in list(scene.objects):
        if obj.type in {"MESH"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=18.0, location=(0.0, 0.0, -1.1))
    floor = bpy.context.active_object
    floor_mat = bpy.data.materials.new("Floor")
    floor_mat.use_nodes = True
    floor_mat.blend_method = "OPAQUE"
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.76, 0.82, 0.88, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.2
    floor.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 1.2, -0.02))
    water = bpy.context.active_object
    water.scale = (1.8, 1.0, 1.0)
    water_mat = bpy.data.materials.new("Water")
    water_mat.use_nodes = True
    if hasattr(water_mat, "blend_method"):
        water_mat.blend_method = "BLEND"
    if hasattr(water_mat, "shadow_method"):
        water_mat.shadow_method = "NONE"
    wbsdf = water_mat.node_tree.nodes["Principled BSDF"]
    wbsdf.inputs["Base Color"].default_value = (0.18, 0.48, 0.90, 1.0)
    wbsdf.inputs["Alpha"].default_value = 0.72
    wbsdf.inputs["Roughness"].default_value = 0.08
    water.data.materials.append(water_mat)

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(1.8, 0.2, 1.7), rotation=(0.0, 0.28, 0.0))
    glass = bpy.context.active_object
    glass.scale = (0.8, 1.1, 1.0)
    glass_mat = bpy.data.materials.new("Glass")
    glass_mat.use_nodes = True
    if hasattr(glass_mat, "blend_method"):
        glass_mat.blend_method = "BLEND"
    if hasattr(glass_mat, "shadow_method"):
        glass_mat.shadow_method = "NONE"
    gbsdf = glass_mat.node_tree.nodes["Principled BSDF"]
    gbsdf.inputs["Base Color"].default_value = (0.76, 0.94, 1.0, 1.0)
    gbsdf.inputs["Alpha"].default_value = 0.34
    gbsdf.inputs["Roughness"].default_value = 0.05
    glass.data.materials.append(glass_mat)

    bpy.ops.mesh.primitive_cube_add(location=(-0.2, -0.1, 0.8), rotation=(0.36, 0.14, 0.44))
    prism = bpy.context.active_object
    prism.scale = (0.6, 0.38, 0.95)
    prism_mat = bpy.data.materials.new("Plastic")
    prism_mat.use_nodes = True
    if hasattr(prism_mat, "blend_method"):
        prism_mat.blend_method = "BLEND"
    if hasattr(prism_mat, "shadow_method"):
        prism_mat.shadow_method = "NONE"
    pbsdf = prism_mat.node_tree.nodes["Principled BSDF"]
    pbsdf.inputs["Base Color"].default_value = (0.40, 0.82, 0.96, 1.0)
    pbsdf.inputs["Alpha"].default_value = 0.78
    pbsdf.inputs["Roughness"].default_value = 0.12
    prism.data.materials.append(prism_mat)

    bpy.ops.mesh.primitive_cube_add(location=(-0.2, -0.1, 0.8), rotation=(0.36, 0.14, 0.44))
    core = bpy.context.active_object
    core.scale = (0.26, 0.16, 0.42)
    core_mat = bpy.data.materials.new("Core")
    core_mat.use_nodes = True
    cbsdf = core_mat.node_tree.nodes["Principled BSDF"]
    cbsdf.inputs["Base Color"].default_value = (0.12, 0.18, 0.30, 1.0)
    cbsdf.inputs["Roughness"].default_value = 0.24
    core.data.materials.append(core_mat)

    for cx, cy, cz, scale in [(-1.0, -2.5, 4.8, (0.9, 0.56, 0.32)), (1.4, -2.8, 5.2, (1.1, 0.62, 0.36))]:
        for ox, oy, oz in [(-0.7, 0.0, 0.0), (0.0, 0.14, 0.12), (0.7, -0.06, 0.0)]:
            bpy.ops.mesh.primitive_uv_sphere_add(location=(cx + ox, cy + oy, cz + oz), segments=10, ring_count=6)
            cloud = bpy.context.active_object
            cloud.scale = scale
            cloud_mat = bpy.data.materials.new(f"Cloud_{cx}_{ox}")
            cloud_mat.use_nodes = True
            if hasattr(cloud_mat, "blend_method"):
                cloud_mat.blend_method = "BLEND"
            if hasattr(cloud_mat, "shadow_method"):
                cloud_mat.shadow_method = "NONE"
            clbsdf = cloud_mat.node_tree.nodes["Principled BSDF"]
            clbsdf.inputs["Base Color"].default_value = (0.97, 0.98, 1.0, 1.0)
            clbsdf.inputs["Alpha"].default_value = 0.92
            clbsdf.inputs["Roughness"].default_value = 0.9
            cloud.data.materials.append(cloud_mat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine", choices=("workbench", "eevee", "opengl"), default="eevee")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--scene", choices=("cube", "materials"), default="cube")
    import sys

    argv = sys.argv
    args = parser.parse_args(argv[argv.index("--") + 1 :] if "--" in argv else [])

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_info, actual_engine, render_method = _setup_scene(args.width, args.height, args.engine)
    scene = bpy.context.scene
    if args.scene == "materials":
        _build_material_scene()
    frame_rows: list[dict[str, object]] = []
    total_started = time.perf_counter()

    for idx in range(1, args.frames + 1):
        png_path = output_dir / f"frame_{idx:03d}.png"
        scene.render.filepath = str(png_path.with_suffix(""))
        started = time.perf_counter()
        if render_method == "opengl":
            bpy.ops.render.opengl(write_still=True, view_context=False)
        else:
            bpy.ops.render.render(write_still=True)
        elapsed = round(time.perf_counter() - started, 3)
        exists = png_path.exists()
        frame_rows.append(
            {
                "frame": idx,
                "render_call_seconds": elapsed,
                "png_path": str(png_path),
                "png_exists": exists,
                "png_size": png_path.stat().st_size if exists else 0,
            }
        )

    metadata = {
        **setup_info,
        "total_script_seconds": round(time.perf_counter() - total_started + float(setup_info["setup_seconds"]), 3),
        "frame_rows": frame_rows,
        "scene": args.scene,
        "output_dir_listing": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "diag_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
