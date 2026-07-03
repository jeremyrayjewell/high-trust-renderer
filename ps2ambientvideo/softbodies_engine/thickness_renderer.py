from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from panda3d.core import BitMask32, CardMaker, CullFaceAttrib, Filename, NodePath, Shader, Texture, Vec4


@dataclass(frozen=True)
class ThicknessTextureSpec:
    name: str
    format_name: str
    component_type: str


def get_thickness_texture_specs() -> tuple[ThicknessTextureSpec, ...]:
    return (
        ThicknessTextureSpec("scene_color", "F_rgba16", "T_float"),
        ThicknessTextureSpec("front_depth", "F_r32", "T_float"),
        ThicknessTextureSpec("back_depth", "F_r32", "T_float"),
        ThicknessTextureSpec("front_normal", "F_rgba16", "T_float"),
        ThicknessTextureSpec("front_local", "F_rgba16", "T_float"),
    )


def compute_thickness(front_depth: np.ndarray, back_depth: np.ndarray) -> np.ndarray:
    return np.maximum(back_depth - front_depth, 0.0)


def compute_absorption_transmittance(absorption_color: np.ndarray, thickness: np.ndarray, density: float) -> np.ndarray:
    return np.exp(-absorption_color[None, None, :] * thickness[:, :, None] * density)


def normalize_debug_scalar(values: np.ndarray, max_value: float) -> np.ndarray:
    return np.clip(values / max(max_value, 1e-6), 0.0, 1.0)


def compute_texture_uv_scale(buffer_size: tuple[int, int], texture_size: tuple[int, int]) -> tuple[float, float]:
    if texture_size[0] <= 0 or texture_size[1] <= 0:
        return (1.0, 1.0)
    return (
        min(buffer_size[0] / texture_size[0], 1.0),
        min(buffer_size[1] / texture_size[1], 1.0),
    )


class ThicknessCompositePipeline:
    def __init__(self, base, width: int, height: int) -> None:
        from panda3d.core import Texture
        shader_dir = Path(__file__).resolve().parent / "shaders"

        self.base = base
        self.width = width
        self.height = height
        self.env_mask = BitMask32.bit(1)
        self.gel_mask = BitMask32.bit(2)
        self.env_root = self.base.render.attachNewNode("thickness-env-root")
        self.gel_root = self.base.render.attachNewNode("thickness-gel-root")

        self.scene_color_tex = Texture("scene-color")
        self.scene_color_tex.setFormat(Texture.F_rgba16)
        self.scene_color_tex.setComponentType(Texture.T_float)

        self.front_depth_tex = Texture("front-depth")
        self.front_depth_tex.setFormat(Texture.F_r32)
        self.front_depth_tex.setComponentType(Texture.T_float)

        self.back_depth_tex = Texture("back-depth")
        self.back_depth_tex.setFormat(Texture.F_r32)
        self.back_depth_tex.setComponentType(Texture.T_float)

        self.front_normal_tex = Texture("front-normal")
        self.front_normal_tex.setFormat(Texture.F_rgba16)
        self.front_normal_tex.setComponentType(Texture.T_float)

        self.front_local_tex = Texture("front-local")
        self.front_local_tex.setFormat(Texture.F_rgba16)
        self.front_local_tex.setComponentType(Texture.T_float)

        self.front_depth_buffer = self.base.win.makeTextureBuffer("gel-front-depth", width, height, self.front_depth_tex)
        self.back_depth_buffer = self.base.win.makeTextureBuffer("gel-back-depth", width, height, self.back_depth_tex)
        self.front_normal_buffer = self.base.win.makeTextureBuffer("gel-front-normal", width, height, self.front_normal_tex)
        self.front_local_buffer = self.base.win.makeTextureBuffer("gel-front-local", width, height, self.front_local_tex)
        self.scene_color_buffer = self.base.win.makeTextureBuffer("scene-color-buffer", width, height, self.scene_color_tex)

        self.front_depth_camera = self.base.makeCamera(self.front_depth_buffer, camName="gelFrontDepthCam", scene=self.gel_root)
        self.back_depth_camera = self.base.makeCamera(self.back_depth_buffer, camName="gelBackDepthCam", scene=self.gel_root)
        self.front_normal_camera = self.base.makeCamera(self.front_normal_buffer, camName="gelFrontNormalCam", scene=self.gel_root)
        self.front_local_camera = self.base.makeCamera(self.front_local_buffer, camName="gelFrontLocalCam", scene=self.gel_root)
        self.scene_color_camera = self.base.makeCamera(self.scene_color_buffer, camName="gelSceneColorCam", scene=self.env_root)

        self.depth_shader = Shader.load(Shader.SL_GLSL, Filename.fromOsSpecific(str(shader_dir / "thickness_depth.vert")), Filename.fromOsSpecific(str(shader_dir / "thickness_depth.frag")))
        self.normal_shader = Shader.load(Shader.SL_GLSL, Filename.fromOsSpecific(str(shader_dir / "thickness_normal.vert")), Filename.fromOsSpecific(str(shader_dir / "thickness_normal.frag")))
        self.local_shader = Shader.load(Shader.SL_GLSL, Filename.fromOsSpecific(str(shader_dir / "thickness_local.vert")), Filename.fromOsSpecific(str(shader_dir / "thickness_local.frag")))
        self.composite_shader = Shader.load(Shader.SL_GLSL, Filename.fromOsSpecific(str(shader_dir / "thickness_composite.vert")), Filename.fromOsSpecific(str(shader_dir / "thickness_composite.frag")))

        self._configure_depth_camera(self.front_depth_camera, self.depth_shader, cull_back=True)
        self._configure_depth_camera(self.back_depth_camera, self.depth_shader, cull_back=False)
        self._configure_depth_camera(self.front_normal_camera, self.normal_shader, cull_back=True)
        self._configure_depth_camera(self.front_local_camera, self.local_shader, cull_back=True)

        self.scene_color_buffer.setClearColor(Vec4(0.0, 0.0, 0.0, 1.0))
        self.front_depth_buffer.setClearColor(Vec4(0.0, 0.0, 0.0, 0.0))
        self.back_depth_buffer.setClearColor(Vec4(0.0, 0.0, 0.0, 0.0))
        self.front_normal_buffer.setClearColor(Vec4(0.5, 0.5, 1.0, 0.0))
        self.front_local_buffer.setClearColor(Vec4(0.5, 0.5, 0.5, 0.0))

        self.fullscreen_quad = self._create_fullscreen_quad()
        self.base.cam.node().setCameraMask(BitMask32.allOff())
        self._update_uv_scale()

    def _configure_depth_camera(self, camera_np: NodePath, shader: Shader, cull_back: bool) -> None:
        camera_np.node().setInitialState(self.base.render.getState())
        state_np = NodePath("thickness-state")
        state_np.setShader(shader)
        if cull_back:
            state_np.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
        else:
            state_np.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullClockwise))
        camera_np.node().setInitialState(state_np.getState())

    def _create_fullscreen_quad(self) -> NodePath:
        card = CardMaker("thickness-composite-card")
        card.setFrameFullscreenQuad()
        quad = self.base.render2d.attachNewNode(card.generate())
        quad.setShader(self.composite_shader)
        quad.setShaderInput("scene_color_tex", self.scene_color_tex)
        quad.setShaderInput("front_depth_tex", self.front_depth_tex)
        quad.setShaderInput("back_depth_tex", self.back_depth_tex)
        quad.setShaderInput("front_normal_tex", self.front_normal_tex)
        quad.setShaderInput("front_local_tex", self.front_local_tex)
        return quad

    def _update_uv_scale(self) -> None:
        scale_x, scale_y = compute_texture_uv_scale(
            (int(self.scene_color_buffer.getXSize()), int(self.scene_color_buffer.getYSize())),
            (int(self.scene_color_tex.getXSize()), int(self.scene_color_tex.getYSize())),
        )
        self.fullscreen_quad.setShaderInput("texture_uv_scale", Vec4(scale_x, scale_y, 0.0, 0.0))

    def update_camera(self, source_camera: NodePath) -> None:
        lens_source = source_camera
        if not hasattr(lens_source.node(), "getLens"):
            lens_source = self.base.cam
        lens = lens_source.node().getLens()
        for camera_np in (self.front_depth_camera, self.back_depth_camera, self.front_normal_camera, self.front_local_camera, self.scene_color_camera):
            camera_np.setMat(lens_source.getMat())
            camera_np.node().setLens(lens)
        self.fullscreen_quad.setShaderInput("near_far", Vec4(lens.getNear(), lens.getFar(), 0.0, 0.0))

    def set_material_inputs(
        self,
        absorption_color: tuple[float, float, float],
        absorption_density: float,
        transmission_gain: float,
        scattering_strength: float,
        cloudiness: float,
        refraction_strength: float,
        ior: float,
        surface_opacity: float,
        specular_strength: float,
        fresnel_strength: float,
        surface_reflection_strength: float,
        thickness_scale: float,
    ) -> None:
        self.fullscreen_quad.setShaderInput("absorption_color", Vec4(absorption_color[0], absorption_color[1], absorption_color[2], 1.0))
        self.fullscreen_quad.setShaderInput("absorption_density", float(absorption_density))
        self.fullscreen_quad.setShaderInput("transmission_gain", float(transmission_gain))
        self.fullscreen_quad.setShaderInput("scattering_strength", float(scattering_strength))
        self.fullscreen_quad.setShaderInput("cloudiness", float(cloudiness))
        self.fullscreen_quad.setShaderInput("refraction_strength", float(refraction_strength))
        self.fullscreen_quad.setShaderInput("ior", float(ior))
        self.fullscreen_quad.setShaderInput("surface_opacity", float(surface_opacity))
        self.fullscreen_quad.setShaderInput("specular_strength", float(specular_strength))
        self.fullscreen_quad.setShaderInput("fresnel_strength", float(fresnel_strength))
        self.fullscreen_quad.setShaderInput("surface_reflection_strength", float(surface_reflection_strength))
        self.fullscreen_quad.setShaderInput("thickness_scale", float(thickness_scale))
        self.fullscreen_quad.setShaderInput("debug_depth_max", 10.0)
        self.fullscreen_quad.setShaderInput("debug_thickness_max", 1.8)

    def set_time_inputs(self, sim_time: float) -> None:
        del sim_time

    def set_view_mode(self, view_name: str) -> None:
        modes = {
            "composite": 0.0,
            "front-depth": 1.0,
            "back-depth": 2.0,
            "thickness": 3.0,
            "normals": 4.0,
            "refraction-offset": 5.0,
            "transmittance": 6.0,
        }
        self.fullscreen_quad.setShaderInput("view_mode", float(modes[view_name]))

    def attach_environment(self, node: NodePath) -> None:
        node.wrtReparentTo(self.env_root)

    def attach_gelatin(self, node: NodePath) -> None:
        node.wrtReparentTo(self.gel_root)
