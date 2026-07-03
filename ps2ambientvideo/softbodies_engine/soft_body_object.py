from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from panda3d.bullet import (
    BulletRigidBodyNode,
    BulletSoftBodyConfig,
    BulletSoftBodyNode,
    BulletSoftBodyWorldInfo,
)
from panda3d.core import (
    Geom,
    GeomNode,
    GeomPoints,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    PTA_LVecBase3f,
    PTA_int,
    Vec3,
    Vec4,
)

from .gummy_object import ShapeMeshData, _compute_vertex_normals, _make_uvs_from_bounds, _orient_triangles_outward, _signed_power, generate_shape_mesh_data, validate_shape_mesh_data


@dataclass(frozen=True)
class SoftSimulationMesh:
    points: np.ndarray
    surface_triangles: np.ndarray
    tetrahedra: np.ndarray | None = None
    rest_volume: float = 0.0


@dataclass(frozen=True)
class RenderBinding:
    indices: np.ndarray
    weights: np.ndarray
    node_indices: np.ndarray


@dataclass(frozen=True)
class SoftBodyPreset:
    name: str
    linear_stiffness: float
    angular_stiffness: float
    volume_preservation: float
    pose_matching: float
    pressure: float
    volume_conservation: float
    damping: float
    position_iterations: int
    velocity_iterations: int
    drift_iterations: int
    cluster_iterations: int
    cluster_count: int


@dataclass(frozen=True)
class SoftBodyMetrics:
    rest_volume: float
    current_volume: float
    volume_ratio: float
    center_of_mass: np.ndarray
    raw_max_displacement: float
    raw_mean_displacement: float
    aligned_rms_deformation: float
    aligned_max_deformation: float
    max_velocity: float
    mean_velocity: float
    kinetic_energy_proxy: float
    bounding_box_dimensions: np.ndarray
    top_region_deformation: float
    bottom_region_deformation: float
    struck_corner_deformation: float
    opposite_corner_deformation: float
    twist_angle_proxy: float
    center_of_mass_drift: float
    minimum_node_height: float
    nodes_below_floor: int
    min_triangle_area_proxy: float
    max_edge_stretch_ratio: float
    min_edge_compression_ratio: float
    has_non_finite: bool
    node_count: int


SOFTBODY_PRESETS: dict[str, SoftBodyPreset] = {
    "soft": SoftBodyPreset(
        name="soft",
        linear_stiffness=0.30,
        angular_stiffness=0.16,
        volume_preservation=0.38,
        pose_matching=0.0,
        pressure=2.4,
        volume_conservation=0.7,
        damping=0.04,
        position_iterations=10,
        velocity_iterations=4,
        drift_iterations=2,
        cluster_iterations=2,
        cluster_count=4,
    ),
    "medium": SoftBodyPreset(
        name="medium",
        linear_stiffness=0.48,
        angular_stiffness=0.28,
        volume_preservation=0.56,
        pose_matching=0.04,
        pressure=4.8,
        volume_conservation=1.6,
        damping=0.07,
        position_iterations=12,
        velocity_iterations=6,
        drift_iterations=3,
        cluster_iterations=3,
        cluster_count=6,
    ),
    "firm": SoftBodyPreset(
        name="firm",
        linear_stiffness=0.60,
        angular_stiffness=0.40,
        volume_preservation=0.70,
        pose_matching=0.10,
        pressure=7.2,
        volume_conservation=2.8,
        damping=0.11,
        position_iterations=14,
        velocity_iterations=7,
        drift_iterations=4,
        cluster_iterations=4,
        cluster_count=8,
    ),
    "stable_soft": SoftBodyPreset(
        name="stable_soft",
        linear_stiffness=0.44,
        angular_stiffness=0.22,
        volume_preservation=0.60,
        pose_matching=0.03,
        pressure=4.0,
        volume_conservation=1.5,
        damping=0.09,
        position_iterations=14,
        velocity_iterations=6,
        drift_iterations=3,
        cluster_iterations=4,
        cluster_count=6,
    ),
    "stable_medium": SoftBodyPreset(
        name="stable_medium",
        linear_stiffness=0.56,
        angular_stiffness=0.33,
        volume_preservation=0.74,
        pose_matching=0.06,
        pressure=5.8,
        volume_conservation=2.7,
        damping=0.13,
        position_iterations=18,
        velocity_iterations=8,
        drift_iterations=4,
        cluster_iterations=5,
        cluster_count=9,
    ),
    "stable_firm": SoftBodyPreset(
        name="stable_firm",
        linear_stiffness=0.62,
        angular_stiffness=0.38,
        volume_preservation=0.80,
        pose_matching=0.08,
        pressure=6.6,
        volume_conservation=3.4,
        damping=0.15,
        position_iterations=20,
        velocity_iterations=9,
        drift_iterations=5,
        cluster_iterations=5,
        cluster_count=10,
    ),
}

TETRA_PRESETS: dict[str, SoftBodyPreset] = {
    "soft": SoftBodyPreset(
        name="soft",
        linear_stiffness=0.28,
        angular_stiffness=0.12,
        volume_preservation=0.62,
        pose_matching=0.02,
        pressure=0.0,
        volume_conservation=0.8,
        damping=0.10,
        position_iterations=10,
        velocity_iterations=4,
        drift_iterations=2,
        cluster_iterations=2,
        cluster_count=4,
    ),
    "medium": SoftBodyPreset(
        name="medium",
        linear_stiffness=0.48,
        angular_stiffness=0.28,
        volume_preservation=0.88,
        pose_matching=0.04,
        pressure=0.0,
        volume_conservation=1.8,
        damping=0.16,
        position_iterations=14,
        velocity_iterations=6,
        drift_iterations=2,
        cluster_iterations=3,
        cluster_count=8,
    ),
    "firm": SoftBodyPreset(
        name="firm",
        linear_stiffness=0.58,
        angular_stiffness=0.36,
        volume_preservation=0.94,
        pose_matching=0.08,
        pressure=0.0,
        volume_conservation=2.6,
        damping=0.18,
        position_iterations=16,
        velocity_iterations=7,
        drift_iterations=3,
        cluster_iterations=4,
        cluster_count=10,
    ),
}


def build_surface_cube_simulation_mesh(radius: float, seed: int, segments: int = 8, detail_multiplier: float = 1.0) -> SoftSimulationMesh:
    del seed
    half = radius * 0.80
    mesh = _build_low_detail_rounded_box_mesh(radius=radius, half=half, segments=max(4, int(round(segments * max(detail_multiplier, 0.35)))))
    validate_shape_mesh_data("surface_sim_rounded_cube", mesh)
    return SoftSimulationMesh(
        points=mesh.vertices.astype(np.float32),
        surface_triangles=mesh.indices.astype(np.int32),
        tetrahedra=None,
        rest_volume=compute_surface_volume(mesh.vertices, mesh.indices),
    )


def _build_low_detail_rounded_box_mesh(radius: float, half: float, segments: int) -> ShapeMeshData:
    half_extents = np.asarray([half, half, half], dtype=np.float32)
    lat_segments = max(4, segments)
    lon_segments = max(8, segments * 2)
    boxiness = 0.42 + 0.10 * np.clip((radius * 0.16) / max(float(np.min(half_extents)), 1e-6), 0.0, 1.0)
    omegas = np.linspace(-np.pi, np.pi, lon_segments, endpoint=False, dtype=np.float32)

    vertices: list[list[float]] = []
    indices: list[list[int]] = []
    vertices.append([0.0, 0.0, float(-half_extents[2])])
    south_index = 0
    ring_starts: list[int] = []
    etas = np.linspace(-0.5 * np.pi, 0.5 * np.pi, lat_segments + 1, dtype=np.float32)[1:-1]
    for eta in etas:
        ring_starts.append(len(vertices))
        ce = np.cos(eta)
        se = np.sin(eta)
        ce_term = _signed_power(np.array([ce], dtype=np.float32), boxiness)[0]
        se_term = _signed_power(np.array([se], dtype=np.float32), boxiness)[0]
        for omega in omegas:
            co = np.cos(omega)
            so = np.sin(omega)
            co_term = _signed_power(np.array([co], dtype=np.float32), boxiness)[0]
            so_term = _signed_power(np.array([so], dtype=np.float32), boxiness)[0]
            vertices.append(
                [
                    float(half_extents[0] * ce_term * co_term),
                    float(half_extents[1] * ce_term * so_term),
                    float(half_extents[2] * se_term),
                ]
            )
    north_index = len(vertices)
    vertices.append([0.0, 0.0, float(half_extents[2])])

    if ring_starts:
        first_ring = ring_starts[0]
        for lon in range(lon_segments):
            next_lon = (lon + 1) % lon_segments
            indices.append([south_index, first_ring + lon, first_ring + next_lon])

        for ring_index in range(len(ring_starts) - 1):
            current_ring = ring_starts[ring_index]
            next_ring = ring_starts[ring_index + 1]
            for lon in range(lon_segments):
                next_lon = (lon + 1) % lon_segments
                a = current_ring + lon
                b = next_ring + lon
                c = current_ring + next_lon
                d = next_ring + next_lon
                indices.append([a, c, b])
                indices.append([c, d, b])

        last_ring = ring_starts[-1]
        for lon in range(lon_segments):
            next_lon = (lon + 1) % lon_segments
            indices.append([last_ring + lon, north_index, last_ring + next_lon])

    vertices_array = np.asarray(vertices, dtype=np.float32)
    indices_array = _orient_triangles_outward(vertices_array, np.asarray(indices, dtype=np.int32))
    normals = _compute_vertex_normals(vertices_array, indices_array)
    edge_mask = np.sort(np.abs(vertices_array) / np.maximum(half_extents[None, :], 1e-6), axis=1)[:, 1].astype(np.float32)
    colors = np.ones((vertices_array.shape[0], 4), dtype=np.float32)
    return ShapeMeshData(
        vertices=vertices_array,
        normals=normals,
        texcoords=_make_uvs_from_bounds(vertices_array),
        indices=indices_array,
        colors=colors,
        edge_mask=edge_mask,
        collider={"kind": "box", "half_extents": Vec3(half, half, half)},
    )


def build_tetrahedral_cube_mesh(radius: float, cells: int = 4) -> SoftSimulationMesh:
    grid = np.linspace(-1.0, 1.0, cells + 1, dtype=np.float32)
    points: list[list[float]] = []
    point_index: dict[tuple[int, int, int], int] = {}
    for iz, z in enumerate(grid):
        for iy, y in enumerate(grid):
            for ix, x in enumerate(grid):
                point_index[(ix, iy, iz)] = len(points)
                points.append([x * radius, y * radius, z * radius])

    tetrahedra: list[list[int]] = []
    pattern = (
        (0, 1, 3, 4),
        (1, 2, 3, 6),
        (1, 4, 5, 6),
        (3, 4, 6, 7),
        (1, 3, 4, 6),
    )
    for iz in range(cells):
        for iy in range(cells):
            for ix in range(cells):
                corners = [
                    point_index[(ix, iy, iz)],
                    point_index[(ix + 1, iy, iz)],
                    point_index[(ix + 1, iy + 1, iz)],
                    point_index[(ix, iy + 1, iz)],
                    point_index[(ix, iy, iz + 1)],
                    point_index[(ix + 1, iy, iz + 1)],
                    point_index[(ix + 1, iy + 1, iz + 1)],
                    point_index[(ix, iy + 1, iz + 1)],
                ]
                for tet in pattern:
                    tetrahedra.append([corners[tet[0]], corners[tet[1]], corners[tet[2]], corners[tet[3]]])

    points_array = np.asarray(points, dtype=np.float32)
    tetra_array = _orient_tetrahedra_outward(points_array, np.asarray(tetrahedra, dtype=np.int32))
    surface_triangles = _orient_triangles_outward(points_array, extract_surface_triangles(tetra_array))
    return SoftSimulationMesh(
        points=points_array,
        surface_triangles=surface_triangles,
        tetrahedra=tetra_array,
        rest_volume=compute_tet_volume(points_array, tetra_array),
    )


def extract_surface_triangles(tetrahedra: np.ndarray) -> np.ndarray:
    face_counts: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    counts: dict[tuple[int, int, int], int] = {}
    for tet in tetrahedra:
        a, b, c, d = map(int, tet)
        faces = ((a, b, c), (a, d, b), (a, c, d), (b, d, c))
        for face in faces:
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
            face_counts[key] = face
    boundary = [face_counts[key] for key, count in counts.items() if count == 1]
    return np.asarray(boundary, dtype=np.int32)


def _orient_tetrahedra_outward(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    oriented = tetrahedra.copy()
    for index, tet in enumerate(oriented):
        a, b, c, d = points[tet]
        signed = float(np.dot(a - d, np.cross(b - d, c - d)) / 6.0)
        if signed < 0.0:
            oriented[index, 0], oriented[index, 1] = oriented[index, 1], oriented[index, 0]
    return oriented


def compute_surface_volume(vertices: np.ndarray, indices: np.ndarray) -> float:
    return abs(compute_signed_surface_volume(vertices, indices))


def compute_signed_surface_volume(vertices: np.ndarray, indices: np.ndarray) -> float:
    volume = 0.0
    for tri in indices:
        a, b, c = vertices[tri]
        volume += float(np.dot(a, np.cross(b, c)) / 6.0)
    return float(volume)


def compute_tet_volume(points: np.ndarray, tetrahedra: np.ndarray) -> float:
    volume = 0.0
    for tet in tetrahedra:
        a, b, c, d = points[tet]
        volume += abs(float(np.dot(a - d, np.cross(b - d, c - d)) / 6.0))
    return volume


def compute_volume_ratio(rest_volume: float, current_volume: float) -> float:
    if rest_volume <= 1e-8:
        return 0.0
    return current_volume / rest_volume


def compute_best_fit_rigid_alignment(rest_positions: np.ndarray, current_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rest_center = rest_positions.mean(axis=0)
    current_center = current_positions.mean(axis=0)
    rest_zero = rest_positions - rest_center[None, :]
    current_zero = current_positions - current_center[None, :]
    covariance = rest_zero.T @ current_zero
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    aligned_rest = (rest_zero @ rotation.T) + current_center[None, :]
    return aligned_rest.astype(np.float32), rotation.astype(np.float32), current_center.astype(np.float32)


def compute_deformation_statistics(rest_positions: np.ndarray, current_positions: np.ndarray) -> dict[str, np.ndarray | float]:
    raw_displacements = np.linalg.norm(current_positions - rest_positions, axis=1)
    aligned_rest, rotation, current_center = compute_best_fit_rigid_alignment(rest_positions, current_positions)
    aligned_displacements = np.linalg.norm(current_positions - aligned_rest, axis=1)
    rest_center = rest_positions.mean(axis=0)
    return {
        "aligned_rest": aligned_rest,
        "rotation": rotation,
        "current_center": current_center,
        "raw_displacements": raw_displacements.astype(np.float32),
        "aligned_displacements": aligned_displacements.astype(np.float32),
        "raw_mean_displacement": float(raw_displacements.mean()),
        "raw_max_displacement": float(raw_displacements.max()),
        "aligned_rms_deformation": float(np.sqrt(np.mean(aligned_displacements**2))),
        "aligned_max_deformation": float(aligned_displacements.max()),
        "center_of_mass_drift": float(np.linalg.norm(current_center - rest_center)),
    }


def compute_surface_triangle_area_proxies(vertices: np.ndarray, triangles: np.ndarray, reference_normals: np.ndarray) -> np.ndarray:
    proxies = np.zeros((triangles.shape[0],), dtype=np.float32)
    for index, tri in enumerate(triangles):
        a, b, c = vertices[tri]
        cross = np.cross(b - a, c - a)
        area = np.linalg.norm(cross) * 0.5
        direction = 1.0 if np.dot(cross, reference_normals[index]) >= 0.0 else -1.0
        proxies[index] = float(area * direction)
    return proxies


def compute_edge_ratios(vertices: np.ndarray, edges: np.ndarray, rest_lengths: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.ones((0,), dtype=np.float32)
    deltas = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    lengths = np.linalg.norm(deltas, axis=1)
    safe_rest = np.maximum(rest_lengths, 1e-6)
    return (lengths / safe_rest).astype(np.float32)


def create_render_binding(render_vertices: np.ndarray, simulation_points: np.ndarray, k: int = 4) -> RenderBinding:
    deltas = render_vertices[:, None, :] - simulation_points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    node_indices = np.argsort(distances, axis=1)[:, :k]
    nearest = np.take_along_axis(distances, node_indices, axis=1)
    nearest = np.maximum(nearest, 1e-5)
    inv = 1.0 / nearest
    weights = inv / inv.sum(axis=1, keepdims=True)
    return RenderBinding(
        indices=np.arange(render_vertices.shape[0], dtype=np.int32),
        weights=weights.astype(np.float32),
        node_indices=node_indices.astype(np.int32),
    )


def apply_binding(binding: RenderBinding, node_positions: np.ndarray) -> np.ndarray:
    selected = node_positions[binding.node_indices]
    return np.sum(selected * binding.weights[:, :, None], axis=1).astype(np.float32)


def create_render_geom_node(name: str, vertices: np.ndarray, normals: np.ndarray, texcoords: np.ndarray, indices: np.ndarray, colors: np.ndarray | None = None) -> GeomNode:
    if colors is None:
        colors = np.ones((vertices.shape[0], 4), dtype=np.float32)
    vdata = GeomVertexData(name, GeomVertexFormat.getV3n3c4t2(), Geom.UHDynamic)
    vdata.setNumRows(vertices.shape[0])
    vertex_writer = GeomVertexWriter(vdata, "vertex")
    normal_writer = GeomVertexWriter(vdata, "normal")
    color_writer = GeomVertexWriter(vdata, "color")
    texcoord_writer = GeomVertexWriter(vdata, "texcoord")
    for vertex, normal, color, texcoord in zip(vertices, normals, colors, texcoords, strict=True):
        vertex_writer.addData3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
        normal_writer.addData3f(float(normal[0]), float(normal[1]), float(normal[2]))
        color_writer.addData4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        texcoord_writer.addData2f(float(texcoord[0]), float(texcoord[1]))

    triangles = GeomTriangles(Geom.UHDynamic)
    for tri in indices:
        triangles.addVertices(int(tri[0]), int(tri[1]), int(tri[2]))
    triangles.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(triangles)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


def create_point_geom_node(name: str, points: np.ndarray, colors: np.ndarray | None = None) -> GeomNode:
    if colors is None:
        colors = np.ones((points.shape[0], 4), dtype=np.float32)
    vdata = GeomVertexData(name, GeomVertexFormat.getV3c4(), Geom.UHDynamic)
    vdata.setNumRows(points.shape[0])
    vertex_writer = GeomVertexWriter(vdata, "vertex")
    color_writer = GeomVertexWriter(vdata, "color")
    for point, color in zip(points, colors, strict=True):
        vertex_writer.addData3f(float(point[0]), float(point[1]), float(point[2]))
        color_writer.addData4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))

    primitives = GeomPoints(Geom.UHDynamic)
    for index in range(points.shape[0]):
        primitives.addVertex(index)
    primitives.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(primitives)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


class TrueSoftBodyObject:
    def __init__(
        self,
        parent: NodePath,
        world,
        world_info: BulletSoftBodyWorldInfo,
        variant: str,
        radius: float,
        mass: float,
        seed: int,
        name: str = "true-soft-body",
        initial_position: Vec3 | None = None,
        preset_name: str | SoftBodyPreset = "soft",
        render_shape_kind: str = "rounded_cube",
        visualization: str = "shaded",
        max_displacement_visual: float = 1.2,
        simulation_detail_multiplier: float = 1.0,
        render_detail_multiplier: float | None = None,
        surface_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.parent = parent
        self.world = world
        self.world_info = world_info
        self.variant = variant
        self.radius = radius
        self.mass = mass
        self.seed = seed
        self.name = name
        self.visualization = visualization
        self.max_displacement_visual = max_displacement_visual
        self.surface_color = np.asarray(surface_color, dtype=np.float32)
        self._selector_counts_logged: set[str] = set()
        if initial_position is None:
            initial_position = Vec3(0.0, 0.0, 0.0)
        self.initial_position = np.asarray([float(initial_position.x), float(initial_position.y), float(initial_position.z)], dtype=np.float32)
        self.render_shape_kind = render_shape_kind
        if render_detail_multiplier is None:
            render_detail_multiplier = 1.45 if variant == "surface" else 1.15
        self.render_mesh = generate_shape_mesh_data(render_shape_kind, seed=seed, radius=radius, detail_multiplier=render_detail_multiplier)

        if isinstance(preset_name, SoftBodyPreset):
            self.preset = preset_name
        elif variant == "surface":
            self.preset = SOFTBODY_PRESETS[preset_name]
        elif variant == "tetra":
            self.preset = TETRA_PRESETS[preset_name]
        else:
            raise ValueError(f"Unsupported soft body variant: {variant}")

        if variant == "surface":
            self.sim_mesh = build_surface_cube_simulation_mesh(radius=radius, seed=seed, detail_multiplier=simulation_detail_multiplier)
            self.render_binding = create_render_binding(
                self.render_mesh.vertices + self.initial_position[None, :],
                self.sim_mesh.points + self.initial_position[None, :],
                k=4,
            )
            self.soft_node = self._make_surface_soft_body()
        elif variant == "tetra":
            self.sim_mesh = build_tetrahedral_cube_mesh(radius=radius, cells=4)
            self.render_binding = create_render_binding(self.render_mesh.vertices + self.initial_position[None, :], self.sim_mesh.points + self.initial_position[None, :], k=6)
            self.soft_node = self._make_tetra_soft_body()
        else:
            raise ValueError(f"Unsupported soft body variant: {variant}")

        self.body_np = parent.attachNewNode(self.soft_node)
        world.attachSoftBody(self.soft_node)
        self._configure_soft_body()

        self.current_render_vertices = self.get_current_render_vertices()
        self.current_normals = _compute_vertex_normals(self.current_render_vertices, self.render_mesh.indices)
        self.current_colors = np.tile(self.surface_color[None, :], (self.current_render_vertices.shape[0], 1)).astype(np.float32)
        render_geom_node = create_render_geom_node(
            f"{name}-surface",
            self.current_render_vertices,
            self.current_normals,
            _make_uvs_from_bounds(self.current_render_vertices),
            self.render_mesh.indices,
            self.current_colors,
        )
        self.render_node = parent.attachNewNode(render_geom_node)
        self.render_geom = self.render_node.node().modifyGeom(0)
        self.render_node.setTransparency(False)

        self.wireframe_np = self.render_node.copyTo(parent)
        self.wireframe_np.setRenderModeWireframe()
        self.wireframe_np.setRenderModeThickness(1.6)
        self.wireframe_np.setColorScale(Vec4(1.0, 1.0, 1.0, 0.65))
        self.wireframe_np.hide()

        point_geom_node = create_point_geom_node(
            f"{name}-nodes",
            self.get_node_positions(),
            np.tile(np.asarray([[0.85, 0.92, 1.0, 1.0]], dtype=np.float32), (self.sim_mesh.points.shape[0], 1)),
        )
        self.node_points_np = parent.attachNewNode(point_geom_node)
        self.node_points_np.setRenderModeThickness(8.0)
        self.node_points_geom = self.node_points_np.node().modifyGeom(0)
        self.node_points_np.hide()

        self.rest_positions = self.get_node_positions()
        self.local_rest_positions = self.rest_positions - self.initial_position[None, :]
        self.rest_volume = self._compute_current_volume()
        self.surface_edges = self._compute_surface_edges()
        self.rest_edge_lengths = np.linalg.norm(self.rest_positions[self.surface_edges[:, 0]] - self.rest_positions[self.surface_edges[:, 1]], axis=1).astype(np.float32)
        self.rest_triangle_normals = self._compute_rest_triangle_normals()
        self.phase_name = "rest"
        self.stress_debug = False
        self._apply_visualization_mode()

    def _compute_surface_edges(self) -> np.ndarray:
        edges: set[tuple[int, int]] = set()
        for tri in self.sim_mesh.surface_triangles:
            a, b, c = map(int, tri)
            edges.add(tuple(sorted((a, b))))
            edges.add(tuple(sorted((b, c))))
            edges.add(tuple(sorted((c, a))))
        return np.asarray(sorted(edges), dtype=np.int32)

    def _compute_rest_triangle_normals(self) -> np.ndarray:
        normals = np.zeros((self.sim_mesh.surface_triangles.shape[0], 3), dtype=np.float32)
        for index, tri in enumerate(self.sim_mesh.surface_triangles):
            a, b, c = self.rest_positions[tri]
            cross = np.cross(b - a, c - a)
            length = np.linalg.norm(cross)
            if length > 1e-6:
                normals[index] = (cross / length).astype(np.float32)
            else:
                normals[index] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return normals

    def _make_surface_soft_body(self) -> BulletSoftBodyNode:
        points = PTA_LVecBase3f()
        indices = PTA_int()
        for point in self.sim_mesh.points:
            world_point = point + self.initial_position
            points.pushBack(Vec3(float(world_point[0]), float(world_point[1]), float(world_point[2])))
        for tri in self.sim_mesh.surface_triangles:
            for index in tri:
                indices.pushBack(int(index))
        return BulletSoftBodyNode.makeTriMesh(self.world_info, points, indices, True)

    def _make_tetra_soft_body(self) -> BulletSoftBodyNode:
        points = PTA_LVecBase3f()
        indices = PTA_int()
        for point in self.sim_mesh.points:
            world_point = point + self.initial_position
            points.pushBack(Vec3(float(world_point[0]), float(world_point[1]), float(world_point[2])))
        assert self.sim_mesh.tetrahedra is not None
        for tetrahedron in self.sim_mesh.tetrahedra:
            for index in tetrahedron:
                indices.pushBack(int(index))
        return BulletSoftBodyNode.makeTetMesh(self.world_info, points, indices, True)

    def _configure_soft_body(self) -> None:
        preset = self.preset
        material = self.soft_node.getMaterial(0) if self.soft_node.getNumMaterials() > 0 else self.soft_node.appendMaterial()
        material.setLinearStiffness(preset.linear_stiffness)
        material.setAngularStiffness(preset.angular_stiffness)
        material.setVolumePreservation(preset.volume_preservation)

        cfg = self.soft_node.getCfg()
        self.soft_node.setCollisionResponse(True)
        cfg.setDampingCoefficient(preset.damping)
        cfg.setDragCoefficient(0.01 if self.variant == "surface" else 0.02)
        cfg.setDynamicFrictionCoefficient(0.72 if self.variant == "surface" else 0.82)
        cfg.setAnchorsHardness(0.95)
        cfg.setPoseMatchingCoefficient(preset.pose_matching)
        cfg.setRigidContactsHardness(0.68 if self.variant == "surface" else 0.84)
        cfg.setKineticContactsHardness(0.70 if self.variant == "surface" else 0.86)
        cfg.setSoftVsRigidHardness(0.70 if self.variant == "surface" else 0.86)
        cfg.setSoftVsKineticHardness(0.70 if self.variant == "surface" else 0.86)
        cfg.setSoftContactsHardness(0.44 if self.variant == "surface" else 0.68)
        cfg.setPositionsSolverIterations(preset.position_iterations)
        cfg.setVelocitiesSolverIterations(preset.velocity_iterations)
        cfg.setDriftSolverIterations(preset.drift_iterations)
        cfg.setClusterSolverIterations(preset.cluster_iterations)
        cfg.setPressureCoefficient(preset.pressure)
        cfg.setVolumeConservationCoefficient(preset.volume_conservation)
        cfg.setMaxvolume(3.0)
        cfg.setCollisionFlag(BulletSoftBodyConfig.CFSdfRigidSoft, True)
        cfg.setCollisionFlag(BulletSoftBodyConfig.CFClusterRigidSoft, True)
        cfg.setCollisionFlag(BulletSoftBodyConfig.CFClusterSoftSoft, False)
        cfg.setCollisionFlag(BulletSoftBodyConfig.CFClusterSelf, False)
        self.soft_node.setPose(True, True)
        self.soft_node.generateBendingConstraints(1 if self.variant == "surface" else 2, material)
        self.soft_node.generateClusters(preset.cluster_count, 64)
        if self.variant == "surface":
            self.soft_node.setTotalMass(self.mass, True)
        else:
            self.soft_node.setVolumeMass(self.mass)

    def destroy(self) -> None:
        self.world.removeSoftBody(self.soft_node)
        self.render_node.removeNode()
        self.wireframe_np.removeNode()
        self.node_points_np.removeNode()
        self.body_np.removeNode()

    def set_phase_name(self, phase_name: str) -> None:
        self.phase_name = phase_name

    def set_visualization_mode(self, visualization: str) -> None:
        self.visualization = visualization
        self._apply_visualization_mode()

    def _apply_visualization_mode(self) -> None:
        show_surface = self.visualization in {"shaded", "displacement", "wireframe"}
        self.render_node.show() if show_surface else self.render_node.hide()
        self.wireframe_np.show() if self.visualization == "wireframe" else self.wireframe_np.hide()
        self.node_points_np.show() if self.visualization == "nodes" else self.node_points_np.hide()

    def _log_selector(self, label: str, indices: np.ndarray) -> None:
        if label and label not in self._selector_counts_logged:
            print(f"[softbody-selector] {self.name}:{label} -> {int(indices.size)} nodes")
            self._selector_counts_logged.add(label)

    def select_nodes_near_point(self, rest_point: np.ndarray, radius: float, label: str = "") -> np.ndarray:
        distances = np.linalg.norm(self.rest_positions - rest_point[None, :], axis=1)
        indices = np.where(distances <= radius)[0].astype(np.int32)
        self._log_selector(label, indices)
        return indices

    def select_nodes_on_face(self, axis: int, sign: int, thickness: float, label: str = "") -> np.ndarray:
        coords = self.local_rest_positions[:, axis]
        target = sign * self.radius
        indices = np.where(np.abs(coords - target) <= thickness)[0].astype(np.int32)
        self._log_selector(label, indices)
        if indices.size == 0:
            distances = np.abs(coords - target)
            fallback = np.argsort(distances)[:16].astype(np.int32)
            self._log_selector(f"{label}-fallback", fallback)
            return fallback
        return indices

    def select_corner_nodes(self, sign_x: int, sign_y: int, sign_z: int, radius: float, label: str = "") -> np.ndarray:
        target = np.asarray([sign_x, sign_y, sign_z], dtype=np.float32) * self.radius + self.initial_position
        indices = self.select_nodes_near_point(target, radius, label=label)
        if indices.size > 0:
            return indices
        distances = np.linalg.norm(self.rest_positions - target[None, :], axis=1)
        fallback = np.argsort(distances)[:8].astype(np.int32)
        self._log_selector(f"{label}-fallback", fallback)
        return fallback

    def select_top_nodes(self, threshold: float = 0.55, label: str = "top") -> np.ndarray:
        limit = self.radius * threshold
        indices = np.where(self.local_rest_positions[:, 2] >= limit)[0].astype(np.int32)
        self._log_selector(label, indices)
        return indices

    def select_bottom_nodes(self, threshold: float = 0.55, label: str = "bottom") -> np.ndarray:
        limit = -self.radius * threshold
        indices = np.where(self.local_rest_positions[:, 2] <= limit)[0].astype(np.int32)
        self._log_selector(label, indices)
        return indices

    def get_node_positions(self) -> np.ndarray:
        return np.asarray([[node.getPos().x, node.getPos().y, node.getPos().z] for node in self.soft_node.getNodes()], dtype=np.float32)

    def get_node_velocities(self) -> np.ndarray:
        return np.asarray([[node.getVelocity().x, node.getVelocity().y, node.getVelocity().z] for node in self.soft_node.getNodes()], dtype=np.float32)

    def get_current_render_vertices(self) -> np.ndarray:
        positions = self.get_node_positions()
        return apply_binding(self.render_binding, positions)

    def _displacement_colors(self, displacement: np.ndarray) -> np.ndarray:
        t = np.clip(displacement / max(self.max_displacement_visual, 1e-5), 0.0, 1.0)
        colors = np.zeros((displacement.shape[0], 4), dtype=np.float32)
        colors[:, 0] = np.clip((t - 0.55) * 2.2, 0.0, 1.0) + np.clip((t - 0.82) * 5.0, 0.0, 1.0)
        colors[:, 1] = np.clip(1.0 - np.abs(t - 0.45) * 2.4, 0.0, 1.0)
        colors[:, 2] = np.clip(1.15 - t * 1.2, 0.0, 1.0)
        colors[:, 3] = 1.0
        return colors

    def _rigid_aligned_rest(self, current_positions: np.ndarray, rest_positions: np.ndarray) -> np.ndarray:
        aligned_rest, _, _ = compute_best_fit_rigid_alignment(rest_positions, current_positions)
        return aligned_rest

    def _write_geom(self, geom, vertices: np.ndarray, normals: np.ndarray, colors: np.ndarray) -> None:
        vdata = geom.modifyVertexData()
        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        color_writer = GeomVertexWriter(vdata, "color")
        vertex_writer.setRow(0)
        normal_writer.setRow(0)
        color_writer.setRow(0)
        for vertex, normal, color in zip(vertices, normals, colors, strict=True):
            vertex_writer.setData3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
            normal_writer.setData3f(float(normal[0]), float(normal[1]), float(normal[2]))
            color_writer.setData4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))

    def sync_render_mesh(self) -> None:
        self.current_render_vertices = self.get_current_render_vertices()
        self.current_normals = _compute_vertex_normals(self.current_render_vertices, self.render_mesh.indices)
        if self.visualization == "displacement":
            render_stats = compute_deformation_statistics(self.render_mesh.vertices + self.initial_position[None, :], self.current_render_vertices)
            render_displacements = render_stats["aligned_displacements"]
            self.current_colors = self._displacement_colors(render_displacements)
        else:
            self.current_colors = np.tile(self.surface_color[None, :], (self.current_render_vertices.shape[0], 1)).astype(np.float32)
        self._write_geom(self.render_geom, self.current_render_vertices, self.current_normals, self.current_colors)

        node_positions = self.get_node_positions()
        node_stats = compute_deformation_statistics(self.rest_positions, node_positions)
        node_displacements = node_stats["aligned_displacements"]
        point_colors = self._displacement_colors(node_displacements)
        self._write_points(node_positions, point_colors)

    def _write_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        vdata = self.node_points_geom.modifyVertexData()
        vertex_writer = GeomVertexWriter(vdata, "vertex")
        color_writer = GeomVertexWriter(vdata, "color")
        vertex_writer.setRow(0)
        color_writer.setRow(0)
        for position, color in zip(positions, colors, strict=True):
            vertex_writer.setData3f(float(position[0]), float(position[1]), float(position[2]))
            color_writer.setData4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))

    def apply_force_to_indices(self, indices: np.ndarray, force: np.ndarray) -> int:
        for index in indices:
            self.soft_node.addForce(Vec3(float(force[0]), float(force[1]), float(force[2])), int(index))
        return int(indices.size)

    def apply_global_force(self, force: np.ndarray) -> None:
        self.soft_node.addForce(Vec3(float(force[0]), float(force[1]), float(force[2])))

    def apply_forces(self, indices: np.ndarray, forces: np.ndarray) -> int:
        for index, force in zip(indices, forces, strict=True):
            self.soft_node.addForce(Vec3(float(force[0]), float(force[1]), float(force[2])), int(index))
        return int(indices.size)

    def apply_velocity_to_indices(self, indices: np.ndarray, velocity: np.ndarray) -> int:
        for index in indices:
            self.soft_node.addVelocity(Vec3(float(velocity[0]), float(velocity[1]), float(velocity[2])), int(index))
        return int(indices.size)

    def apply_velocities(self, indices: np.ndarray, velocities: np.ndarray) -> int:
        for index, velocity in zip(indices, velocities, strict=True):
            self.soft_node.addVelocity(Vec3(float(velocity[0]), float(velocity[1]), float(velocity[2])), int(index))
        return int(indices.size)

    def apply_force_to_region(self, center: np.ndarray, radius: float, force: np.ndarray, rest_space: bool = False, label: str = "") -> int:
        positions = self.rest_positions if rest_space else self.get_node_positions()
        distances = np.linalg.norm(positions - center[None, :], axis=1)
        matches = np.where(distances <= radius)[0].astype(np.int32)
        self._log_selector(label, matches)
        return self.apply_force_to_indices(matches, force)

    def select_nodes_near_rest_point(self, rest_point: np.ndarray, radius: float, label: str = "") -> np.ndarray:
        distances = np.linalg.norm(self.local_rest_positions - rest_point[None, :], axis=1)
        matches = np.where(distances <= radius)[0].astype(np.int32)
        self._log_selector(label, matches)
        return matches

    def apply_region_force(self, selector: np.ndarray, force: np.ndarray) -> int:
        return self.apply_force_to_indices(selector.astype(np.int32), force)

    def apply_force_to_face(self, direction: np.ndarray, strength: float) -> int:
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        dots = self.local_rest_positions @ direction
        threshold = float(dots.max() - self.radius * 0.35)
        matches = np.where(dots >= threshold)[0].astype(np.int32)
        return self.apply_force_to_indices(matches, direction * strength)

    def apply_corner_impulse(self, corner_id: int, impulse: np.ndarray) -> int:
        corners = np.asarray(
            [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ],
            dtype=np.float32,
        ) * self.radius
        target = corners[int(corner_id) % len(corners)] + self.initial_position
        distances = np.linalg.norm(self.rest_positions - target[None, :], axis=1)
        index = int(np.argmin(distances))
        self.soft_node.addVelocity(Vec3(float(impulse[0]), float(impulse[1]), float(impulse[2])), index)
        return index

    def apply_twist(self, axis: np.ndarray, strength: float) -> tuple[int, int]:
        axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
        upper = np.where(self.local_rest_positions @ axis >= np.quantile(self.local_rest_positions @ axis, 0.70))[0].astype(np.int32)
        lower = np.where(self.local_rest_positions @ axis <= np.quantile(self.local_rest_positions @ axis, 0.30))[0].astype(np.int32)
        tangent = np.cross(axis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if np.linalg.norm(tangent) < 1e-5:
            tangent = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        self.apply_force_to_indices(upper, tangent * strength)
        self.apply_force_to_indices(lower, -tangent * strength)
        return int(upper.size), int(lower.size)

    def apply_opposing_twist(self, axis: np.ndarray, strength: float) -> tuple[int, int]:
        return self.apply_twist(axis, strength)

    def apply_spring_anchor(self, indices: np.ndarray, target_positions: np.ndarray, stiffness: float, damping: float) -> int:
        current_positions = self.get_node_positions()[indices]
        current_velocities = self.get_node_velocities()[indices]
        forces = (target_positions - current_positions) * stiffness - current_velocities * damping
        for index, force in zip(indices, forces, strict=True):
            self.soft_node.addForce(Vec3(float(force[0]), float(force[1]), float(force[2])), int(index))
        return int(indices.size)

    def apply_planar_anchor(self, indices: np.ndarray, target_xy: np.ndarray, stiffness: float, damping: float) -> int:
        current_positions = self.get_node_positions()[indices]
        current_velocities = self.get_node_velocities()[indices]
        planar_error = target_xy - current_positions[:, :2]
        planar_force = planar_error * stiffness - current_velocities[:, :2] * damping
        for index, force in zip(indices, planar_force, strict=True):
            self.soft_node.addForce(Vec3(float(force[0]), float(force[1]), 0.0), int(index))
        return int(indices.size)

    def apply_uniform_pressure_modulation(self, delta: float) -> None:
        cfg = self.soft_node.getCfg()
        cfg.setPressureCoefficient(max(0.0, cfg.getPressureCoefficient() + delta))

    def apply_radial_pressure_like_force(self, strength: float, center: np.ndarray | None = None) -> int:
        positions = self.get_node_positions()
        if center is None:
            center = positions.mean(axis=0)
        directions = positions - center[None, :]
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        safe = np.maximum(lengths, 1e-5)
        normalized = directions / safe
        falloff = np.clip(lengths / max(self.radius, 1e-5), 0.35, 1.0)
        forces = normalized * falloff * float(strength)
        indices = np.arange(positions.shape[0], dtype=np.int32)
        return self.apply_forces(indices, forces.astype(np.float32))

    def apply_bass_breath(self, strength: float) -> int:
        center = self.rest_positions.mean(axis=0)
        return self.apply_radial_pressure_like_force(strength, center=center)

    def set_soft_soft_collision(self, enabled: bool) -> None:
        cfg = self.soft_node.getCfg()
        cfg.setCollisionFlag(BulletSoftBodyConfig.CFClusterSoftSoft, bool(enabled))

    def _compute_current_volume(self) -> float:
        positions = self.get_node_positions()
        if self.sim_mesh.tetrahedra is not None:
            return compute_tet_volume(positions, self.sim_mesh.tetrahedra)
        return compute_surface_volume(positions, self.sim_mesh.surface_triangles)

    def _compute_twist_angle_proxy(self) -> float:
        top_indices = self.select_top_nodes(0.45, label="metrics-top")
        bottom_indices = self.select_bottom_nodes(0.45, label="metrics-bottom")
        left_indices = self.select_nodes_on_face(0, -1, 0.28, label="metrics-left")
        right_indices = self.select_nodes_on_face(0, 1, 0.28, label="metrics-right")
        positions = self.get_node_positions()
        top_center = positions[top_indices].mean(axis=0)
        bottom_center = positions[bottom_indices].mean(axis=0)
        right_dir = positions[right_indices].mean(axis=0) - positions[left_indices].mean(axis=0)
        top_dir = top_center - bottom_center
        right_xy = right_dir[:2]
        top_xy = top_dir[:2]
        if np.linalg.norm(right_xy) < 1e-6 or np.linalg.norm(top_xy) < 1e-6:
            return 0.0
        angle_right = np.arctan2(right_xy[1], right_xy[0])
        angle_top = np.arctan2(top_xy[1], top_xy[0])
        return float(np.degrees(angle_right - angle_top))

    def get_metrics(self) -> SoftBodyMetrics:
        positions = self.get_node_positions()
        velocities = self.get_node_velocities()
        current_volume = self._compute_current_volume()
        stats = compute_deformation_statistics(self.rest_positions, positions)
        aligned_rest = stats["aligned_rest"]
        aligned_displacements = stats["aligned_displacements"]
        raw_displacements = stats["raw_displacements"]
        speeds = np.linalg.norm(velocities, axis=1)
        kinetic = 0.5 * np.sum(speeds**2)
        min_corner = self.select_corner_nodes(1, 1, 1, 0.55, label="metrics-struck")
        opposite_corner = self.select_corner_nodes(-1, -1, -1, 0.55, label="metrics-opposite")
        top_indices = self.select_top_nodes(0.45, label="metrics-top")
        bottom_indices = self.select_bottom_nodes(0.45, label="metrics-bottom")
        bbox = positions.max(axis=0) - positions.min(axis=0)
        top_disp = float(np.linalg.norm(positions[top_indices] - aligned_rest[top_indices], axis=1).mean()) if top_indices.size else 0.0
        bottom_disp = float(np.linalg.norm(positions[bottom_indices] - aligned_rest[bottom_indices], axis=1).mean()) if bottom_indices.size else 0.0
        struck_disp = float(np.linalg.norm(positions[min_corner] - aligned_rest[min_corner], axis=1).mean()) if min_corner.size else 0.0
        opposite_disp = float(np.linalg.norm(positions[opposite_corner] - aligned_rest[opposite_corner], axis=1).mean()) if opposite_corner.size else 0.0
        area_proxies = compute_surface_triangle_area_proxies(positions, self.sim_mesh.surface_triangles, self.rest_triangle_normals)
        edge_ratios = compute_edge_ratios(positions, self.surface_edges, self.rest_edge_lengths)
        min_height = float(positions[:, 2].min())
        nodes_below_floor = int(np.count_nonzero(positions[:, 2] < -0.02))
        has_non_finite = bool((not np.all(np.isfinite(positions))) or (not np.all(np.isfinite(velocities))) or (not np.isfinite(current_volume)))
        return SoftBodyMetrics(
            rest_volume=self.rest_volume,
            current_volume=current_volume,
            volume_ratio=compute_volume_ratio(self.rest_volume, current_volume),
            center_of_mass=positions.mean(axis=0),
            raw_max_displacement=float(stats["raw_max_displacement"]),
            raw_mean_displacement=float(stats["raw_mean_displacement"]),
            aligned_rms_deformation=float(stats["aligned_rms_deformation"]),
            aligned_max_deformation=float(stats["aligned_max_deformation"]),
            max_velocity=float(speeds.max()),
            mean_velocity=float(speeds.mean()),
            kinetic_energy_proxy=float(kinetic),
            bounding_box_dimensions=bbox.astype(np.float32),
            top_region_deformation=top_disp,
            bottom_region_deformation=bottom_disp,
            struck_corner_deformation=struck_disp,
            opposite_corner_deformation=opposite_disp,
            twist_angle_proxy=self._compute_twist_angle_proxy(),
            center_of_mass_drift=float(stats["center_of_mass_drift"]),
            minimum_node_height=min_height,
            nodes_below_floor=nodes_below_floor,
            min_triangle_area_proxy=float(area_proxies.min()) if area_proxies.size else 0.0,
            max_edge_stretch_ratio=float(edge_ratios.max()) if edge_ratios.size else 1.0,
            min_edge_compression_ratio=float(edge_ratios.min()) if edge_ratios.size else 1.0,
            has_non_finite=has_non_finite,
            node_count=int(positions.shape[0]),
        )


def create_debug_plate(parent: NodePath, half_extents: Vec3 | None = None, position: Vec3 | None = None, kinematic: bool = True) -> tuple[NodePath, BulletRigidBodyNode]:
    from panda3d.bullet import BulletBoxShape

    if half_extents is None:
        half_extents = Vec3(1.7, 1.7, 0.18)
    if position is None:
        position = Vec3(0.0, 0.0, 4.0)
    plate_node = BulletRigidBodyNode("soft-debug-plate")
    plate_node.addShape(BulletBoxShape(half_extents))
    if kinematic:
        plate_node.setKinematic(True)
    plate_np = parent.attachNewNode(plate_node)
    plate_np.setPos(position)
    return plate_np, plate_node
