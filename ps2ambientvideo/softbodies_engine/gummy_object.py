from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletConvexHullShape, BulletMultiSphereShape, BulletRigidBodyNode
from panda3d.core import Filename, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexRewriter, GeomVertexWriter
from panda3d.core import NodePath, PTA_LVecBase3f, PTA_float, Point3, Quat, Shader, TransparencyAttrib, Vec3, Vec4

from .audio_analysis import AudioSample


SHADER_DIR = Path(__file__).resolve().parent / "shaders"


@dataclass(frozen=True)
class GelMaterialConfig:
    base_color: tuple[float, float, float, float]
    opacity: float
    transmission_strength: float
    cloudiness: float
    fresnel_strength: float
    specular_strength: float
    emissive_strength: float
    absorption_strength: float
    rear_face_strength: float
    thickness_absorption: float
    minimum_body_light: float
    exposure: float
    inner_layer_opacity: float
    inner_scale: float = 1.0


@dataclass
class GelatinControls:
    gelatin_opacity: float = 0.90
    gelatin_cloudiness: float = 0.92
    transmission_strength: float = 0.30
    fresnel_strength: float = 0.08
    specular_strength: float = 0.36
    wobble_strength: float = 1.0
    deformation_strength: float = 1.0
    backlight_strength: float = 1.60
    rear_face_strength: float = 0.01
    thickness_absorption: float = 0.96
    minimum_body_light: float = 0.50
    exposure: float = 1.30
    inner_layer_opacity: float = 0.0


@dataclass
class GummyConfig:
    name: str
    radius: float
    color: tuple[float, float, float, float]
    position: tuple[float, float, float]
    mass: float
    beat_cooldown: float
    onset_cooldown: float
    beat_gain: float
    onset_gain: float
    bass_sensitivity: float
    wobble_gain: float
    audio_delay: float
    shape_kind: str
    shape_seed: int


@dataclass
class ShapeMeshData:
    vertices: np.ndarray
    normals: np.ndarray
    texcoords: np.ndarray
    indices: np.ndarray
    colors: np.ndarray
    edge_mask: np.ndarray
    collider: dict


@dataclass(frozen=True)
class MeshValidationReport:
    shape_name: str
    vertex_count: int
    triangle_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    signed_volume: float


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-6)
    return values / lengths


def _subdivide_triangle(triangle: tuple[np.ndarray, np.ndarray, np.ndarray], depth: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tris = [triangle]
    for _ in range(depth):
        refined: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for a, b, c in tris:
            ab = (a + b) * 0.5
            bc = (b + c) * 0.5
            ca = (c + a) * 0.5
            refined.extend(
                [
                    (a, ab, ca),
                    (ab, b, bc),
                    (ca, bc, c),
                    (ab, bc, ca),
                ]
            )
        tris = refined
    return tris


def _triangulate_face(face: list[int], vertices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    face_vertices = vertices[np.asarray(face, dtype=np.int32)]
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(1, len(face) - 1):
        triangles.append((face_vertices[0], face_vertices[index], face_vertices[index + 1]))
    return triangles


def _compute_face_normal(face_vertices: np.ndarray) -> np.ndarray:
    normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
    return _normalize_rows(normal[None, :])[0]


def _rounded_key(vertex: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(round(float(component) * 1000000.0)) for component in vertex)


def _rounded_uv_key(texcoord: np.ndarray) -> tuple[int, int]:
    return tuple(int(round(float(component) * 1000000.0)) for component in texcoord)


def _compute_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    tris = vertices[indices]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for tri_index, face in enumerate(indices):
        normals[face] += face_normals[tri_index]
    return _normalize_rows(normals).astype(np.float32)


def _orient_triangles_outward(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    oriented = np.asarray(indices, dtype=np.int32).copy()
    for tri_index, tri in enumerate(oriented):
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        triangle_center = (a + b + c) / 3.0
        if float(np.dot(normal, triangle_center - center)) < 0.0:
            oriented[tri_index, 1], oriented[tri_index, 2] = oriented[tri_index, 2], oriented[tri_index, 1]
    return oriented


def _make_uvs_from_bounds(vertices: np.ndarray) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return np.column_stack(
        [
            (vertices[:, 0] - mins[0]) / max(maxs[0] - mins[0], 1e-6),
            (vertices[:, 2] - mins[2]) / max(maxs[2] - mins[2], 1e-6),
        ]
    ).astype(np.float32)


def _make_shape_colors(normals: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            0.5 + 0.5 * normals[:, 0],
            0.5 + 0.5 * normals[:, 1],
            0.5 + 0.5 * normals[:, 2],
            np.asarray(edge_mask, dtype=np.float32),
        ]
    ).astype(np.float32)


def validate_shape_mesh_data(shape_name: str, mesh_data: ShapeMeshData) -> MeshValidationReport:
    vertices = np.asarray(mesh_data.vertices, dtype=np.float64)
    indices = np.asarray(mesh_data.indices, dtype=np.int32)
    if not np.isfinite(vertices).all():
        raise ValueError(f"{shape_name}: mesh contains non-finite vertex positions.")
    if not np.isfinite(mesh_data.normals).all():
        raise ValueError(f"{shape_name}: mesh contains non-finite normals.")

    edge_counts: dict[tuple[int, int], int] = {}
    signed_volume = 0.0
    for tri in indices:
        a_idx, b_idx, c_idx = map(int, tri)
        a, b, c = vertices[tri]
        area = np.linalg.norm(np.cross(b - a, c - a)) * 0.5
        if area <= 1e-10:
            raise ValueError(f"{shape_name}: zero-area triangle detected.")
        signed_volume += float(np.dot(a, np.cross(b, c)) / 6.0)
        for u, v in ((a_idx, b_idx), (b_idx, c_idx), (c_idx, a_idx)):
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary_edge_count = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold_edge_count = sum(1 for count in edge_counts.values() if count > 2)
    if boundary_edge_count != 0:
        raise ValueError(f"{shape_name}: open mesh detected; boundary edge count = {boundary_edge_count}.")
    if non_manifold_edge_count != 0:
        raise ValueError(f"{shape_name}: non-manifold mesh detected; non-manifold edge count = {non_manifold_edge_count}.")
    if abs(signed_volume) <= 1e-10:
        raise ValueError(f"{shape_name}: signed volume is zero.")

    return MeshValidationReport(
        shape_name=shape_name,
        vertex_count=int(vertices.shape[0]),
        triangle_count=int(indices.shape[0]),
        boundary_edge_count=boundary_edge_count,
        non_manifold_edge_count=non_manifold_edge_count,
        signed_volume=signed_volume,
    )


def format_mesh_validation_report(report: MeshValidationReport) -> str:
    return (
        f"{report.shape_name}: vertex count={report.vertex_count} triangle count={report.triangle_count} "
        f"boundary edge count={report.boundary_edge_count} non-manifold edge count={report.non_manifold_edge_count} "
        f"signed volume={report.signed_volume:.6f}"
    )


def _build_rounded_polyhedron_mesh(
    base_vertices: np.ndarray,
    faces: list[list[int]],
    round_radius: float,
    subdivision_depth: int,
    collider: dict,
) -> ShapeMeshData:
    unique_positions: list[np.ndarray] = []
    unique_faces: list[set[int]] = []
    triangle_indices: list[list[int]] = []
    weld_map: dict[tuple[int, int, int], int] = {}
    solid_center = base_vertices.mean(axis=0)
    oriented_faces: list[list[int]] = []

    for face in faces:
        face_vertices = base_vertices[np.asarray(face, dtype=np.int32)]
        face_normal = _compute_face_normal(face_vertices)
        face_center = face_vertices.mean(axis=0)
        if float(np.dot(face_normal, face_center - solid_center)) < 0.0:
            oriented_faces.append(list(reversed(face)))
        else:
            oriented_faces.append(face)

    for face_index, face in enumerate(oriented_faces):
        face_vertices = base_vertices[np.asarray(face, dtype=np.int32)]
        for triangle in _triangulate_face(face, base_vertices):
            for sub_triangle in _subdivide_triangle(triangle, subdivision_depth):
                tri_ids: list[int] = []
                for vertex in sub_triangle:
                    key = _rounded_key(vertex)
                    vertex_id = weld_map.get(key)
                    if vertex_id is None:
                        vertex_id = len(unique_positions)
                        weld_map[key] = vertex_id
                        unique_positions.append(np.asarray(vertex, dtype=np.float32))
                        unique_faces.append(set())
                    unique_faces[vertex_id].add(face_index)
                    tri_ids.append(vertex_id)
                triangle_indices.append(tri_ids)

    positions = np.asarray(unique_positions, dtype=np.float32)
    face_normals = [_compute_face_normal(base_vertices[np.asarray(face, dtype=np.int32)]) for face in oriented_faces]
    averaged_normals = []
    edge_mask = []
    for contributing_faces in unique_faces:
        normals = np.asarray([face_normals[face_id] for face_id in sorted(contributing_faces)], dtype=np.float32)
        averaged_normals.append(_normalize_rows(normals.mean(axis=0, keepdims=True))[0])
        edge_mask.append(float(np.clip((len(contributing_faces) - 1) / 2.0, 0.0, 1.0)))
    averaged_normals_array = np.asarray(averaged_normals, dtype=np.float32)
    rounded_positions = positions + averaged_normals_array * float(round_radius)
    indices = _orient_triangles_outward(rounded_positions.astype(np.float32), np.asarray(triangle_indices, dtype=np.int32))
    normals = _compute_vertex_normals(rounded_positions, indices)
    texcoords = _make_uvs_from_bounds(rounded_positions)
    colors = _make_shape_colors(normals, np.asarray(edge_mask, dtype=np.float32))

    return ShapeMeshData(
        vertices=rounded_positions.astype(np.float32),
        normals=normals.astype(np.float32),
        texcoords=texcoords,
        indices=indices,
        colors=colors,
        edge_mask=np.asarray(edge_mask, dtype=np.float32),
        collider=collider,
    )


def _signed_power(value: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(value) * np.power(np.abs(value), exponent)


def _build_rounded_box_mesh(half_extents: np.ndarray, round_radius: float, segments: int, collider: dict) -> ShapeMeshData:
    lat_segments = max(12, segments)
    lon_segments = max(24, segments * 2)
    boxiness = 0.42 + 0.10 * np.clip(round_radius / max(float(np.min(half_extents)), 1e-6), 0.0, 1.0)
    omegas = np.linspace(-np.pi, np.pi, lon_segments, endpoint=False, dtype=np.float32)

    vertices: list[list[float]] = []
    indices: list[list[int]] = []
    south_index = 0
    vertices.append([0.0, 0.0, float(-half_extents[2])])
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
    normalized = np.abs(vertices_array) / np.maximum(half_extents[None, :], 1e-6)
    edge_mask = np.sort(normalized, axis=1)[:, 1].astype(np.float32)
    return ShapeMeshData(
        vertices=vertices_array,
        normals=normals,
        texcoords=_make_uvs_from_bounds(vertices_array),
        indices=indices_array,
        colors=_make_shape_colors(normals, edge_mask),
        edge_mask=edge_mask,
        collider=collider,
    )


def make_rounded_cube_mesh_data(radius: float, seed: int, detail_multiplier: float = 1.0) -> ShapeMeshData:
    del seed
    half = radius * 0.80
    return _build_rounded_box_mesh(
        half_extents=np.asarray([half, half, half], dtype=np.float32),
        round_radius=radius * 0.16,
        segments=max(14, int(round(14 * detail_multiplier))),
        collider={"kind": "box", "half_extents": Vec3(half, half, half)},
    )


def make_rectangular_cuboid_mesh_data(radius: float, seed: int, detail_multiplier: float = 1.0) -> ShapeMeshData:
    del seed
    hx, hy, hz = radius * 0.96, radius * 0.66, radius * 0.60
    return _build_rounded_box_mesh(
        half_extents=np.asarray([hx, hy, hz], dtype=np.float32),
        round_radius=radius * 0.15,
        segments=max(14, int(round(14 * detail_multiplier))),
        collider={"kind": "box", "half_extents": Vec3(hx, hy, hz)},
    )


def make_triangular_prism_mesh_data(radius: float, seed: int, detail_multiplier: float = 1.0) -> ShapeMeshData:
    del seed
    prism_height = radius * 0.66
    tri_radius = radius * 0.84
    base_vertices = np.asarray(
        [
            [tri_radius, 0.0, -prism_height],
            [-tri_radius * 0.45, tri_radius * 0.78, -prism_height],
            [-tri_radius * 0.45, -tri_radius * 0.78, -prism_height],
            [tri_radius, 0.0, prism_height],
            [-tri_radius * 0.45, tri_radius * 0.78, prism_height],
            [-tri_radius * 0.45, -tri_radius * 0.78, prism_height],
        ],
        dtype=np.float32,
    )
    faces = [
        [0, 1, 2],
        [3, 5, 4],
        [0, 3, 4, 1],
        [1, 4, 5, 2],
        [2, 5, 3, 0],
    ]
    return _build_rounded_polyhedron_mesh(
        base_vertices=base_vertices,
        faces=faces,
        round_radius=radius * 0.18,
        subdivision_depth=max(5, min(7, int(round(5 * detail_multiplier)))),
        collider={"kind": "convex_hull", "points": base_vertices.copy()},
    )


def make_rounded_tetrahedron_mesh_data(radius: float, seed: int, detail_multiplier: float = 1.0) -> ShapeMeshData:
    del seed
    raw = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    base_vertices = _normalize_rows(raw) * (radius * 0.92)
    faces = [
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ]
    return _build_rounded_polyhedron_mesh(
        base_vertices=base_vertices,
        faces=faces,
        round_radius=radius * 0.20,
        subdivision_depth=max(5, min(7, int(round(5 * detail_multiplier)))),
        collider={"kind": "convex_hull", "points": base_vertices.copy()},
    )


def make_rounded_octahedron_mesh_data(radius: float, seed: int, detail_multiplier: float = 1.0) -> ShapeMeshData:
    del seed
    base_vertices = np.asarray(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float32,
    ) * 0.82
    faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ]
    return _build_rounded_polyhedron_mesh(
        base_vertices=base_vertices,
        faces=faces,
        round_radius=radius * 0.20,
        subdivision_depth=max(5, min(7, int(round(5 * detail_multiplier)))),
        collider={"kind": "convex_hull", "points": base_vertices.copy()},
    )


def make_hexagonal_prism_mesh_data(radius: float, seed: int, detail_multiplier: float = 1.0) -> ShapeMeshData:
    del seed
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False, dtype=np.float32)
    ring_radius = radius * 0.76
    height = radius * 0.56
    bottom = np.column_stack([np.cos(angles) * ring_radius, np.sin(angles) * ring_radius, np.full(6, -height, dtype=np.float32)])
    top = np.column_stack([np.cos(angles) * ring_radius, np.sin(angles) * ring_radius, np.full(6, height, dtype=np.float32)])
    base_vertices = np.vstack([bottom, top]).astype(np.float32)
    faces: list[list[int]] = [list(range(6)), list(range(11, 5, -1))]
    for index in range(6):
        next_index = (index + 1) % 6
        faces.append([index, next_index, next_index + 6, index + 6])
    return _build_rounded_polyhedron_mesh(
        base_vertices=base_vertices,
        faces=faces,
        round_radius=radius * 0.17,
        subdivision_depth=max(5, min(7, int(round(5 * detail_multiplier)))),
        collider={"kind": "convex_hull", "points": base_vertices.copy()},
    )


def make_torus_ring_mesh_data(radius: float, seed: int, detail_multiplier: float = 1.0) -> ShapeMeshData:
    del seed
    major_radius = radius * 0.72
    minor_radius = radius * 0.30
    ring_segments = max(56, int(round(56 * detail_multiplier)))
    tube_segments = max(28, int(round(28 * detail_multiplier)))
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    texcoords: list[list[float]] = []
    indices: list[list[int]] = []

    for ring in range(ring_segments):
        u = ring / ring_segments
        theta = 2.0 * np.pi * u
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        center = np.array([major_radius * cos_theta, major_radius * sin_theta, 0.0], dtype=np.float32)
        for tube in range(tube_segments):
            v = tube / tube_segments
            phi = 2.0 * np.pi * v
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            normal = np.array([cos_theta * cos_phi, sin_theta * cos_phi, sin_phi], dtype=np.float32)
            position = center + normal * minor_radius
            vertices.append(position.tolist())
            normals.append(normal.tolist())
            texcoords.append([u, v])

    for ring in range(ring_segments):
        next_ring = (ring + 1) % ring_segments
        for tube in range(tube_segments):
            next_tube = (tube + 1) % tube_segments
            a = ring * tube_segments + tube
            b = next_ring * tube_segments + tube
            c = ring * tube_segments + next_tube
            d = next_ring * tube_segments + next_tube
            indices.append([a, b, c])
            indices.append([c, b, d])

    edge_mask = np.ones(len(vertices), dtype=np.float32)
    colors = np.column_stack(
        [
            0.5 + 0.5 * np.asarray(normals, dtype=np.float32)[:, 0],
            0.5 + 0.5 * np.asarray(normals, dtype=np.float32)[:, 1],
            0.5 + 0.5 * np.asarray(normals, dtype=np.float32)[:, 2],
            edge_mask,
        ]
    ).astype(np.float32)

    sphere_points = PTA_LVecBase3f()
    sphere_radii = PTA_float()
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        sphere_points.pushBack(Point3(np.cos(angle) * major_radius, np.sin(angle) * major_radius, 0.0))
        sphere_radii.pushBack(minor_radius * 0.92)

    return ShapeMeshData(
        vertices=np.asarray(vertices, dtype=np.float32),
        normals=np.asarray(normals, dtype=np.float32),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.int32),
        colors=colors,
        edge_mask=edge_mask,
        collider={"kind": "multi_sphere", "points": sphere_points, "radii": sphere_radii},
    )


SHAPE_GENERATORS = {
    "rounded_cube": make_rounded_cube_mesh_data,
    "rectangular_cuboid": make_rectangular_cuboid_mesh_data,
    "triangular_prism": make_triangular_prism_mesh_data,
    "rounded_tetrahedron": make_rounded_tetrahedron_mesh_data,
    "rounded_octahedron": make_rounded_octahedron_mesh_data,
    "hexagonal_prism": make_hexagonal_prism_mesh_data,
    "torus_ring": make_torus_ring_mesh_data,
}


def generate_shape_mesh_data(kind: str, seed: int, radius: float = 1.0, detail_multiplier: float = 1.0) -> ShapeMeshData:
    try:
        generator = SHAPE_GENERATORS[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported shape kind: {kind}") from exc
    return generator(radius=radius, seed=seed, detail_multiplier=detail_multiplier)


def generate_shape_validation_reports(radius: float = 1.0, seed: int = 42) -> list[MeshValidationReport]:
    reports: list[MeshValidationReport] = []
    for shape_name in SHAPE_GENERATORS:
        mesh_data = generate_shape_mesh_data(shape_name, seed=seed, radius=radius)
        reports.append(validate_shape_mesh_data(shape_name, mesh_data))
    return reports


def _build_dynamic_geom(name: str, mesh_data: ShapeMeshData) -> GeomNode:
    fmt = GeomVertexFormat.getV3n3c4t2()
    vdata = GeomVertexData(name, fmt, Geom.UHDynamic)
    vdata.setNumRows(mesh_data.vertices.shape[0])

    vertex_writer = GeomVertexWriter(vdata, "vertex")
    normal_writer = GeomVertexWriter(vdata, "normal")
    color_writer = GeomVertexWriter(vdata, "color")
    texcoord_writer = GeomVertexWriter(vdata, "texcoord")

    for vertex in mesh_data.vertices:
        vertex_writer.addData3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
    for normal in mesh_data.normals:
        normal_writer.addData3f(float(normal[0]), float(normal[1]), float(normal[2]))
    for color in mesh_data.colors:
        color_writer.addData4f(float(color[0]), float(color[1]), float(color[2]), float(color[3]))
    for texcoord in mesh_data.texcoords:
        texcoord_writer.addData2f(float(texcoord[0]), float(texcoord[1]))

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in mesh_data.indices:
        triangles.addVertices(int(tri[0]), int(tri[1]), int(tri[2]))
    triangles.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(triangles)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


class DynamicGelLayer:
    def __init__(self, parent: NodePath, name: str, mesh_data: ShapeMeshData, material: GelMaterialConfig, shader: Shader, bin_sort: int) -> None:
        self.node = parent.attachNewNode(_build_dynamic_geom(name, mesh_data))
        self.node.setShader(shader)
        self.node.setTwoSided(False)
        self.transparent = material.opacity < 0.85
        self.base_bin_sort = bin_sort
        self.node.setTransparency(TransparencyAttrib.M_alpha if self.transparent else TransparencyAttrib.M_none)
        self.node.setDepthWrite(True)
        self.node.setDepthTest(True)
        self.node.setBin("transparent" if self.transparent else "fixed", bin_sort)
        self.node.setShaderInput("base_color", Vec4(*material.base_color))
        self.node.setShaderInput("opacity", float(material.opacity))
        self.node.setShaderInput("transmission_strength", float(material.transmission_strength))
        self.node.setShaderInput("cloudiness", float(material.cloudiness))
        self.node.setShaderInput("fresnel_strength", float(material.fresnel_strength))
        self.node.setShaderInput("specular_strength", float(material.specular_strength))
        self.node.setShaderInput("emissive_strength", float(material.emissive_strength))
        self.node.setShaderInput("absorption_strength", float(material.absorption_strength))
        self.node.setShaderInput("rear_face_strength", float(material.rear_face_strength))
        self.node.setShaderInput("thickness_absorption", float(material.thickness_absorption))
        self.node.setShaderInput("minimum_body_light", float(material.minimum_body_light))
        self.node.setShaderInput("exposure", float(material.exposure))
        self.node.setShaderInput("inner_layer_opacity", float(material.inner_layer_opacity))
        self.node.setShaderInput("audio_high", 0.0)
        self.node.setShaderInput("audio_rms", 0.0)
        self.node.setShaderInput("time", 0.0)
        self.node.setShaderInput("layer_scale", float(material.inner_scale))
        self.node.setShaderInput("camera_world_position", Vec3(0.0, -12.0, 5.0))
        self.node.setShaderInput("key_light_position", Vec3(-5.0, -7.0, 10.0))
        self.node.setShaderInput("fill_light_position", Vec3(-4.5, -4.0, 4.0))
        self.node.setShaderInput("back_light_position", Vec3(2.5, 7.5, 6.0))
        self.node.setShaderInput("key_light_color", Vec3(0.95, 0.98, 1.02))
        self.node.setShaderInput("fill_light_color", Vec3(0.20, 0.22, 0.25))
        self.node.setShaderInput("back_light_color", Vec3(0.82, 0.92, 1.06))

        geom = self.node.node().modifyGeom(0)
        vdata = geom.modifyVertexData()
        self.vertex_rewriter = GeomVertexRewriter(vdata, "vertex")
        self.normal_rewriter = GeomVertexRewriter(vdata, "normal")

    def write(self, vertices: np.ndarray, normals: np.ndarray) -> None:
        self.vertex_rewriter.setRow(0)
        self.normal_rewriter.setRow(0)
        for vertex, normal in zip(vertices, normals, strict=True):
            self.vertex_rewriter.setData3f(float(vertex[0]), float(vertex[1]), float(vertex[2]))
            self.normal_rewriter.setData3f(float(normal[0]), float(normal[1]), float(normal[2]))

    def update_bin_sort(self, bin_sort: int) -> None:
        if self.transparent:
            self.node.setBin("transparent", bin_sort)
        else:
            self.node.setBin("fixed", self.base_bin_sort)


def _make_bullet_shape(collider: dict):
    kind = collider["kind"]
    if kind == "box":
        return BulletBoxShape(collider["half_extents"])
    if kind == "convex_hull":
        shape = BulletConvexHullShape()
        for point in collider["points"]:
            shape.addPoint(Point3(float(point[0]), float(point[1]), float(point[2])))
        return shape
    if kind == "multi_sphere":
        return BulletMultiSphereShape(collider["points"], collider["radii"])
    raise ValueError(f"Unsupported collider kind: {kind}")


class GummyObject:
    _shader: Shader | None = None

    def __init__(self, parent: NodePath, world, rng: np.random.Generator, config: GummyConfig, controls: GelatinControls) -> None:
        self.config = config
        self.controls = controls
        self.rng = rng
        self.world = world
        self.last_speed = 0.0
        self.last_spin = 0.0
        self.wobble_energy = 0.0
        self.ripple_energy = 0.0
        self.bend_state = np.zeros(2, dtype=np.float32)
        self.bend_velocity = np.zeros(2, dtype=np.float32)
        self.lag_state = np.zeros(3, dtype=np.float32)
        self.squash_state = 0.0
        self.inflate_state = 0.0
        self.shimmer = 0.0
        self.next_beat_time = -999.0
        self.next_onset_time = -999.0
        self.last_impulse_world = Vec3(0.0, 0.0, 1.0)

        self.mesh_data = generate_shape_mesh_data(config.shape_kind, config.shape_seed, radius=config.radius)
        self.rest_vertices = self.mesh_data.vertices.copy()
        self.rest_normals = self.mesh_data.normals.copy()
        self.edge_mask = self.mesh_data.edge_mask.copy()
        self.indices = self.mesh_data.indices

        body_node = BulletRigidBodyNode(config.name)
        body_node.setMass(config.mass)
        body_node.addShape(_make_bullet_shape(self.mesh_data.collider))
        body_node.setFriction(0.72)
        body_node.setRestitution(0.14)
        body_node.setLinearDamping(0.10)
        body_node.setAngularDamping(0.12)
        self.body_np = parent.attachNewNode(body_node)
        self.body_np.setPos(*config.position)
        world.attachRigidBody(body_node)

        self.visual_root = self.body_np.attachNewNode(f"{config.name}-visual-root")
        shader = self._get_shader()
        outer_material = GelMaterialConfig(
            base_color=config.color,
            opacity=np.clip(controls.gelatin_opacity, 0.38, 0.90),
            transmission_strength=np.clip(controls.transmission_strength, 0.0, 3.0),
            cloudiness=np.clip(controls.gelatin_cloudiness, 0.0, 1.8),
            fresnel_strength=np.clip(controls.fresnel_strength, 0.0, 3.0),
            specular_strength=np.clip(controls.specular_strength, 0.12, 3.0),
            emissive_strength=0.02,
            absorption_strength=0.62,
            rear_face_strength=np.clip(controls.rear_face_strength, 0.0, 1.0),
            thickness_absorption=np.clip(controls.thickness_absorption, 0.0, 2.0),
            minimum_body_light=np.clip(controls.minimum_body_light, 0.0, 1.0),
            exposure=np.clip(controls.exposure, 0.5, 3.0),
            inner_layer_opacity=np.clip(controls.inner_layer_opacity, 0.0, 0.5),
            inner_scale=1.0,
        )
        inner_color = (
            min(config.color[0] * 1.03, 1.0),
            min(config.color[1] * 1.03, 1.0),
            min(config.color[2] * 1.04, 1.0),
            config.color[3],
        )
        inner_material = GelMaterialConfig(
            base_color=inner_color,
            opacity=np.clip(controls.inner_layer_opacity, 0.0, 0.04),
            transmission_strength=np.clip(controls.transmission_strength * 0.90, 0.0, 3.0),
            cloudiness=np.clip(controls.gelatin_cloudiness * 1.16, 0.0, 2.2),
            fresnel_strength=np.clip(controls.fresnel_strength * 0.10, 0.0, 2.0),
            specular_strength=np.clip(controls.specular_strength * 0.04, 0.0, 0.4),
            emissive_strength=0.01,
            absorption_strength=0.36,
            rear_face_strength=np.clip(controls.rear_face_strength * 0.25, 0.0, 1.0),
            thickness_absorption=np.clip(controls.thickness_absorption * 1.08, 0.0, 2.0),
            minimum_body_light=np.clip(controls.minimum_body_light * 1.14, 0.0, 1.0),
            exposure=np.clip(controls.exposure * 1.02, 0.5, 3.5),
            inner_layer_opacity=np.clip(controls.inner_layer_opacity, 0.0, 0.5),
            inner_scale=0.992,
        )
        self.outer_layer = DynamicGelLayer(self.visual_root, f"{config.name}-outer", self.mesh_data, outer_material, shader, bin_sort=30)
        self.inner_layer = DynamicGelLayer(self.visual_root, f"{config.name}-inner", self.mesh_data, inner_material, shader, bin_sort=29)
        if inner_material.opacity <= 0.001:
            self.inner_layer.node.hide()

    @classmethod
    def _get_shader(cls) -> Shader:
        if cls._shader is None:
            cls._shader = Shader.load(
                Shader.SL_GLSL,
                vertex=Filename.fromOsSpecific(str(SHADER_DIR / "gelatin.vert")),
                fragment=Filename.fromOsSpecific(str(SHADER_DIR / "gelatin.frag")),
            )
        return cls._shader

    @property
    def body(self) -> BulletRigidBodyNode:
        return self.body_np.node()

    def destroy(self) -> None:
        self.world.removeRigidBody(self.body)
        self.body_np.removeNode()

    def apply_music(self, sim_time: float, sample: AudioSample, dt: float) -> None:
        lateral = Vec3(float(self.rng.uniform(-0.9, 0.9)), float(self.rng.uniform(-0.9, 0.9)), 0.0)
        if lateral.lengthSquared() < 1e-6:
            lateral = Vec3(0.3, 0.0, 0.0)
        lateral.normalize()

        if sample.is_beat and sim_time >= self.next_beat_time and sample.bass > 0.18:
            impulse = lateral * (0.28 + 0.28 * self.config.beat_gain * sample.mid) + Vec3(0.0, 0.0, (2.0 + 2.1 * sample.bass) * self.config.beat_gain)
            self.body.applyCentralImpulse(impulse)
            self.last_impulse_world = Vec3(impulse)
            self.next_beat_time = sim_time + self.config.beat_cooldown
            self.wobble_energy += (0.18 + sample.bass * 0.24) * self.controls.wobble_strength
            self.inflate_state += sample.bass * 0.08

        if sample.onset > 0.52 and sim_time >= self.next_onset_time:
            torque = Vec3(float(self.rng.uniform(-1.0, 1.0)), float(self.rng.uniform(-1.0, 1.0)), float(self.rng.uniform(-0.4, 0.4)))
            if torque.lengthSquared() < 1e-6:
                torque = Vec3(0.0, 1.0, 0.0)
            torque.normalize()
            self.body.applyTorqueImpulse(torque * (0.8 + 1.2 * sample.onset * self.config.onset_gain))
            self.next_onset_time = sim_time + self.config.onset_cooldown
            self.ripple_energy += 0.16 * self.controls.deformation_strength
            self.wobble_energy += 0.08

        self.wobble_energy += sample.mid * self.config.wobble_gain * dt * 0.42 * self.controls.wobble_strength
        self.bend_velocity[0] += (sample.mid - 0.45) * dt * 0.40
        self.bend_velocity[1] += (sample.high - 0.35) * dt * 0.18
        self.shimmer = 0.92 * self.shimmer + 0.08 * sample.high

    def update_visual(self, dt: float, sim_time: float, sample: AudioSample, camera_position: Vec3) -> None:
        linear_velocity = self.body.getLinearVelocity()
        angular_velocity = self.body.getAngularVelocity()
        speed = float(linear_velocity.length())
        spin = float(angular_velocity.length())
        impact_boost = max(0.0, self.last_speed - speed)
        self.last_speed = speed
        self.last_spin = spin

        inverse_quat: Quat = self.body_np.getQuat().conjugate()
        velocity_local = inverse_quat.xform(linear_velocity)
        impulse_axis_local = inverse_quat.xform(self.last_impulse_world)
        impulse_axis = np.asarray([impulse_axis_local.x, impulse_axis_local.y, impulse_axis_local.z], dtype=np.float32)
        if np.linalg.norm(impulse_axis) < 1e-5:
            impulse_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        impulse_axis /= max(float(np.linalg.norm(impulse_axis)), 1e-6)

        speed_factor = speed * 0.06 + spin * 0.012 + impact_boost * 0.16 + sample.bass * self.config.bass_sensitivity
        self.squash_state = float(np.clip(0.90 * self.squash_state + 0.14 * speed_factor * self.controls.deformation_strength, 0.0, 0.16))
        self.inflate_state = float(np.clip(0.94 * self.inflate_state + sample.bass * 0.010, 0.0, 0.10))
        self.wobble_energy = max(0.0, self.wobble_energy * np.exp(-dt * 2.3) + impact_boost * 0.02)
        self.ripple_energy = max(0.0, self.ripple_energy * np.exp(-dt * 4.0) + sample.onset * 0.004)

        lag_target = np.array([velocity_local.x, velocity_local.y, velocity_local.z], dtype=np.float32) * 0.018
        self.lag_state += (lag_target - self.lag_state) * min(1.0, dt * 4.2)

        self.bend_velocity *= np.exp(-dt * 5.0)
        self.bend_state += self.bend_velocity * dt * 5.0
        self.bend_state *= np.exp(-dt * 2.2)
        bend = np.clip(self.bend_state, -0.08, 0.08) * self.controls.deformation_strength

        vertices = self._deform_vertices(sim_time, sample, impulse_axis, bend)
        normals = _compute_vertex_normals(vertices, self.indices)
        inner_vertices = vertices * 0.97

        self.outer_layer.write(vertices, normals)
        self.inner_layer.write(inner_vertices, normals)

        shimmer_value = sample.high * 0.7 + self.shimmer * 0.3
        rms_value = sample.rms * 0.75 + self.wobble_energy * 0.08
        for layer in (self.outer_layer.node, self.inner_layer.node):
            layer.setShaderInput("time", float(sim_time))
            layer.setShaderInput("audio_high", float(shimmer_value))
            layer.setShaderInput("audio_rms", float(rms_value))
            layer.setShaderInput("camera_world_position", camera_position)

    def _deform_vertices(self, sim_time: float, sample: AudioSample, squash_axis: np.ndarray, bend: np.ndarray) -> np.ndarray:
        vertices = self.rest_vertices.copy()
        axis = squash_axis / max(float(np.linalg.norm(squash_axis)), 1e-6)
        squash = np.clip(self.squash_state + sample.bass * 0.03, 0.0, 0.18)
        inflate = np.clip(self.inflate_state + sample.bass * 0.02, 0.0, 0.10)

        dot_axis = vertices @ axis
        axis_component = np.outer(dot_axis, axis)
        perp_component = vertices - axis_component
        kind = self.config.shape_kind
        face_bias = 1.0 - self.edge_mask
        corner_bias = self.edge_mask
        shape_squash = {
            "rounded_cube": 0.18,
            "rectangular_cuboid": 0.20,
            "triangular_prism": 0.18,
            "rounded_tetrahedron": 0.14,
            "rounded_octahedron": 0.15,
            "hexagonal_prism": 0.18,
            "torus_ring": 0.14,
        }.get(kind, 0.16)
        vertices += -axis_component * (squash * shape_squash / 0.16) + perp_component * (0.12 * squash + inflate * 0.18)
        vertices += self.rest_normals * ((inflate * 0.12 + squash * 0.06) * face_bias[:, None])

        edge_mask = self.edge_mask[:, None]
        up_curve = np.clip((self.rest_vertices[:, 2] / max(self.config.radius, 1e-6)) + 0.25, -1.0, 1.2)
        bend_scale = {
            "rounded_cube": 0.10,
            "rectangular_cuboid": 0.16,
            "triangular_prism": 0.14,
            "rounded_tetrahedron": 0.08,
            "rounded_octahedron": 0.09,
            "hexagonal_prism": 0.13,
            "torus_ring": 0.10,
        }.get(kind, 0.10)
        vertices[:, 0] += bend[0] * up_curve * (bend_scale * self.config.radius) * edge_mask[:, 0]
        vertices[:, 1] += bend[1] * up_curve * ((bend_scale * 0.85) * self.config.radius) * edge_mask[:, 0]

        lag = self.lag_state * self.controls.deformation_strength
        lag_scale = {
            "rounded_cube": 0.20,
            "rectangular_cuboid": 0.26,
            "triangular_prism": 0.24,
            "rounded_tetrahedron": 0.18,
            "rounded_octahedron": 0.18,
            "hexagonal_prism": 0.24,
            "torus_ring": 0.22,
        }.get(kind, 0.20)
        vertices[:, 0] -= lag[0] * lag_scale * (0.65 + 0.55 * edge_mask[:, 0])
        vertices[:, 1] -= lag[1] * lag_scale * (0.65 + 0.55 * edge_mask[:, 0])
        vertices[:, 2] -= lag[2] * lag_scale * 0.62
        vertices += self.rest_normals * (self.wobble_energy * 0.012 * face_bias[:, None])

        if kind == "rectangular_cuboid":
            shear = 0.06 * np.sin(sim_time * 4.2 + self.rest_vertices[:, 2] * 1.2) * (sample.mid + self.wobble_energy)
            vertices[:, 0] += shear * self.rest_vertices[:, 1]
        elif kind == "triangular_prism":
            vertices[:, 0] += 0.05 * corner_bias * np.sin(sim_time * 5.0 + self.rest_vertices[:, 2] * 2.0)
        elif kind == "rounded_tetrahedron" or kind == "rounded_octahedron":
            twist = 0.04 * corner_bias * np.sin(sim_time * (6.0 + self.last_spin * 0.05))
            vertices[:, 0] += twist * self.rest_vertices[:, 1]
            vertices[:, 1] -= twist * self.rest_vertices[:, 0]
        elif kind == "hexagonal_prism":
            torsion = 0.05 * (sample.mid + self.wobble_energy) * np.sin(sim_time * 4.4)
            vertices[:, 0] += torsion * self.rest_vertices[:, 2]
            vertices[:, 1] -= torsion * self.rest_vertices[:, 2] * 0.7
        elif kind == "torus_ring":
            radial = np.sqrt(self.rest_vertices[:, 0] ** 2 + self.rest_vertices[:, 1] ** 2)
            ring_wave = np.sin(sim_time * 4.8 + radial * 2.2) * (0.05 + sample.bass * 0.03)
            vertices[:, 2] += ring_wave * (0.4 + corner_bias)

        ripple = np.sin((self.rest_vertices @ np.array([1.0, -0.7, 0.5], dtype=np.float32)) / max(self.config.radius, 1e-6) * 2.2 - sim_time * 6.8)
        vertices += self.rest_normals * (ripple[:, None] * self.ripple_energy * 0.030 * edge_mask)

        return vertices.astype(np.float32)

    def update_sort(self, camera_distance: float) -> None:
        base_sort = 20 + int(np.clip(camera_distance * 1.2, 0.0, 200.0))
        self.inner_layer.update_bin_sort(base_sort)
        self.outer_layer.update_bin_sort(base_sort + 1)
