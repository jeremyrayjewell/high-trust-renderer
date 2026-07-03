from __future__ import annotations

from dataclasses import dataclass

from .variant_specs import GENERATED_VARIANTS


@dataclass(frozen=True)
class ModeMeta:
    display_name: str
    family: str
    palette: str
    motifs: tuple[str, ...]
    dominant_geometry: tuple[str, ...]
    movement_type: tuple[str, ...]
    camera_roles: tuple[str, ...]
    audio_mappings: tuple[str, ...]
    persistent_elements: tuple[str, ...]
    suitable_presets: tuple[str, ...]


BASE_MODE_MANIFEST: dict[str, ModeMeta] = {
    "endless_highway": ModeMeta("Endless Highway", "transit", "neon_nocturne", ("road plane", "lane markers", "light posts", "tunnel horizon"), ("rails", "signage_blocks", "columns", "arches"), ("depth_pass", "slide_past", "parallax"), ("forward_travel", "reverse_pullback", "low_flythrough"), ("beat pulses lane lights", "bass changes speed"), ("fog", "reflections", "scanlines"), ("night_transit",)),
    "glass_mall_atrium": ModeMeta("Glass Mall Atrium", "water_glass", "frutiger_aero", ("escalators", "glass panes", "planters", "reflective floor"), ("ramps", "floating_panels", "columns", "water_slabs"), ("rise_fall", "parallax", "slide_past"), ("wide_establishing", "side_parallax", "wide_crane_rise"), ("energy shifts color", "highs add sparkle"), ("ui panels", "fog", "floor reflections"), ("lofi", "frutiger_water", "high_trust_society")),
    "submerged_city": ModeMeta("Submerged City", "water_glass", "sunken_plaza", ("underwater towers", "wavering skyline", "bubble columns"), ("towers", "water_slabs", "floating_panels", "columns"), ("undulate", "parallax", "rise_fall"), ("long_horizon_glide", "high_overlook_drift", "side_parallax"), ("bass drives pressure waves", "mids sway buildings"), ("bubbles", "fog", "refraction bands"), ("frutiger_water",)),
    "memory_train": ModeMeta("Memory Train", "transit", "bios_ocean", ("train windows", "track rails", "passing signs", "station rhythm"), ("rails", "columns", "signage_blocks", "cuboids"), ("depth_pass", "slide_past", "parallax"), ("close_foreground_pass", "side_tracking_shot", "low_flythrough"), ("drums drive track rhythm", "sections shift scenery"), ("window glare", "speed lines", "mist"), ("lofi", "night_transit")),
    "ring_corridor": ModeMeta("Ring Corridor", "abstract_geometry", "chrome_dream", ("portals", "receding center", "depth fog", "halos"), ("ring_gates", "shards", "columns", "rails"), ("spiral", "depth_pass", "slow_rotate"), ("forward_travel", "spiral_approach", "slow_orbit_around_object"), ("kick expands rings", "snare twists geometry"), ("fog", "particles", "ui traces"), ("polygon_motion_gallery", "bios_dream")),
    "digital_aquarium": ModeMeta("Digital Aquarium", "water_glass", "frutiger_aero", ("bubbles", "glass tanks", "liquid ui", "aqua arches"), ("water_slabs", "arches", "floating_panels", "columns"), ("undulate", "orbit", "rise_fall"), ("side_parallax", "slow_orbit_around_object", "wide_crane_rise"), ("highs create bubbles", "bass bulges water"), ("ui panels", "bubbles", "mist"), ("frutiger_water", "cyber_aero_liquid")),
    "cyber_shrine": ModeMeta("Cyber Shrine", "cyber_urban", "neon_nocturne", ("neon gates", "ritual path", "foggy skyline", "sacred pylons"), ("ring_gates", "columns", "towers", "shards"), ("slow_rotate", "depth_pass", "parallax"), ("low_angle_corridor", "spiral_approach", "wide_crane_rise"), ("beat pulses gates", "energy boosts bloom"), ("fog", "gates", "light trails"), ("cyber_aero_liquid",)),
    "airport_at_night": ModeMeta("Airport At Night", "transit", "bios_ocean", ("runway lights", "moving walkway", "terminal glass", "jet silhouettes"), ("rails", "floating_panels", "columns", "signage_blocks"), ("depth_pass", "slide_past", "parallax"), ("wide_establishing", "long_horizon_glide", "diagonal_dolly"), ("rhythm animates lights", "bass drifts camera"), ("fog", "floor reflections", "panels"), ("lofi", "night_transit")),
    "skybridge_city": ModeMeta("Skybridge City", "cyber_urban", "neon_nocturne", ("elevated walkways", "teal towers", "window grids", "bridge rails"), ("towers", "rails", "signage_blocks", "floating_panels"), ("parallax", "slide_past", "long_glide"), ("wide_establishing", "side_parallax", "high_overlook_drift"), ("sections change route", "highs flicker windows"), ("fog", "signs", "particles"), ("lofi", "night_transit")),
    "data_center_dream": ModeMeta("Data Center Dream", "ui_computer", "terminal_mist", ("server aisles", "status leds", "cable arcs", "fan tunnels"), ("cuboids", "columns", "cable_arcs", "floating_panels"), ("depth_pass", "sway", "parallax"), ("forward_travel", "low_flythrough", "vertical_elevator_rise"), ("percussion blinks leds", "bass distorts fog"), ("scanlines", "fog", "ui strips"), ("high_trust_society",)),
    "liquid_chrome_room": ModeMeta("Liquid Chrome Room", "water_glass", "chrome_dream", ("chrome pools", "reflective blobs", "ripple rings", "mirror walls"), ("water_slabs", "shards", "ring_gates", "floating_panels"), ("undulate", "orbit", "slow_rotate"), ("close_foreground_pass", "slow_orbit_around_object", "spiral_approach"), ("bass drives waves", "mids deform blobs"), ("ripples", "reflections", "fog"), ("cyber_aero_liquid",)),
    "bios_temple": ModeMeta("BIOS Temple", "ui_computer", "bios_ocean", ("floating menu panels", "grid floor", "orbits", "bios glyphs"), ("floating_panels", "ring_gates", "columns", "shards"), ("orbit", "slow_rotate", "rise_fall"), ("wide_establishing", "vertical_elevator_rise", "slow_orbit_around_object"), ("beat pulses symbols", "chords alter orbits"), ("scanlines", "dither", "panels"), ("bios_dream",)),
    "solar_sail_space": ModeMeta("Solar Sail Space", "space_cosmic", "bios_ocean", ("station rings", "solar sails", "stars", "lens haze"), ("ring_gates", "shards", "antenna_arrays", "floating_panels"), ("slow_rotate", "orbit", "parallax"), ("wide_establishing", "high_overlook_drift", "slow_orbit_around_object"), ("pads move sails", "energy expands station"), ("stars", "fog", "ui traces"), ("camera_direction_gallery",)),
    "neon_rain_alley": ModeMeta("Neon Rain Alley", "cyber_urban", "rain_alley", ("wet street", "vending signs", "rain streaks", "alley towers"), ("signage_blocks", "towers", "water_slabs", "columns"), ("slide_past", "undulate", "parallax"), ("close_foreground_pass", "side_tracking_shot", "crossing_parallax_pan"), ("snare flickers signs", "bass ripples puddles"), ("rain", "fog", "reflections"), ("cyber_aero_liquid",)),
    "dream_office_lobby": ModeMeta("Dream Office Lobby", "architecture_interior", "vapor_civic", ("office glass", "fountain", "columns", "crt desk wall"), ("columns", "water_slabs", "cuboids", "floating_panels"), ("parallax", "rise_fall", "slide_past"), ("wide_establishing", "wide_crane_rise", "close_object_pass"), ("pads shift lighting", "beat lifts fountain"), ("fog", "panels", "floor bands"), ("high_trust_society",)),
    "arcology_exterior": ModeMeta("Arcology Exterior", "cyber_urban", "terminal_mist", ("mega towers", "flying rails", "distant fog", "stacked megastructures"), ("towers", "rails", "signage_blocks", "columns"), ("parallax", "long_glide", "slide_past"), ("wide_establishing", "high_overlook_drift", "reverse_pullback"), ("intensity changes altitude", "bass adds movement"), ("fog", "signs", "particles"), ("high_trust_society",)),
    "soft_geometry_field": ModeMeta("Soft Geometry Field", "abstract_geometry", "chrome_dream", ("rounded cubes", "spheres", "ribbons", "floating solids"), ("prisms", "cuboids", "water_slabs", "shards"), ("tumble", "orbit", "rise_fall"), ("slow_orbit_around_object", "wide_crane_rise", "close_object_pass"), ("bands scale objects", "mids rotate shapes"), ("particles", "fog", "soft highlights"), ("polygon_motion_gallery",)),
    "vapor_plaza": ModeMeta("Vapor Plaza", "nature_weather", "vapor_civic", ("tiles", "public square", "fountains", "open sky"), ("water_slabs", "ramps", "columns", "floating_panels"), ("jet_pulse", "long_glide", "parallax"), ("wide_establishing", "long_horizon_glide", "wide_crane_rise"), ("beat drives fountains", "chords shift sky"), ("mist", "grid lines", "panels"), ("lofi",)),
    "crt_navigation_map": ModeMeta("CRT Navigation Map", "ui_computer", "terminal_mist", ("radar circles", "route lines", "map panels", "grids"), ("floating_panels", "ring_gates", "columns", "rails"), ("slow_rotate", "orbit", "slide_past"), ("calm_horizon_hold", "slow_orbit_around_object", "high_overlook_drift"), ("drums create scan jumps", "bass zooms map"), ("scanlines", "grid", "ui panels"), ("bios_dream",)),
    "moon_pool": ModeMeta("Moon Pool", "water_glass", "chrome_dream", ("black water", "moon rings", "platforms", "reflected stars"), ("water_slabs", "ring_gates", "columns", "rails"), ("undulate", "long_glide", "slow_rotate"), ("calm_horizon_hold", "long_horizon_glide", "slow_orbit_around_object"), ("bass creates ripple radius", "highs shimmer"), ("mist", "rings", "reflection streaks"), ("lofi",)),
    "polygon_garden": ModeMeta("Polygon Garden", "nature_weather", "frutiger_aero", ("low poly plants", "glass domes", "flower beds", "pastel ui"), ("low_poly_plants", "floating_panels", "pyramids", "water_slabs"), ("sway", "rise_fall", "orbit"), ("wide_establishing", "high_overlook_drift", "wide_crane_rise"), ("melody opens growth", "energy expands petals"), ("ui traces", "mist", "sparkles"), ("frutiger_water",)),
    "virtual_hotel": ModeMeta("Virtual Hotel", "architecture_interior", "vapor_civic", ("corridors", "carpet stripes", "elevators", "lounge lamps"), ("columns", "cuboids", "floating_panels", "rails"), ("depth_pass", "slide_past", "parallax"), ("close_foreground_pass", "reverse_pullback", "low_angle_corridor"), ("sections alter floors", "bass shifts camera"), ("fog", "panels", "hall lights"), ("high_trust_society",)),
    "dream_metro": ModeMeta("Dream Metro", "transit", "rain_alley", ("platform lights", "train silhouette", "tunnel blur", "fog signage"), ("rails", "columns", "signage_blocks", "arches"), ("depth_pass", "slide_past", "parallax"), ("low_angle_corridor", "low_flythrough", "close_foreground_pass"), ("tempo drives train speed", "kick flashes tunnel"), ("fog", "speed lines", "light rows"), ("night_transit",)),
    "hologram_market": ModeMeta("Hologram Market", "cyber_urban", "neon_nocturne", ("stalls", "hologram ads", "awning signs", "street canyon"), ("signage_blocks", "floating_panels", "cuboids", "columns"), ("slide_past", "parallax", "lateral_drift"), ("forward_travel", "side_tracking_shot", "close_object_pass"), ("mids animate signs", "beat brightens stalls"), ("fog", "signs", "particles"), ("cyber_aero_liquid",)),
    "blue_os_desktop_world": ModeMeta("Blue OS Desktop World", "ui_computer", "bios_ocean", ("desktop windows", "task bars", "floating icons", "blue gradients"), ("floating_panels", "cuboids", "columns", "rails"), ("orbit", "rise_fall", "slide_past"), ("side_parallax", "vertical_elevator_rise", "slow_orbit_around_object"), ("beat pulses windows", "mids move routes"), ("scanlines", "ui panels", "grids"), ("bios_dream",)),
    "weather_simulation_room": ModeMeta("Weather Simulation Room", "nature_weather", "frutiger_aero", ("storm chamber", "cloud bands", "terrain basin", "weather monitors"), ("water_slabs", "floating_panels", "columns", "pyramids"), ("undulate", "slow_rotate", "rise_fall"), ("wide_establishing", "wide_crane_rise", "high_overlook_drift"), ("energy drives storms", "bass deepens clouds"), ("fog", "ui rings", "panels"), ("frutiger_water",)),
    "crystal_server_lake": ModeMeta("Crystal Server Lake", "water_glass", "terminal_mist", ("crystal pylons", "server reflections", "lake surface", "fiber glints"), ("shards", "water_slabs", "cuboids", "columns"), ("undulate", "sway", "parallax"), ("calm_horizon_hold", "wide_crane_rise", "slow_orbit_around_object"), ("bass ripples lake", "highs glint crystals"), ("mist", "ripples", "ui traces"), ("cyber_aero_liquid",)),
    "tunnel_of_screens": ModeMeta("Tunnel Of Screens", "ui_computer", "terminal_mist", ("screen tunnel", "floating windows", "monitor walls", "route strips"), ("floating_panels", "cuboids", "columns", "rails"), ("depth_pass", "slide_past", "parallax"), ("low_angle_corridor", "close_foreground_pass", "low_flythrough"), ("beat flashes screens", "energy adds panel depth"), ("scanlines", "fog", "panels"), ("bios_dream",)),
    "rooftop_antenna_field": ModeMeta("Rooftop Antenna Field", "cyber_urban", "bios_ocean", ("roofline silhouettes", "antenna arrays", "dish clusters", "distant city haze"), ("antenna_arrays", "towers", "rails", "cuboids"), ("sway", "parallax", "long_glide"), ("high_overlook_drift", "wide_crane_rise", "crossing_parallax_pan"), ("highs flicker towers", "bass deepens horizon"), ("fog", "particles", "sign lights"), ("lofi",)),
    "fractal_parking_garage": ModeMeta("Fractal Parking Garage", "architecture_interior", "chrome_dream", ("parking ramps", "concrete decks", "sodium lights", "repeating columns"), ("ramps", "columns", "rails", "signage_blocks"), ("depth_pass", "rise_fall", "parallax"), ("low_angle_corridor", "reverse_pullback", "diagonal_dolly"), ("beat pulses sodium lights", "bass deepens ramps"), ("fog", "floor bands", "lights"), ("high_trust_society", "night_transit")),
    "ocean_interface": ModeMeta("Ocean Interface", "water_glass", "sunken_plaza", ("ocean horizon", "glass overlays", "wave planes", "marine hud"), ("water_slabs", "floating_panels", "ring_gates", "rails"), ("undulate", "long_glide", "orbit"), ("calm_horizon_hold", "long_horizon_glide", "high_overlook_drift"), ("bass swells water", "highs add bubbles"), ("ui panels", "mist", "wave bands"), ("frutiger_water",)),
    "green_wireframe_valley": ModeMeta("Green Wireframe Valley", "nature_weather", "terminal_mist", ("terrain grid", "mountain ridges", "horizon sun", "wire hills"), ("rails", "pyramids", "columns", "floating_panels"), ("parallax", "rise_fall", "long_glide"), ("wide_establishing", "high_overlook_drift", "long_horizon_glide"), ("bass reshapes ridges", "beat pulses grid"), ("fog", "particles", "grid traces"), ("camera_direction_gallery",)),
    "neon_monorail": ModeMeta("Neon Monorail", "transit", "neon_nocturne", ("monorail beam", "train pods", "city lights", "speed trails"), ("rails", "pods", "ring_gates", "towers"), ("depth_pass", "slide_past", "parallax"), ("forward_travel", "side_tracking_shot", "spiral_approach"), ("tempo drives pod speed", "bass thickens rails"), ("light rows", "fog", "streaks"), ("lofi", "night_transit")),
    "dream_arcade_floor": ModeMeta("Dream Arcade Floor", "architecture_interior", "rain_alley", ("checker floor", "cabinet silhouettes", "token lights", "neon aisles"), ("cuboids", "columns", "signage_blocks", "floating_panels"), ("slide_past", "parallax", "lateral_drift"), ("side_tracking_shot", "close_object_pass", "wide_crane_rise"), ("beat flashes cabinets", "highs sparkle lights"), ("fog", "floor glow", "panels"), ("high_trust_society",)),
    "satellite_weather_map": ModeMeta("Satellite Weather Map", "ui_computer", "frutiger_aero", ("cyclone spirals", "tracking lines", "weather panels", "radar circles"), ("ring_gates", "floating_panels", "columns", "water_slabs"), ("slow_rotate", "orbit", "rise_fall"), ("high_overlook_drift", "vertical_elevator_rise", "slow_orbit_around_object"), ("drums jolt scan", "bass expands systems"), ("scanlines", "route lines", "panels"), ("bios_dream",)),
    "corporate_fountain_core": ModeMeta("Corporate Fountain Core", "water_glass", "vapor_civic", ("fountain basin", "corporate atrium", "glass rails", "water jets"), ("water_slabs", "columns", "ramps", "floating_panels"), ("jet_pulse", "rise_fall", "parallax"), ("low_angle_corridor", "wide_crane_rise", "vertical_elevator_rise"), ("beat drives jets", "energy brightens lobby"), ("mist", "reflections", "ui traces"), ("lofi", "high_trust_society", "frutiger_water")),
    "memory_beach": ModeMeta("Memory Beach", "nature_weather", "sunken_plaza", ("sunset sky", "ocean bands", "foam lines", "distant towers"), ("water_slabs", "rails", "towers", "signage_blocks"), ("undulate", "long_glide", "parallax"), ("calm_horizon_hold", "long_horizon_glide", "reverse_pullback"), ("bass expands waves", "highs shimmer foam"), ("mist", "wave bands", "soft scanlines"), ("lofi",)),
}


MODE_MANIFEST: dict[str, ModeMeta] = dict(BASE_MODE_MANIFEST)
MODE_MANIFEST.update(
    {
        spec.name: ModeMeta(
            spec.display_name,
            spec.family,
            spec.palette,
            spec.motifs,
            spec.dominant_geometry,
            spec.movement_type,
            spec.camera_roles,
            spec.audio_mappings,
            spec.persistent_elements,
            spec.suitable_presets,
        )
        for spec in GENERATED_VARIANTS
    }
)


def get_mode_meta(name: str) -> ModeMeta:
    return MODE_MANIFEST[name]

