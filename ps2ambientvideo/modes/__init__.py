from .airport_at_night import AirportAtNight
from .arcology_exterior import ArcologyExterior
from .blue_os_desktop_world import BlueOsDesktopWorld
from .corporate_fountain_core import CorporateFountainCore
from .crt_navigation_map import CrtNavigationMap
from .crystal_server_lake import CrystalServerLake
from .cyber_shrine import CyberShrine
from .data_center_dream import DataCenterDream
from .digital_aquarium import DigitalAquarium
from .dream_arcade_floor import DreamArcadeFloor
from .dream_metro import DreamMetro
from .dream_office_lobby import DreamOfficeLobby
from .endless_highway import EndlessHighway
from .fractal_parking_garage import FractalParkingGarage
from .glass_mall_atrium import GlassMallAtrium
from .green_wireframe_valley import GreenWireframeValley
from .hologram_market import HologramMarket
from .liquid_chrome_room import LiquidChromeRoom
from .memory_beach import MemoryBeach
from .memory_train import MemoryTrain
from .moon_pool import MoonPool
from .neon_monorail import NeonMonorail
from .neon_rain_alley import NeonRainAlley
from .ocean_interface import OceanInterface
from .polygon_garden import PolygonGarden
from .bios_temple import BiosTemple
from .ring_corridor import RingCorridor
from .rooftop_antenna_field import RooftopAntennaField
from .satellite_weather_map import SatelliteWeatherMap
from .skybridge_city import SkybridgeCity
from .soft_geometry_field import SoftGeometryField
from .solar_sail_space import SolarSailSpace
from .submerged_city import SubmergedCity
from .tunnel_of_screens import TunnelOfScreens
from .vapor_plaza import VaporPlaza
from .virtual_hotel import VirtualHotel
from .weather_simulation_room import WeatherSimulationRoom
from ..variant_specs import GENERATED_VARIANTS
from ._factory import make_mode


BASE_MODE_CLASSES = [
    EndlessHighway,
    GlassMallAtrium,
    SubmergedCity,
    MemoryTrain,
    RingCorridor,
    DigitalAquarium,
    CyberShrine,
    AirportAtNight,
    SkybridgeCity,
    DataCenterDream,
    LiquidChromeRoom,
    BiosTemple,
    SolarSailSpace,
    NeonRainAlley,
    DreamOfficeLobby,
    ArcologyExterior,
    SoftGeometryField,
    VaporPlaza,
    CrtNavigationMap,
    MoonPool,
    PolygonGarden,
    VirtualHotel,
    DreamMetro,
    HologramMarket,
    BlueOsDesktopWorld,
    WeatherSimulationRoom,
    CrystalServerLake,
    TunnelOfScreens,
    RooftopAntennaField,
    FractalParkingGarage,
    OceanInterface,
    GreenWireframeValley,
    NeonMonorail,
    DreamArcadeFloor,
    SatelliteWeatherMap,
    CorporateFountainCore,
    MemoryBeach,
]


GENERATED_MODE_CLASSES = [
    make_mode(
        spec.name,
        spec.scene,
        spec.palette,
        speed=spec.speed,
        density=spec.density,
        structure=spec.structure,
        atmosphere=spec.atmosphere,
        family=spec.family,
        scene_recipe=spec.scene_recipe,
        dominant_geometry=spec.dominant_geometry,
        movement_type=spec.movement_type,
        camera_roles=spec.camera_roles,
        suitable_presets=spec.suitable_presets,
    )
    for spec in GENERATED_VARIANTS
]


ALL_MODE_CLASSES = [*BASE_MODE_CLASSES, *GENERATED_MODE_CLASSES]
