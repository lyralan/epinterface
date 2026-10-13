"""Kiva ground heat transfer for slab-on-grade floors.

By default the builder sets Site:GroundTemperature:BuildingSurface from the thermostat setpoints,
so a slab exchanges heat with a ground held a little below the setpoint all year. Kiva instead
solves two-dimensional heat conduction in the soil around the slab edge at every timestep, which
captures seasonal storage in the soil and edge losses. The defaults follow OpenStudio-HPXML
(ResStock): soil properties, a 40 m far field and deep ground with a zero-flux bottom, and a
foundation wall 0.1016 m above grade and 0.2032 m below the slab.

Only slab-on-grade floors are handled. The slab construction is the GroundSlab assembly as built
(it should hold the slab and its finish, not under-slab insulation, which Kiva models from the
fields below).
"""

from typing import Literal

import numpy as np
from archetypal.idfclass import IDF
from pydantic import BaseModel, Field

KIVA_FOUNDATION_NAME = "KivaSlabFoundation"
KIVA_INSULATION_CONDUCTIVITY = 0.029427  # W/m.K, as OpenStudio-HPXML's rigid insulation


class KivaFoundation(BaseModel, extra="forbid"):
    """A Kiva slab-on-grade foundation (Foundation:Kiva + Foundation:Kiva:Settings)."""

    SoilConductivity: float = Field(default=1.9041, gt=0, title="Soil conductivity [W/m.K]")
    SoilDensity: float = Field(default=1842.3, gt=0, title="Soil density [kg/m3]")
    SoilSpecificHeat: float = Field(default=418.7, gt=0, title="Soil specific heat [J/kg.K]")
    GroundSolarAbsorptivity: float = Field(default=0.9, ge=0, le=1)
    GroundThermalAbsorptivity: float = Field(default=0.9, ge=0, le=1)
    GroundSurfaceRoughness: float = Field(default=0.03, ge=0, title="[m]")
    FarFieldWidth: float = Field(default=40, gt=0, title="[m]")
    DeepGroundBoundaryCondition: Literal["ZeroFlux", "GroundWater", "Autoselect"] = "ZeroFlux"
    DeepGroundDepth: float = Field(default=40, gt=0, title="[m]")
    MinimumCellDimension: float = Field(default=0.2, gt=0, title="[m]")
    MaximumCellGrowthCoefficient: float = Field(default=3, gt=1)
    WallHeightAboveGrade: float = Field(default=0.1016, ge=0, title="[m]")
    WallDepthBelowSlab: float = Field(default=0.2032, ge=0, title="[m]")
    InitialIndoorAirTemperature: float = Field(default=15.5556, title="[degC]")
    UnderSlabRValue: float = Field(
        default=0, ge=0, title="Interior horizontal (under-slab) insulation R-value [m2.K/W]"
    )
    UnderSlabWidth: float = Field(
        default=0, ge=0, title="Width of the under-slab insulation from the slab edge [m]"
    )
    PerimeterRValue: float = Field(
        default=0, ge=0, title="Interior vertical (slab edge) insulation R-value [m2.K/W]"
    )
    PerimeterDepth: float = Field(
        default=0.1016, ge=0, title="Depth of the slab edge insulation [m]"
    )

    def _insulation(self, idf: IDF, name: str, r_value: float) -> str:
        idf.newidfobject(
            "MATERIAL",
            Name=name,
            Roughness="Rough",
            Thickness=round(r_value * KIVA_INSULATION_CONDUCTIVITY, 5),
            Conductivity=KIVA_INSULATION_CONDUCTIVITY,
            Density=32.04,
            Specific_Heat=1214.23,
        )
        return name

    def add(self, idf: IDF) -> IDF:
        """Put every slab-on-grade floor (outside boundary "Ground") on this Kiva foundation.

        Each slab gets its exposed perimeter: the length of its edges on the outer footprint
        boundary, so core/perimeter zoning gives core slabs zero exposed perimeter.

        Args:
            idf (IDF): The model, after geometry and constructions are added.

        Returns:
            idf (IDF): The model with Kiva objects added.
        """
        slabs = [
            srf
            for srf in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
            if srf.Outside_Boundary_Condition.lower() == "ground"
            and srf.Surface_Type.lower() == "floor"
        ]
        if not slabs:
            return idf
        ground_walls = [
            srf
            for srf in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
            if srf.Outside_Boundary_Condition.lower() == "ground"
            and srf.Surface_Type.lower() == "wall"
        ]
        if ground_walls:
            msg = "Kiva is implemented for slab-on-grade floors only, not basements."
            raise NotImplementedError(msg)

        idf.newidfobject(
            "FOUNDATION:KIVA:SETTINGS",
            Soil_Conductivity=self.SoilConductivity,
            Soil_Density=self.SoilDensity,
            Soil_Specific_Heat=self.SoilSpecificHeat,
            Ground_Solar_Absorptivity=self.GroundSolarAbsorptivity,
            Ground_Thermal_Absorptivity=self.GroundThermalAbsorptivity,
            Ground_Surface_Roughness=self.GroundSurfaceRoughness,
            FarField_Width=self.FarFieldWidth,
            DeepGround_Boundary_Condition=self.DeepGroundBoundaryCondition,
            DeepGround_Depth=self.DeepGroundDepth,
            Minimum_Cell_Dimension=self.MinimumCellDimension,
            Maximum_Cell_Growth_Coefficient=self.MaximumCellGrowthCoefficient,
            Simulation_Timestep="Timestep",
        )
        insulation = {}
        if self.UnderSlabRValue > 0 and self.UnderSlabWidth > 0:
            insulation.update(
                Interior_Horizontal_Insulation_Material_Name=self._insulation(
                    idf, f"{KIVA_FOUNDATION_NAME}_UnderSlabInsulation", self.UnderSlabRValue
                ),
                Interior_Horizontal_Insulation_Depth=0,
                Interior_Horizontal_Insulation_Width=self.UnderSlabWidth,
            )
        if self.PerimeterRValue > 0 and self.PerimeterDepth > 0:
            insulation.update(
                Interior_Vertical_Insulation_Material_Name=self._insulation(
                    idf, f"{KIVA_FOUNDATION_NAME}_PerimeterInsulation", self.PerimeterRValue
                ),
                Interior_Vertical_Insulation_Depth=self.PerimeterDepth,
            )
        idf.newidfobject(
            "FOUNDATION:KIVA",
            Name=KIVA_FOUNDATION_NAME,
            Initial_Indoor_Air_Temperature=self.InitialIndoorAirTemperature,
            Wall_Height_Above_Grade=self.WallHeightAboveGrade,
            Wall_Depth_Below_Slab=self.WallDepthBelowSlab,
            **insulation,
        )

        footprint = np.vstack([np.array(s.coords, dtype=float)[:, :2] for s in slabs])
        lo, hi = footprint.min(axis=0), footprint.max(axis=0)
        for srf in slabs:
            srf.Outside_Boundary_Condition = "Foundation"
            srf.Outside_Boundary_Condition_Object = KIVA_FOUNDATION_NAME
            srf.Sun_Exposure = "NoSun"
            srf.Wind_Exposure = "NoWind"
            idf.newidfobject(
                "SURFACEPROPERTY:EXPOSEDFOUNDATIONPERIMETER",
                Surface_Name=srf.Name,
                Exposed_Perimeter_Calculation_Method="TotalExposedPerimeter",
                Total_Exposed_Perimeter=round(exposed_perimeter(srf.coords, lo, hi), 4),
            )
        return idf


def exposed_perimeter(coords, lo: np.ndarray, hi: np.ndarray, tol: float = 1e-3) -> float:
    """Length of a floor polygon's edges that lie on the rectangular footprint boundary [lo, hi]."""
    pts = np.array(coords, dtype=float)[:, :2]
    total = 0.0
    for a, b in zip(pts, np.roll(pts, -1, axis=0), strict=True):
        on_x = any(abs(a[0] - x) < tol and abs(b[0] - x) < tol for x in (lo[0], hi[0]))
        on_y = any(abs(a[1] - y) < tol and abs(b[1] - y) < tol for y in (lo[1], hi[1]))
        if on_x or on_y:
            total += float(np.linalg.norm(b - a))
    return total
