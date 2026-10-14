"""Systems components for the SBEM library."""

from typing import Literal

from archetypal.idfclass import IDF
from pydantic import Field, model_validator

from epinterface.sbem.common import BoolStr, MetadataMixin, NamedObject
from epinterface.sbem.components.schedules import YearComponent

FuelType = Literal[
    "Electricity",
    "NaturalGas",
    "Propane",
    "FuelOil",
    "WoodPellets",
    "Coal",
    "Gasoline",
    "Diesel",
    "CustomFuel",
    "Steam",
    "ChilledWater",
]
HeatingSystemType = Literal[
    "ElectricResistance", "GasBoiler", "GasFurnace", "ASHP", "GSHP"
]
CoolingSystemType = Literal["DX", "EvapCooling"]
DistributionType = Literal["Hydronic", "Air", "Steam"]


class ThermalSystemComponent(NamedObject, MetadataMixin, extra="forbid"):
    """A thermal system object in the SBEM library."""

    ConditioningType: Literal["Heating", "Cooling", "HeatingAndCooling"]
    Fuel: FuelType
    SystemCOP: float = Field(
        ...,
        title="System COP",
        ge=0,
    )
    DistributionCOP: float = Field(..., title="Distribution COP", ge=0)

    @property
    def effective_system_cop(self) -> float:
        """Compute the effective system COP based on the system and distribution COPs.

        Returns:
            cop (float): The effective system COP.
        """
        return self.SystemCOP * self.DistributionCOP

    @property
    def HeatingSystemType(self) -> HeatingSystemType:
        """Compute the heating system type based on the system COP.

        Returns:
            heating_system_type (HeatingSystemType): The heating system type.
        """
        if (
            self.ConditioningType != "Heating"
            and self.ConditioningType != "HeatingAndCooling"
        ):
            msg = "Heating system type is only applicable to heating systems."
            raise ValueError(msg)
        # TODO: compute based off of CoP
        msg = "Heating system type is not implemented."
        raise NotImplementedError(msg)
        return "ElectricResistance"

    @property
    def CoolingSystemType(self) -> CoolingSystemType:
        """Compute the cooling system type based on the system COP.

        Returns:
            cooling_system_type (CoolingSystemType): The cooling system type.
        """
        if (
            self.ConditioningType != "Cooling"
            and self.ConditioningType != "HeatingAndCooling"
        ):
            msg = "Cooling system type is only applicable to cooling systems."
            raise ValueError(msg)
        # TODO: compute based off of CoP
        msg = "Cooling system type is not implemented."
        raise NotImplementedError(msg)
        return "DX"

    @property
    def DistributionType(self) -> DistributionType:
        """Compute the distribution type based on the system COP.

        Returns:
            distribution_type (DistributionType): The distribution type.
        """
        # TODO: compute based off of CoP
        msg = "Distribution type is not implemented."
        raise NotImplementedError(msg)
        return "Hydronic"


class ConditioningSystemsComponent(NamedObject, MetadataMixin, extra="forbid"):
    """A conditioning system object in the SBEM library."""

    Heating: ThermalSystemComponent | None
    Cooling: ThermalSystemComponent | None

    @model_validator(mode="after")
    def validate_conditioning_types(self):
        """Validate that the conditioning types are correct.

        Cannot have a heating system assigned to a cooling system and vice versa.
        """
        if self.Heating and "heating" not in self.Heating.ConditioningType.lower():
            msg = "Heating system type is only applicable to heating systems."
            raise ValueError(msg)
        if self.Cooling and "cooling" not in self.Cooling.ConditioningType.lower():
            msg = "Cooling system type is only applicable to cooling systems."
            raise ValueError(msg)

        return self


VentilationProvider = Literal["None", "Natural", "Mechanical", "Both"]

EconomizerMethod = Literal[
    "NoEconomizer", "DifferentialDryBulb", "DifferentialEnthalpy"
]

HRVMethod = Literal["NoHRV", "Sensible", "Enthalpy"]

DCVMethod = Literal["NoDCV", "OccupancySchedule", "CO2Setpoint"]


class VentilationComponent(NamedObject, MetadataMixin, extra="forbid"):
    """A ventilation object in the SBEM library."""

    # TODO: add unit notes in field descriptions
    FreshAirPerFloorArea: float = Field(
        ...,
        title="Fresh air per m2 of the object [m³/s/m²]",
        ge=0,
        le=0.05,
    )
    FreshAirPerPerson: float = Field(
        ...,
        title="Fresh air per person of the object [m³/s/p]",
        ge=0,
        le=0.05,
    )
    Schedule: YearComponent = Field(..., title="Ventilation schedule of the object")
    Provider: VentilationProvider = Field(..., title="Type of the object")
    HRV: HRVMethod = Field(..., title="HRV type of the object")
    Economizer: EconomizerMethod = Field(..., title="Economizer type of the object")
    DCV: DCVMethod = Field(..., title="DCV type of the object")
    NaturalVentMinOutdoorTemperature: float | None = Field(
        default=None, title="Natural ventilation: minimum outdoor temperature [C] (unset: 12)"
    )
    NaturalVentMaxOutdoorTemperature: float | None = Field(
        default=None, title="Natural ventilation: maximum outdoor temperature [C] (unset: 25)"
    )
    NaturalVentDeltaTemperature: float | None = Field(
        default=None,
        title="Natural ventilation: open only when indoor exceeds outdoor by this much [C] (unset: 2)",
    )
    NaturalVentAboveSetpointMidpoint: bool | None = Field(
        default=None,
        title="Natural ventilation: open only when indoor is above the mean of the heating and "
        "cooling setpoints (as OpenStudio-HPXML)",
    )

    @model_validator(mode="after")
    def validate_ventilation_systems(self):
        """Validate that the ventilation systems are correct.

        If the ventilation type is natural, then the zone cannot have variants of mechanical ventilation systems (e.g, HRV, DCV, Economizer).
        """
        if self.Provider == "Natural":
            if self.HRV != "NoHRV":
                msg = "Natural ventilation systems can't have HRV."
                raise ValueError(msg)
            if self.DCV != "NoDCV":
                msg = "Natural ventilation systems can't have DCV."
                raise ValueError(msg)
            if self.Economizer != "NoEconomizer":
                msg = "Natural ventilation systems can't have an Economizer."
                raise ValueError(msg)
        if self.Provider == "None":
            if self.HRV != "NoHRV":
                msg = "None ventilation systems can't have HRV."
                raise ValueError(msg)
            if self.DCV != "NoDCV":
                msg = "None ventilation systems can't have DCV."
                raise ValueError(msg)
            if self.Economizer != "NoEconomizer":
                msg = "None ventilation systems can't have an Economizer."
                raise ValueError(msg)
        return self


class ZoneHVACComponent(
    NamedObject,
    MetadataMixin,
    extra="forbid",
):
    """Conditioning object in the SBEM library."""

    ConditioningSystems: ConditioningSystemsComponent
    Ventilation: VentilationComponent
    # TODO: change the structure like ZoneSpaceUse
    """Zone conditioning object."""

    def add_conditioning_to_idf_zone(self, idf: IDF, target_zone_name: str) -> IDF:
        """Add conditioning to an IDF zone.

        This constructs HVAC template objects which get assigned to the zone.

        NB: currently, many of the climate studio parameters are ignored -
        particularly the ones related to humidity control.

        Args:
            idf (IDF): The IDF object to add the conditioning to.
            target_zone_name (str): The name of the zone to add the conditioning to.

        Returns:
            IDF: The updated IDF object.
        """
        # TODO: add the idf conversion functions

        return idf


DHWFuelType = Literal["Electricity", "NaturalGas", "Propane", "FuelOil", "Steam"]


class DHWComponent(
    NamedObject,
    MetadataMixin,
    extra="forbid",
):
    """Domestic Hot Water object."""

    SystemCOP: float = Field(
        ...,
        title="Domestic hot water coefficient of performance",
        ge=0,
    )
    WaterTemperatureInlet: float = Field(
        ...,
        title="Water temperature inlet [°C]",
        ge=0,
        le=100,
    )  # TODO:remove this or just set as constant. Leaving for flexibility here

    DistributionCOP: float = Field(
        ...,
        title="Distribution coefficient of performance",
        ge=0,
        le=1,
    )

    WaterSupplyTemperature: float = Field(
        ...,
        title="Water supply temperature [°C]",
        ge=0,
        le=100,
    )
    IsOn: BoolStr = Field(..., title="Is on")
    FuelType: DHWFuelType = Field(..., title="Hot water fuel type")

    @model_validator(mode="after")
    def validate_supply_greater_than_inlet(self):
        """Validate that the supply temperature is greater than the inlet temperature."""
        if self.WaterSupplyTemperature <= self.WaterTemperatureInlet:
            msg = "Water supply temperature must be greater than the inlet temperature."
            raise ValueError(msg)
        return self

    @property
    def effective_system_cop(self) -> float:
        """Compute the effective system COP based on the system and distribution COPs.

        Returns:
            cop (float): The effective system COP.
        """
        return self.SystemCOP * self.DistributionCOP
