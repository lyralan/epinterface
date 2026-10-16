"""Operations components for the SBEM library."""

from logging import getLogger

from archetypal.idfclass import IDF
from archetypal.schedule import Schedule, ScheduleTypeLimits

from epinterface.constants import assumed_constants
from epinterface.geometry import get_zone_floor_area, get_zone_glazed_area
from epinterface.interface import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
    ScheduleDayHourly,
    WaterUseEquipment,
    ZoneVentilationWindAndStackOpenArea,
)
from epinterface.sbem.common import MetadataMixin, NamedObject
from epinterface.sbem.components.space_use import ZoneSpaceUseComponent
from epinterface.sbem.components.systems import DHWComponent, ZoneHVACComponent

logger = getLogger(__name__)


class ZoneOperationsComponent(
    NamedObject,
    MetadataMixin,
    extra="forbid",
):
    """Zone use consolidation across space use, HVAC, DHW."""

    SpaceUse: ZoneSpaceUseComponent
    HVAC: ZoneHVACComponent
    DHW: DHWComponent

    def add_water_use_to_idf_zone(self, idf: IDF, target_zone_name: str) -> IDF:
        """Handle adding water use to the zone based on both DHW and Operations.SpaceUse.WaterUse.

        We choose to execute this from the zone component interface because it requires context from both
        zone.Operations.SpaceUse.WaterUse and zone.Operations.

        Note: the water use component's flow rate represents the total amount of water used per person per day;
        in order to calculate a peak flow rate, we will need to ensure that the product of the schedule and the peak
        flow rate would resolve to the same daily average value per person.

        This will be responsible for:
            1. Extracting the total area of the zone by finding the matching surfaces.
            2. Computing the total number of people in the zone based on the occupancy density and the total area.
            3. Computing the average flow rate of water use/s in the zone based on the flow rate per person/day and the total number of people.
            4. Computing the peak flow rate of water use/s in the zone based on the average flow rate and the average fractional value of the schedule.
            5. Adding the water use equipment to the zone.


        # TODO: major problem - this assumes the zone has already been correctly sized!
        # this method should only actually be called AFTER
        # the building has been properly rescaled, which currently is executed at the very end.
        # that hook should probably be moved up.

        Args:
            idf (IDF): The IDF object to add the operations to.
            target_zone_name (str): The name of the zone to add the operations to.

        Returns:
            idf (IDF): The updated IDF object.
        """
        # TODO: should IsOn definitely live in DHW? should dhw be nullable?
        # should it live in spaceuse.wateruse for consistency?
        if not self.DHW.IsOn:
            return idf

        # Acquire the relevant data fields
        flow_rate_per_person_per_day = (
            self.SpaceUse.WaterUse.FlowRatePerPerson
        )  # m3/person/day
        occupant_density = self.SpaceUse.Occupancy.PeopleDensity  # ppl/m2
        water_supply_temperature = self.DHW.WaterSupplyTemperature  # degC
        water_temperature_inlet = self.DHW.WaterTemperatureInlet  # degC
        water_use_frac_sched = self.SpaceUse.WaterUse.Schedule
        water_use_name = f"{target_zone_name}_{self.SpaceUse.WaterUse.safe_name}_{self.DHW.safe_name}_WATER"

        # determine zone area to then compute total people
        # we will do this by finding all matching surfaces and computing their areas.
        zone = next(
            (x for x in idf.idfobjects["ZONE"] if x.Name == target_zone_name), None
        )
        if zone is None:
            raise ValueError(f"NO_ZONE:{target_zone_name}")
        area = get_zone_floor_area(idf, zone.Name)
        total_ppl = occupant_density * area

        # Compute final flow rates.
        total_flow_per_day = flow_rate_per_person_per_day * total_ppl  # m3/day
        avg_flow_per_s = total_flow_per_day / (3600 * 24)  # m3/s
        # TODO: Update this rather than being constant rate

        lim = "Temperature"
        if not idf.getobject("SCHEDULETYPELIMITS", lim):
            lim = ScheduleTypeLimits(
                Name="Temperature",
                LowerLimit=-60,
                UpperLimit=200,
            )
            lim.to_epbunch(idf)

        # TODO: should we be using time-varying temperatures?
        target_temperature_schedule = Schedule.constant_schedule(
            value=water_supply_temperature,  # pyright: ignore [reportArgumentType]
            Name=f"{water_use_name}_TargetWaterTemperatureSch",
            Type="Temperature",
        )
        inlet_temperature_schedule = Schedule.constant_schedule(
            value=water_temperature_inlet,  # pyright: ignore [reportArgumentType]
            Name=f"{water_use_name}_InletWaterTemperatureSch",
            Type="Temperature",
        )

        target_temperature_yr_schedule, *_ = (
            target_temperature_schedule.to_year_week_day()
        )
        inlet_temperature_yr_schedule, *_ = (
            inlet_temperature_schedule.to_year_week_day()
        )

        target_temperature_yr_schedule.to_epbunch(idf)
        inlet_temperature_yr_schedule.to_epbunch(idf)

        if water_use_frac_sched.schedule_type_limits != "Fraction":
            msg = f"WATER_USE_FRACTION_SCHEDULE_TYPE_NOT_FRACTION:{water_use_name}"
            raise ValueError(msg)

        idf, water_use_frac_sch_name = water_use_frac_sched.add_year_to_idf(
            idf, name_prefix=None
        )
        # sch_obj = idf.getobject("SCHEDULE:YEAR", water_use_frac_sch_name)
        # arch_sch = Schedule.from_epbunch(sch_obj)
        # values = np.array(arch_sch.Values)
        # avg_fractional_value = np.sum(values) / 8760
        logger.warning("USING HARDCODED 2018 to compute avg fractional value")
        avg_fractional_value = water_use_frac_sched.fractional_year_sum(2018)
        # total_fractional_value * peak_flow_rate_per_s = avg_float_per_s
        peak_flow_rate_per_s = avg_flow_per_s / avg_fractional_value

        hot_water = WaterUseEquipment(
            Name=water_use_name,
            EndUse_Subcategory="Domestic Hot Water",
            Peak_Flow_Rate=peak_flow_rate_per_s,
            Flow_Rate_Fraction_Schedule_Name=water_use_frac_sch_name,
            Zone_Name=target_zone_name,
            Target_Temperature_Schedule_Name=target_temperature_yr_schedule.Name,
            Hot_Water_Supply_Temperature_Schedule_Name=target_temperature_schedule.Name,
            Cold_Water_Supply_Temperature_Schedule_Name=inlet_temperature_schedule.Name,
            Sensible_Fraction_Schedule_Name=None,
            Latent_Fraction_Schedule_Name=None,
        )
        idf = hot_water.add(idf)
        return idf

    def add_thermostat_to_idf_zone(
        self, idf: IDF, target_zone_name: str
    ) -> HVACTemplateThermostat:
        """Add a thermostat to the zone.

        Args:
            idf (IDF): The IDF object to add the thermostat to.
            target_zone_name (str): The name of the zone to add the thermostat to.

        Returns:
            idf (IDF): The updated IDF object.
        """
        thermostat_name = (
            f"{target_zone_name}_{self.SpaceUse.Thermostat.safe_name}_THERMOSTAT"
        )
        heating_schedule = self.SpaceUse.Thermostat.HeatingSchedule
        cooling_schedule = self.SpaceUse.Thermostat.CoolingSchedule

        heating_schedule_name = None
        cooling_schedule_name = None
        heat_high = cool_low = None
        if heating_schedule is not None:
            _, heat_high = heating_schedule.bounds
        if cooling_schedule is not None:
            cool_low, _ = cooling_schedule.bounds
        if heat_high is not None and cool_low is not None and heat_high > cool_low:
            # the year's highest heating and lowest cooling setpoints can cross even when the
            # hourly schedules never do; E+ rejects crossed design-day setpoints, so use the mean
            heat_high = cool_low = (heat_high + cool_low) / 2
        if heating_schedule is not None:
            heating_design_day = ScheduleDayHourly(
                Name=f"{thermostat_name}_HeatingDesignDay",
                Schedule_Type_Limits_Name="Temperature",
                Hour_1=heat_high,
                Hour_2=heat_high,
                Hour_3=heat_high,
                Hour_4=heat_high,
                Hour_5=heat_high,
                Hour_6=heat_high,
                Hour_7=heat_high,
                Hour_8=heat_high,
                Hour_9=heat_high,
                Hour_10=heat_high,
                Hour_11=heat_high,
                Hour_12=heat_high,
                Hour_13=heat_high,
                Hour_14=heat_high,
                Hour_15=heat_high,
                Hour_16=heat_high,
                Hour_17=heat_high,
                Hour_18=heat_high,
                Hour_19=heat_high,
                Hour_20=heat_high,
                Hour_21=heat_high,
                Hour_22=heat_high,
                Hour_23=heat_high,
                Hour_24=heat_high,
            )
            idf = heating_design_day.add(idf)
            idf, heating_schedule_name = heating_schedule.add_year_to_idf(
                idf,
                name_prefix=None,
                winter_design_day_sch_name=heating_design_day.Name,
                summer_design_day_sch_name=heating_design_day.Name,
            )
        if cooling_schedule is not None:
            cooling_design_day = ScheduleDayHourly(
                Name=f"{thermostat_name}_CoolingDesignDay",
                Schedule_Type_Limits_Name="Temperature",
                Hour_1=cool_low,
                Hour_2=cool_low,
                Hour_3=cool_low,
                Hour_4=cool_low,
                Hour_5=cool_low,
                Hour_6=cool_low,
                Hour_7=cool_low,
                Hour_8=cool_low,
                Hour_9=cool_low,
                Hour_10=cool_low,
                Hour_11=cool_low,
                Hour_12=cool_low,
                Hour_13=cool_low,
                Hour_14=cool_low,
                Hour_15=cool_low,
                Hour_16=cool_low,
                Hour_17=cool_low,
                Hour_18=cool_low,
                Hour_19=cool_low,
                Hour_20=cool_low,
                Hour_21=cool_low,
                Hour_22=cool_low,
                Hour_23=cool_low,
                Hour_24=cool_low,
            )

            idf = cooling_design_day.add(idf)
            idf, cooling_schedule_name = cooling_schedule.add_year_to_idf(
                idf,
                name_prefix=None,
                winter_design_day_sch_name=cooling_design_day.Name,
                summer_design_day_sch_name=cooling_design_day.Name,
            )

        thermostat = HVACTemplateThermostat(
            Name=thermostat_name,
            Heating_Setpoint_Schedule_Name=heating_schedule_name,
            Constant_Heating_Setpoint=self.SpaceUse.Thermostat.HeatingSetpoint
            if self.SpaceUse.Thermostat.HeatingSchedule is None
            else None,
            Cooling_Setpoint_Schedule_Name=cooling_schedule_name,
            Constant_Cooling_Setpoint=self.SpaceUse.Thermostat.CoolingSetpoint
            if self.SpaceUse.Thermostat.CoolingSchedule is None
            else None,
        )

        idf = thermostat.add(idf)

        return thermostat

    def add_conditioning_to_idf_zone(self, idf: IDF, target_zone_name: str) -> IDF:
        """Add conditioning to an IDF zone."""
        thermostat = self.add_thermostat_to_idf_zone(idf, target_zone_name)
        if self.HVAC.Ventilation.DCV != "NoDCV":
            # check the design spec outdoor air for the DCV
            raise NotImplementedError("DCV not implemented.")
        hvac_template = HVACTemplateZoneIdealLoadsAirSystem(
            Zone_Name=target_zone_name,
            Template_Thermostat_Name=thermostat.Name,
            Heating_Availability_Schedule_Name=None
            if self.HVAC.ConditioningSystems.Heating
            else "AlwaysOff",
            Cooling_Availability_Schedule_Name=None
            if self.HVAC.ConditioningSystems.Cooling
            else "AlwaysOff",
            Heating_Limit="LimitFlowRateAndCapacity",
            Maximum_Heating_Air_Flow_Rate="autosize",
            Maximum_Sensible_Heating_Capacity="autosize",
            Cooling_Limit="LimitFlowRateAndCapacity",
            Maximum_Cooling_Air_Flow_Rate="autosize",
            Maximum_Total_Cooling_Capacity="autosize",
            Minimum_Cooling_Supply_Air_Temperature=13,
            Humidification_Control_Type="None",
            Outdoor_Air_Flow_Rate_per_Person=self.HVAC.Ventilation.FreshAirPerPerson,
            Outdoor_Air_Flow_Rate_per_Zone_Floor_Area=self.HVAC.Ventilation.FreshAirPerFloorArea,
            Outdoor_Air_Flow_Rate_per_Zone=0,
            Demand_Controlled_Ventilation_Type="None"
            if self.HVAC.Ventilation.DCV == "NoDCV"
            else self.HVAC.Ventilation.DCV,
            Outdoor_Air_Economizer_Type=self.HVAC.Ventilation.Economizer,
            Heat_Recovery_Type="None"
            if self.HVAC.Ventilation.HRV == "NoHRV"
            else self.HVAC.Ventilation.HRV,
            Sensible_Heat_Recovery_Effectiveness=assumed_constants.Sensible_Heat_Recovery_Effectiveness,
            Latent_Heat_Recovery_Effectiveness=assumed_constants.Latent_Heat_Recovery_Effectiveness,
            Outdoor_Air_Method="Sum"
            if self.HVAC.Ventilation.Provider == "Mechanical"
            or self.HVAC.Ventilation.Provider == "Both"
            else "None",
        )
        idf = hvac_template.add(idf)

        if self.HVAC.Ventilation.Provider == "Natural":
            # total_window_area = calculate_window_area_for_zone(idf, target_zone_name)
            total_window_area = get_zone_glazed_area(idf, target_zone_name)

            if total_window_area == 0:
                logger.warning(
                    f"No windows found for natural ventilation in zone {target_zone_name}"
                )
                return idf
            vent_wind_stack_name = f"{target_zone_name}_{self.SpaceUse.Thermostat.safe_name}_{self.HVAC.Ventilation.safe_name}_VENTILATION_WIND_AND_STACK_OPEN_AREA"

            idf, vent_wind_stack_name_sch = (
                self.HVAC.Ventilation.Schedule.add_year_to_idf(
                    idf,
                    name_prefix=None,
                    summer_design_day_sch_name="d_AllOff_00",
                    winter_design_day_sch_name="d_AllOff_00",
                )
            )
            vent = self.HVAC.Ventilation
            min_out = vent.NaturalVentMinOutdoorTemperature
            max_out = vent.NaturalVentMaxOutdoorTemperature
            delta = vent.NaturalVentDeltaTemperature
            thermostat = self.SpaceUse.Thermostat
            ventilation_wind_and_stack_open_area = ZoneVentilationWindAndStackOpenArea(
                Name=vent_wind_stack_name,
                Zone_or_Space_Name=target_zone_name,
                Minimum_Outdoor_Temperature=12 if min_out is None else min_out,
                Height_Difference=0,
                Delta_Temperature=2 if delta is None else delta,
                Maximum_Outdoor_Temperature=25 if max_out is None else max_out,
                Minimum_Indoor_Temperature=(
                    (thermostat.HeatingSetpoint + thermostat.CoolingSetpoint) / 2
                    if vent.NaturalVentAboveSetpointMidpoint
                    else None
                ),
                Opening_Area=total_window_area,
                Opening_Area_Fraction_Schedule_Name=vent_wind_stack_name_sch,
            )
            idf = ventilation_wind_and_stack_open_area.add(idf)

        return idf
