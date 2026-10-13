"""Window shading by interior coverings and eaves.

Interior coverings (blinds, curtains) are applied as OpenStudio-HPXML (ResStock) does: a seasonal
multiplier on the solar radiation incident on each window
(SurfaceProperty:IncidentSolarMultiplier), e.g. 0.85 in winter and 0.70 in summer. Eaves are a
horizontal overhang above each window. Neighboring buildings are not handled here; they come in
as shading blocks from the GIS context (see epinterface.geometry).
"""

import numpy as np
from archetypal.idfclass import IDF
from pydantic import BaseModel, Field, model_validator

INTERIOR_SHADING_SCHEDULE = "InteriorShadingMultiplier"


class WindowShading(BaseModel, extra="forbid"):
    """Interior shading multipliers by season and an optional eave over each window."""

    InteriorShadingWinter: float = Field(default=1.0, ge=0, le=1)
    InteriorShadingSummer: float = Field(default=1.0, ge=0, le=1)
    SummerStart: tuple[int, int] = Field(default=(5, 1), title="First day of summer (month, day)")
    SummerEnd: tuple[int, int] = Field(default=(9, 30), title="Last day of summer (month, day)")
    EavesDepth: float = Field(default=0, ge=0, title="Eave projection from the wall [m]")
    EavesOffset: float = Field(default=0.3048, ge=0, title="Height of the eave above the window top [m]")
    EavesExtension: float = Field(
        default=0.3048, ge=0, title="Eave extension past each side of the window [m]"
    )

    @model_validator(mode="after")
    def check_season(self):
        """Summer must start before it ends within the calendar year."""
        if self.SummerStart >= self.SummerEnd:
            msg = f"SummerStart {self.SummerStart} must be before SummerEnd {self.SummerEnd}."
            raise ValueError(msg)
        return self

    def _schedule(self, idf: IDF) -> str:
        sm, sd = self.SummerStart
        em, ed = self.SummerEnd
        day_before = np.datetime64(f"2001-{sm:02d}-{sd:02d}") - np.timedelta64(1, "D")
        bm, bd = int(str(day_before)[5:7]), int(str(day_before)[8:10])
        if idf.getobject("SCHEDULETYPELIMITS", "Fraction") is None:
            idf.newidfobject(
                "SCHEDULETYPELIMITS",
                Name="Fraction",
                Lower_Limit_Value=0,
                Upper_Limit_Value=1,
                Numeric_Type="Continuous",
            )
        fields = []
        for through, value in (
            (f"{bm}/{bd}", self.InteriorShadingWinter),
            (f"{em}/{ed}", self.InteriorShadingSummer),
            ("12/31", self.InteriorShadingWinter),
        ):
            fields += [f"Through: {through}", "For: AllDays", "Until: 24:00", value]
        sched = idf.newidfobject("SCHEDULE:COMPACT", Name=INTERIOR_SHADING_SCHEDULE)
        sched.obj = [*sched.obj[:2], "Fraction", *fields]
        return INTERIOR_SHADING_SCHEDULE

    def add(self, idf: IDF) -> IDF:
        """Add the multipliers and eaves to every exterior window.

        Args:
            idf (IDF): The model, after geometry and windows are added.

        Returns:
            idf (IDF): The model with shading added.
        """
        windows = [
            w
            for w in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]
            if w.Surface_Type.lower() == "window"
        ]
        if not windows:
            return idf
        if self.InteriorShadingWinter < 1 or self.InteriorShadingSummer < 1:
            sched = self._schedule(idf)
            for w in windows:
                idf.newidfobject(
                    "SURFACEPROPERTY:INCIDENTSOLARMULTIPLIER",
                    Surface_Name=w.Name,
                    Incident_Solar_Multiplier=1,
                    Incident_Solar_Multiplier_Schedule_Name=sched,
                )
        if self.EavesDepth > 0:
            for w in windows:
                pts = np.array(w.coords, dtype=float)
                normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                normal = normal / np.linalg.norm(normal)
                top = pts[:, 2].max()
                p1, p2 = pts[np.isclose(pts[:, 2], top)][:2]
                along = (p2 - p1) / np.linalg.norm(p2 - p1)
                a = p1 - along * self.EavesExtension
                b = p2 + along * self.EavesExtension
                a[2] = b[2] = top + self.EavesOffset
                out = normal * self.EavesDepth
                eave = idf.newidfobject(
                    "SHADING:ZONE:DETAILED",
                    Name=f"{w.Name} Eave",
                    Base_Surface_Name=w.Building_Surface_Name,
                )
                eave.obj = [
                    *eave.obj[:3],
                    "",
                    4,
                    *(round(float(x), 4) for p in (a, b, b + out, a + out) for x in p),
                ]
        return idf
