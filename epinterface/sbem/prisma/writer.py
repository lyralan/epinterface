"""Write SBEM components into a component database.

The reverse of the deep fetchers in `client.py`: given a validated component tree (e.g. a
ZoneComponent, an EnvelopeAssemblyComponent or a YearComponent), insert it and every child it
references, connecting children by Name. A component whose Name is already in the database is
left as it is and reused, so shared children (materials, schedules, assemblies) are written once.

Field names on the components match the Prisma models, so the mapping is generic: scalar fields
are copied, component fields become `connect` relations, and an assembly's layers are created in
place. Metadata (DataSource, Comment) is written to the Metadata table; it is not read back.
"""

from typing import Any

from prisma import Prisma
from pydantic import BaseModel

from epinterface.sbem.common import NamedObject
from epinterface.sbem.components.envelope import (
    ConstructionLayerComponent,
    ConstructionMaterialComponent,
)
from epinterface.sbem.components.schedules import DayComponent, WeekComponent
from epinterface.sbem.prisma.client import deep_fetcher

METADATA_FIELDS = ("Category", "Comment", "DataSource", "Version")
ENVIRONMENTAL_FIELDS = ("Cost", "RateUnit", "Life", "EmbodiedCarbon")


def _model_name(component: NamedObject) -> str:
    """The Prisma model a component is stored in."""
    if isinstance(component, DayComponent):
        return "Day"
    if isinstance(component, WeekComponent):
        return "Week"
    if isinstance(component, ConstructionMaterialComponent):
        return "ConstructionMaterial"
    for key, link_type in type(deep_fetcher).__annotations__.items():
        if hasattr(link_type, "__args__") and link_type.__args__[-1] is type(component):
            return key
    msg = f"No database table for {type(component).__name__}."
    raise TypeError(msg)


def _table(db: Prisma, model_name: str):
    return getattr(db, model_name.lower())


def _model_fields(model_name: str) -> set[str]:
    import prisma.models

    return set(getattr(prisma.models, model_name).model_fields)


def write_component(db: Prisma, component: NamedObject) -> str:
    """Insert a component and its children; return its Name.

    Args:
        db (Prisma): A connected Prisma client.
        component (NamedObject): The component to write.

    Returns:
        name (str): The component's Name, for connecting it elsewhere.
    """
    model_name = _model_name(component)
    table = _table(db, model_name)
    if table.find_unique(where={"Name": component.Name}) is not None:
        return component.Name

    fields = _model_fields(model_name)
    data: dict[str, Any] = {}
    for key in type(component).model_fields:
        value = getattr(component, key)
        if value is None or key in METADATA_FIELDS or key in ENVIRONMENTAL_FIELDS:
            continue
        if key not in fields:
            continue
        if isinstance(value, NamedObject):
            data[key] = {"connect": {"Name": write_component(db, value)}}
        elif isinstance(value, list) and value and isinstance(value[0], ConstructionLayerComponent):
            data[key] = {"create": [_layer(db, layer) for layer in value]}
        elif isinstance(value, BaseModel):
            msg = f"{model_name}.{key} is a nested model without a Name; cannot write it."
            raise TypeError(msg)
        else:
            data[key] = value

    meta = {
        "source": getattr(component, "DataSource", None),
        "comments": getattr(component, "Comment", None),
    }
    if "Metadata" in fields and any(meta.values()):
        data["Metadata"] = {"create": {k: v for k, v in meta.items() if v}}
    env = {k: getattr(component, k, None) for k in ENVIRONMENTAL_FIELDS}
    if "EnvironmentalData" in fields and any(v is not None for v in env.values()):
        data["EnvironmentalData"] = {"create": {k: v for k, v in env.items() if v is not None}}

    table.create(data=data)
    return component.Name


def _layer(db: Prisma, layer: ConstructionLayerComponent) -> dict[str, Any]:
    return {
        "ConstructionMaterial": {"connect": {"Name": write_component(db, layer.ConstructionMaterial)}},
        "Thickness": layer.Thickness,
        "LayerOrder": layer.LayerOrder,
    }
