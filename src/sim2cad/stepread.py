"""Read a STEP assembly back, keeping names, nesting and placements.

``cadquery.importers.importStep`` flattens everything into one compound, which
is no use for checking that the hierarchy survived the round trip.  XCAF keeps
the structure, so we go through it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool

from .frames import Pose

__all__ = [
    "StepNode",
    "read_step_assembly",
    "shape_bbox",
    "surface_centroid",
    "solid_volume",
    "count_faces",
]


@dataclass
class StepNode:
    name: str
    loc: Pose
    children: list["StepNode"] = field(default_factory=list)
    shape: TopoDS_Shape | None = None

    def walk(self, parent: Pose | None = None):
        """Yield ``(node, world_pose)`` for this node and everything under it."""
        world = self.loc if parent is None else parent @ self.loc
        yield self, world
        for child in self.children:
            yield from child.walk(world)

    def leaves(self, parent: Pose | None = None):
        for node, world in self.walk(parent):
            if node.shape is not None:
                yield node, world

    def names(self) -> list[str]:
        return [node.name for node, _ in self.walk()]


def _pose_from_toploc(loc: TopLoc_Location) -> Pose:
    trsf = loc.Transformation()
    rot = np.array([[trsf.Value(i, j) for j in range(1, 4)] for i in range(1, 4)], dtype=float)
    pos = np.array([trsf.Value(i, 4) for i in range(1, 4)], dtype=float)
    return Pose(pos, rot)


def _label_name(label: TDF_Label) -> str:
    from OCP.TDataStd import TDataStd_Name

    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def read_step_assembly(path: str) -> StepNode:
    """Load a STEP file into a :class:`StepNode` tree."""
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    status = reader.ReadFile(str(path))
    if int(status) != 1:  # IFSelect_RetDone
        raise RuntimeError(f"could not read STEP file {path} (status {status})")

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("sim2cad"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    if not reader.Transfer(doc):
        raise RuntimeError(f"STEP transfer failed for {path}")

    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    def build(label: TDF_Label, instance_name: str = "", loc: Pose | None = None) -> StepNode:
        # Sub-assembly instance labels get auto-generated numeric names on the
        # way through STEP; the name we wrote survives on the referred
        # (prototype) label, so that one wins.
        name = _label_name(label) or instance_name
        here = loc or Pose.identity()

        if tool.IsAssembly_s(label):
            comps = TDF_LabelSequence()
            tool.GetComponents_s(label, comps)
            children = []
            for i in range(1, comps.Length() + 1):
                comp = comps.Value(i)
                comp_loc = _pose_from_toploc(tool.GetLocation_s(comp))
                comp_name = _label_name(comp)
                ref = TDF_Label()
                if tool.GetReferredShape_s(comp, ref):
                    children.append(build(ref, comp_name, comp_loc))
                else:
                    children.append(build(comp, comp_name, comp_loc))
            return StepNode(name=name, loc=here, children=children)

        shape = tool.GetShape_s(label)
        return StepNode(name=name, loc=here, shape=shape)

    free = TDF_LabelSequence()
    tool.GetFreeShapes(free)
    if free.Length() == 1:
        return build(free.Value(1))

    root = StepNode(name="root", loc=Pose.identity())
    for i in range(1, free.Length() + 1):
        root.children.append(build(free.Value(i)))
    return root


# -- measurements ----------------------------------------------------------


def shape_bbox(shape: TopoDS_Shape, optimal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Tight axis-aligned bounds of a shape.

    ``AddOptimal`` is slower but exact for analytic surfaces -- the ordinary
    ``Add`` inflates the box around curved faces, which would swamp the
    machine-precision tolerance the primitives are held to.
    """
    box = Bnd_Box()
    if optimal:
        BRepBndLib.AddOptimal_s(shape, box, True, False)
    else:
        BRepBndLib.Add_s(shape, box, True)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])


def surface_centroid(shape: TopoDS_Shape) -> tuple[np.ndarray, float]:
    """Area-weighted centroid of a shape's surface, and its total area.

    Surface rather than volume, so the same measure works for the open shells
    that come out of non-watertight meshes.
    """
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    com = props.CentreOfMass()
    return np.array([com.X(), com.Y(), com.Z()]), float(props.Mass())


def solid_volume(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def count_faces(shape: TopoDS_Shape) -> int:
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    n = 0
    while explorer.More():
        n += 1
        explorer.Next()
    return n
