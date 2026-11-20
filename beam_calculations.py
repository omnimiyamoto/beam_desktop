"""Beam statics and deflection calculations, independent of the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal

import numpy as np

SupportType = Literal["simple", "cantilever"]
MaterialType = Literal["steel", "lumber"]


@dataclass(frozen=True)
class PointLoad:
    magnitude_kips: float
    position_ft: float


@dataclass(frozen=True)
class UniformLoad:
    magnitude_kip_per_ft: float
    start_ft: float
    end_ft: float


@dataclass(frozen=True)
class Beam:
    span_ft: float
    support: SupportType
    point_loads: tuple[PointLoad, ...] = ()
    uniform_loads: tuple[UniformLoad, ...] = ()


@dataclass(frozen=True)
class Section:
    name: str
    material: MaterialType
    ix_in4: float
    sx_in3: float
    elastic_modulus_ksi: float
    allowable_bending_ksi: float
    weight_lb_per_ft: float | None = None


@dataclass(frozen=True)
class Reactions:
    left_kips: float
    right_kips: float
    fixed_moment_kip_ft: float


@dataclass(frozen=True)
class ForceAnalysis:
    beam: Beam
    reactions: Reactions
    x_ft: np.ndarray
    shear_kips: np.ndarray
    moment_kip_ft: np.ndarray
    plot_x_ft: np.ndarray
    plot_shear_kips: np.ndarray


STEEL_SECTIONS = (
    Section("W8x10", "steel", 30.8, 7.81, 29000, 30, 10),
    Section("W10x12", "steel", 53.8, 10.9, 29000, 30, 12),
    Section("W12x26", "steel", 204, 33.4, 29000, 30, 26),
    Section("W14x22", "steel", 199, 29.0, 29000, 30, 22),
    Section("W16x26", "steel", 301, 38.4, 29000, 30, 26),
    Section("W18x35", "steel", 510, 57.6, 29000, 30, 35),
    Section("W21x44", "steel", 843, 81.6, 29000, 30, 44),
    Section("W24x55", "steel", 1350, 114, 29000, 30, 55),
)

LUMBER_SECTIONS = (
    Section("2x8", "lumber", 47.63, 13.14, 1600, 0.900),
    Section("2x10", "lumber", 98.93, 21.39, 1600, 0.900),
    Section("2x12", "lumber", 177.98, 31.64, 1600, 0.900),
)

SECTIONS: dict[MaterialType, tuple[Section, ...]] = {
    "steel": STEEL_SECTIONS,
    "lumber": LUMBER_SECTIONS,
}

_GRID_STEP_FT = 0.01


def sections_for(material: MaterialType) -> tuple[Section, ...]:
    """Return sections in the order specified for the selected material."""
    try:
        return SECTIONS[material]
    except KeyError as exc:
        raise ValueError(f"Unknown material: {material}") from exc


def _validate_beam(beam: Beam) -> None:
    if not isfinite(beam.span_ft) or not 4 <= beam.span_ft <= 40:
        raise ValueError("Beam span must be between 4 and 40 ft.")
    if beam.support not in ("simple", "cantilever"):
        raise ValueError(f"Unknown support type: {beam.support}")

    for load in beam.point_loads:
        if not isfinite(load.magnitude_kips) or load.magnitude_kips < 0:
            raise ValueError("Point load magnitudes must be nonnegative and finite.")
        if not isfinite(load.position_ft) or load.position_ft < 0:
            raise ValueError("Point load positions must be nonnegative and finite.")

    for load in beam.uniform_loads:
        values = (load.magnitude_kip_per_ft, load.start_ft, load.end_ft)
        if not all(isfinite(value) for value in values):
            raise ValueError("Uniform load values must be finite.")
        if load.magnitude_kip_per_ft < 0:
            raise ValueError("Uniform load magnitudes must be nonnegative.")
        if load.start_ft < 0 or load.end_ft < load.start_ft:
            raise ValueError("Uniform load positions must satisfy 0 <= start <= end.")


def clip_point_loads(beam: Beam) -> tuple[PointLoad, ...]:
    return tuple(
        PointLoad(load.magnitude_kips, min(load.position_ft, beam.span_ft))
        for load in beam.point_loads
    )


def clip_uniform_loads(beam: Beam) -> tuple[UniformLoad, ...]:
    return tuple(
        UniformLoad(
            load.magnitude_kip_per_ft,
            min(load.start_ft, beam.span_ft),
            min(load.end_ft, beam.span_ft),
        )
        for load in beam.uniform_loads
    )


def calculate_reactions(beam: Beam) -> Reactions:
    """Solve vertical reactions and any cantilever fixed-end moment."""
    _validate_beam(beam)
    span = beam.span_ft
    points = clip_point_loads(beam)
    uniforms = clip_uniform_loads(beam)
    total_load = sum(load.magnitude_kips for load in points)
    total_moment = sum(
        load.magnitude_kips * load.position_ft for load in points
    )

    for load in uniforms:
        length = max(0.0, load.end_ft - load.start_ft)
        resultant = load.magnitude_kip_per_ft * length
        centroid = (load.start_ft + load.end_ft) / 2
        total_load += resultant
        total_moment += resultant * centroid

    if beam.support == "simple":
        right = total_moment / span
        return Reactions(total_load - right, right, 0.0)

    return Reactions(total_load, 0.0, total_moment)


def _moment_at(beam: Beam, reactions: Reactions, x_ft: np.ndarray) -> np.ndarray:
    moment = np.zeros_like(x_ft, dtype=float)
    if beam.support == "simple":
        moment += reactions.left_kips * x_ft
        for load in clip_point_loads(beam):
            moment -= load.magnitude_kips * np.maximum(x_ft - load.position_ft, 0)
        for load in clip_uniform_loads(beam):
            end = np.clip(x_ft, load.start_ft, load.end_ft)
            length = end - load.start_ft
            moment -= load.magnitude_kip_per_ft * (
                x_ft * length - (end**2 - load.start_ft**2) / 2
            )
        return moment

    for load in clip_point_loads(beam):
        moment += load.magnitude_kips * np.maximum(load.position_ft - x_ft, 0)
    for load in clip_uniform_loads(beam):
        start = np.maximum(load.start_ft, x_ft)
        length = np.maximum(load.end_ft - start, 0)
        contribution = (
            (load.end_ft - x_ft) ** 2 - (start - x_ft) ** 2
        ) / 2
        moment += np.where(
            length > 0, load.magnitude_kip_per_ft * contribution, 0
        )
    return moment


def _shear_at(
    beam: Beam,
    reactions: Reactions,
    x_ft: np.ndarray,
    side: Literal["left", "right"] = "right",
) -> np.ndarray:
    shear = np.full_like(x_ft, reactions.left_kips, dtype=float)
    if beam.support == "simple":
        for load in clip_point_loads(beam):
            before_load = x_ft > load.position_ft if side == "left" else x_ft >= load.position_ft
            shear -= load.magnitude_kips * before_load
        for load in clip_uniform_loads(beam):
            length = np.clip(x_ft - load.start_ft, 0, load.end_ft - load.start_ft)
            shear -= load.magnitude_kip_per_ft * length
        if side == "right":
            shear = np.where(x_ft >= beam.span_ft, shear + reactions.right_kips, shear)
        elif side == "left":
            shear = np.where(x_ft <= 0, 0.0, shear)
        return shear

    shear.fill(0.0)
    for load in clip_point_loads(beam):
        beyond_load = x_ft <= load.position_ft if side == "left" else x_ft < load.position_ft
        shear += load.magnitude_kips * beyond_load
    for load in clip_uniform_loads(beam):
        start = np.maximum(load.start_ft, x_ft)
        shear += load.magnitude_kip_per_ft * np.maximum(load.end_ft - start, 0)
    return shear


def calculate_forces(beam: Beam) -> ForceAnalysis:
    """Calculate reactions, shear, and moment on a 0.01 ft grid."""
    _validate_beam(beam)
    reactions = calculate_reactions(beam)
    count = max(1, ceil(beam.span_ft / _GRID_STEP_FT))
    base_grid = np.linspace(0.0, beam.span_ft, count + 1)
    breaks = [0.0, beam.span_ft]
    breaks.extend(load.position_ft for load in clip_point_loads(beam))
    for load in clip_uniform_loads(beam):
        breaks.extend((load.start_ft, load.end_ft))
    x_ft = np.unique(np.concatenate((base_grid, np.asarray(breaks, dtype=float))))
    shear = _shear_at(beam, reactions, x_ft)
    moment = _moment_at(beam, reactions, x_ft)

    plot_x: list[float] = []
    plot_shear: list[float] = []
    point_positions = {load.position_ft for load in clip_point_loads(beam)}
    for x, value in zip(x_ft, shear):
        if x == 0 or x == beam.span_ft or x in point_positions:
            left_value = float(_shear_at(beam, reactions, np.array([x]), "left")[0])
            right_value = float(_shear_at(beam, reactions, np.array([x]), "right")[0])
            if left_value != right_value:
                plot_x.extend((float(x), float(x)))
                plot_shear.extend((left_value, right_value))
                continue
        plot_x.append(float(x))
        plot_shear.append(float(value))

    return ForceAnalysis(
        beam=beam,
        reactions=reactions,
        x_ft=x_ft,
        shear_kips=shear,
        moment_kip_ft=moment,
        plot_x_ft=np.asarray(plot_x),
        plot_shear_kips=np.asarray(plot_shear),
    )


def calculate_deflection(forces: ForceAnalysis, section: Section) -> np.ndarray:
    """Integrate M/EI twice, applying the beam's support boundary conditions."""
    x_inches = forces.x_ft * 12
    curvature_per_in = forces.moment_kip_ft * 12 / (
        section.elastic_modulus_ksi * section.ix_in4
    )
    dx_inches = np.diff(x_inches)

    slope = np.zeros_like(x_inches)
    slope[1:] = np.cumsum(
        (curvature_per_in[:-1] + curvature_per_in[1:]) * 0.5 * dx_inches
    )
    deflection = np.zeros_like(x_inches)
    deflection[1:] = np.cumsum((slope[:-1] + slope[1:]) * 0.5 * dx_inches)

    if forces.beam.support == "simple":
        end_deflection = deflection[-1]
        deflection -= end_deflection * forces.x_ft / forces.beam.span_ft
    return deflection


def response_at(
    forces: ForceAnalysis, deflection_in: np.ndarray, x_ft: float
) -> tuple[float, float, float]:
    """Return shear, moment, and deflection at a cursor position."""
    x = float(np.clip(x_ft, 0, forces.beam.span_ft))
    x_array = np.array([x])
    shear = float(_shear_at(forces.beam, forces.reactions, x_array)[0])
    moment = float(_moment_at(forces.beam, forces.reactions, x_array)[0])
    deflection = float(np.interp(x, forces.x_ft, deflection_in))
    return shear, moment, deflection


def recommend_section(
    forces: ForceAnalysis,
    material: MaterialType,
    deflection_limit_divisor: int,
) -> Section | None:
    """Return the lightest passing steel or first passing lumber section."""
    if deflection_limit_divisor not in (240, 360):
        raise ValueError("Deflection limit divisor must be 240 or 360.")

    max_moment = float(np.max(np.abs(forces.moment_kip_ft)))
    limit_in = forces.beam.span_ft * 12 / deflection_limit_divisor
    candidates: list[Section] = []
    for section in sections_for(material):
        bending = max_moment * 12 / section.sx_in3
        deflection = calculate_deflection(forces, section)
        max_deflection = float(np.max(np.abs(deflection)))
        if (
            bending <= section.allowable_bending_ksi
            and max_deflection <= limit_in
        ):
            candidates.append(section)

    if material == "steel":
        return min(
            candidates,
            key=lambda section: (section.weight_lb_per_ft or float("inf"), -section.ix_in4),
            default=None,
        )
    return candidates[0] if candidates else None
