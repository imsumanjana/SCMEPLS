from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binding import SceneBindingRegistry
from .physics import DigitalTwinFrame, SimulationTimeline


@dataclass(frozen=True)
class AnimationSnapshot:
    frame: DigitalTwinFrame
    transforms: dict[str, np.ndarray]
    progress: float


class PlaybackController:
    """Deterministic playback state independent of Qt and the renderer."""

    def __init__(self, timeline: SimulationTimeline, *, speed: float = 1.0, loop: bool = False) -> None:
        if speed <= 0 or not np.isfinite(speed):
            raise ValueError("Playback speed must be finite and positive.")
        self.timeline = timeline
        self.speed = float(speed)
        self.loop = bool(loop)
        self.playing = False
        self.current_time_s = float(timeline.time_s[0])

    @property
    def minimum_time_s(self) -> float:
        return float(self.timeline.time_s[0])

    @property
    def maximum_time_s(self) -> float:
        return float(self.timeline.time_s[-1])

    @property
    def progress(self) -> float:
        span = self.maximum_time_s - self.minimum_time_s
        if span <= 0:
            return 0.0
        return float((self.current_time_s - self.minimum_time_s) / span)

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def reset(self) -> None:
        self.playing = False
        self.current_time_s = self.minimum_time_s

    def set_speed(self, speed: float) -> None:
        if speed <= 0 or not np.isfinite(speed):
            raise ValueError("Playback speed must be finite and positive.")
        self.speed = float(speed)

    def seek_time(self, time_s: float) -> DigitalTwinFrame:
        self.current_time_s = float(np.clip(time_s, self.minimum_time_s, self.maximum_time_s))
        return self.timeline.frame_at_time(self.current_time_s)

    def seek_fraction(self, fraction: float) -> DigitalTwinFrame:
        f = float(np.clip(fraction, 0.0, 1.0))
        return self.seek_time(self.minimum_time_s + f * (self.maximum_time_s - self.minimum_time_s))

    def advance(self, wall_elapsed_s: float) -> DigitalTwinFrame:
        if wall_elapsed_s < 0 or not np.isfinite(wall_elapsed_s):
            raise ValueError("Elapsed playback time must be finite and non-negative.")
        if self.playing:
            next_time = self.current_time_s + float(wall_elapsed_s) * self.speed
            if next_time > self.maximum_time_s:
                if self.loop:
                    span = self.maximum_time_s - self.minimum_time_s
                    next_time = self.minimum_time_s if span <= 0 else self.minimum_time_s + ((next_time - self.minimum_time_s) % span)
                else:
                    next_time = self.maximum_time_s
                    self.playing = False
            self.current_time_s = next_time
        return self.timeline.frame_at_time(self.current_time_s)


def component_transforms(
    registry: SceneBindingRegistry,
    timeline: SimulationTimeline,
    frame: DigitalTwinFrame,
) -> dict[str, np.ndarray]:
    """Build actor transforms for the frame without mutating geometry or physics."""
    transforms: dict[str, np.ndarray] = {}
    for component_id, binding in registry.bindings.items():
        if binding.dynamic_group == "world":
            transforms[component_id] = np.eye(4, dtype=float)
        else:
            transforms[component_id] = timeline.relative_transform(frame, binding.pivot_m)
    return transforms


def snapshot(
    registry: SceneBindingRegistry,
    controller: PlaybackController,
) -> AnimationSnapshot:
    frame = controller.timeline.frame_at_time(controller.current_time_s)
    return AnimationSnapshot(
        frame=frame,
        transforms=component_transforms(registry, controller.timeline, frame),
        progress=controller.progress,
    )
