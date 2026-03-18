from __future__ import annotations

from dataclasses import dataclass, field

from traphill.engine.types import (
    CrossingEvent,
    Detection,
    TripwireConfig,
    VehicleEvent,
)


def compute_speed(
    crossing_a: CrossingEvent,
    crossing_b: CrossingEvent,
    distance_m: float,
) -> float:
    """Compute speed in km/h from two tripwire crossings."""
    elapsed_s = abs(crossing_b.timestamp_s - crossing_a.timestamp_s)
    if elapsed_s == 0:
        raise ValueError("zero elapsed time between crossings")
    return (distance_m / elapsed_s) * 3.6


@dataclass
class _TrackState:
    vehicle_class: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_ts: float
    last_seen_ts: float
    last_centroid_x: int | None = None
    crossings: list[CrossingEvent] = field(default_factory=list)


class SpeedEstimator:
    def __init__(self, config: TripwireConfig, fps: float, stale_timeout_s: float = 2.0):
        self._config = config
        self._fps = fps
        self._stale_frames = max(1, int(fps * stale_timeout_s))
        self._active: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection], frame_number: int) -> list[VehicleEvent]:
        """
        Feed detections from one frame. Returns VehicleEvents that completed this frame
        (both wires crossed or track became stale).
        """
        seen_ids: set[int] = set()

        for det in detections:
            tid = det.tracker_id
            seen_ids.add(tid)
            cx = det.centroid[0]

            if tid not in self._active:
                self._active[tid] = _TrackState(
                    vehicle_class=det.class_name,
                    first_seen_frame=frame_number,
                    last_seen_frame=frame_number,
                    first_seen_ts=det.timestamp_s,
                    last_seen_ts=det.timestamp_s,
                    last_centroid_x=cx,
                )
                continue

            state = self._active[tid]
            state.last_seen_frame = frame_number
            state.last_seen_ts = det.timestamp_s

            prev_cx = state.last_centroid_x
            if prev_cx is not None:
                self._check_crossing(state, det, prev_cx, cx)

            state.last_centroid_x = cx

        # Evict stale tracks
        completed: list[VehicleEvent] = []
        stale_ids = [
            tid
            for tid, state in self._active.items()
            if tid not in seen_ids
            and (frame_number - state.last_seen_frame) >= self._stale_frames
        ]
        for tid in stale_ids:
            completed.append(self._finalize(tid))

        return completed

    def flush(self) -> list[VehicleEvent]:
        """Finalize all remaining active tracks (e.g. at end of stream)."""
        events = [self._finalize(tid) for tid in list(self._active.keys())]
        return events

    def _check_crossing(
        self,
        state: _TrackState,
        det: Detection,
        prev_cx: int,
        cx: int,
    ) -> None:
        cfg = self._config

        # Only record the first crossing of each wire per track
        crossed_wires = {c.wire for c in state.crossings}

        for wire, line_x in (("A", cfg.line_a_x), ("B", cfg.line_b_x)):
            if wire in crossed_wires:
                continue
            # Sign-change crossing detection
            if (prev_cx - line_x) * (cx - line_x) < 0:
                state.crossings.append(
                    CrossingEvent(
                        tracker_id=det.tracker_id,
                        wire=wire,  # type: ignore[arg-type]
                        timestamp_s=det.timestamp_s,
                        frame_number=det.frame_number,
                        centroid=det.centroid,
                    )
                )

    def _finalize(self, tracker_id: int) -> VehicleEvent:
        state = self._active.pop(tracker_id)
        crossings_by_wire = {c.wire: c for c in state.crossings}
        ca = crossings_by_wire.get("A")
        cb = crossings_by_wire.get("B")

        speed_kmh: float | None = None
        direction = None
        incomplete = not (ca and cb)

        if ca and cb:
            speed_kmh = compute_speed(ca, cb, self._config.physical_distance_m)
            direction = "left_to_right" if ca.timestamp_s <= cb.timestamp_s else "right_to_left"

        return VehicleEvent(
            tracker_id=tracker_id,
            vehicle_class=state.vehicle_class,
            direction=direction,
            speed_kmh=speed_kmh,
            entered_at_s=state.first_seen_ts,
            exited_at_s=state.last_seen_ts,
            incomplete=incomplete,
            crossing_a=ca,
            crossing_b=cb,
        )
