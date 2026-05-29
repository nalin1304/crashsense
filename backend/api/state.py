"""
Shared in-process state for the Backend_API.

Holds the latest known drone position so /drone-status can answer in O(1)
(Requirement 11 criteria 7 and 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DroneStatusValue = Literal["idle", "in_transit", "arrived"]


@dataclass
class DroneState:
    lat: float = 0.0
    lon: float = 0.0
    status: DroneStatusValue = "idle"

    def update(self, *, lat: float, lon: float, status: DroneStatusValue) -> None:
        self.lat = lat
        self.lon = lon
        self.status = status


drone_state = DroneState()
