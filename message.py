from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    """Represents a message being passed between nodes in the simulation"""

    # Absolute arrival time
    arrival_time: int

    # Absolute departure time
    departure_time: Optional[int] = None

    # How long has the message waited (Cumulative)
    wait_time: int = 0

    # How long has the message been worked on (Cumulative)
    work_time: int = 0
