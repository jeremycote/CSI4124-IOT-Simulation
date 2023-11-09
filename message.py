from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


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

    uuid: str = field(default_factory=lambda: str(uuid4()))

    def __str__(self) -> str:
        return f"Message {self.uuid} [Arrival Time: {self.arrival_time}, Departure Time: {self.departure_time}, Wait Time: {self.departure_time}, Work Time: {self.work_time}]"
