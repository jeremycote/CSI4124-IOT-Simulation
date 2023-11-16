from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4
from enum import Enum


class MessageState(Enum):
    IN_SYSTEM = 0
    COMPLETED = 1
    LOST = 2
    EXPIRED = 3

    def __str__(self) -> str:
        match self:
            case MessageState.IN_SYSTEM:
                return "In System"
            case MessageState.COMPLETED:
                return "Completed"
            case MessageState.LOST:
                return "Lost"
            case MessageState.EXPIRED:
                return "Expired"


@dataclass
class Message:
    """Represents a message being passed between nodes in the simulation"""

    # Absolute arrival time
    arrival_time: int

    # Time at which the message expires and automatically leaves the system
    expiration_time: int

    # Absolute departure time
    departure_time: Optional[int] = None

    # How long has the message waited (Cumulative)
    wait_time: int = 0

    # How long has the message been worked on (Cumulative)
    work_time: int = 0

    uuid: str = field(default_factory=lambda: str(uuid4()))

    state: MessageState = MessageState.IN_SYSTEM

    def __str__(self) -> str:
        return f"Message {self.uuid}:\nState: {self.state}\nArrival Time: {self.arrival_time},\nDeparture Time: {self.departure_time},\nWait Time: {self.departure_time},\nWork Time: {self.work_time},\nExpiration Time: {self.expiration_time}\n"

    def response_time(self) -> Optional[int]:
        if self.departure_time is None:
            return None

        return self.departure_time - self.arrival_time
