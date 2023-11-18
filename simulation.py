from dataclasses import dataclass


@dataclass
class SimulationParameters:
    service_lambda: float
    on_off_lambda: float
    capacity: int
    loss_probability: float
    message_lambda: float
    message_lifetime: int
    simulation_length: int


@dataclass
class SimulationResults:
    n_messages_generated: float
    n_messages_lost: float
    n_messages_expired: float
    n_messages_completed: float
    averate_response_time: float
