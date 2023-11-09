from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from message import Message

import numpy as np


class OnOffServerState(Enum):
    """The state of an ON OFF Server"""

    OFF = 0
    ON = 1


class SimulationComponent(ABC):
    """Protocol for objects that are orchastrated by the simulation"""

    @abstractmethod
    def tick(self, time: int):
        """Method called every tick of the simulation"""


class Simulation:
    time: int = 0
    components: List[SimulationComponent] = []

    def __init__(self) -> None:
        pass

    def register_component(self, component: SimulationComponent):
        self.components.append(component)

    def simulate(self, start_time: int = 0, end_time: int = 10000):
        for i in range(start_time, end_time):
            for component in self.components:
                component.tick(i)


class Input(ABC):
    """Protocol for objects that can take the output of other objects"""

    @abstractmethod
    def process_input(self, message: Message, time: int) -> bool:
        """Returns True if the input is accepeted"""


class Output(ABC):
    """Protocol for objects that can output"""

    listeners: List[Input]

    def __init__(self) -> None:
        self.listeners = []

    def register_output_listener(self, destination):
        self.listeners.append(destination)

    def output(self, message: Message, time: int):
        for listener in self.listeners:
            message_accepted = listener.process_input(message, time)

            if message_accepted:
                return


class Server(Input, Output, SimulationComponent):
    """"""


class Generator(Output, SimulationComponent):
    """"""


class Sink(Input, SimulationComponent):
    """"""

    def process_input(self, message: Message, time: int) -> bool:
        return super().process_input(message, time)

    def tick(self, time: int):
        return super().tick(time)


class MessageGenerator(Generator):
    """Generator for IOT Messages"""

    # Time at which to generate the next message
    next_generation: int = 0

    # Lambda value for poisson distribution
    l: float

    # Create message generator with lambda parameter
    def __init__(self, l: float) -> None:
        super().__init__()
        self.l = l

    def tick(self, time: int):
        if self.next_generation <= time:
            # Round the inter arrival time to the nearest millisecond
            inter_arrival_time = round(np.random.exponential(scale=1 / self.l) * 1000)
            self.next_generation += inter_arrival_time

            # Output the message from the generator
            self.output(Message(arrival_time=time), time)


class OnOffServer(Server):
    """A server that can be ON or OFF"""

    # The lambda parameter to generate service times
    l: float

    # Is the server ON or OFF
    state: OnOffServerState = OnOffServerState.ON

    # The message that the server is currently processing
    message: Optional[Message] = None

    # How long it will take to process the message
    message_service_length: Optional[int] = None

    # When did message processing start
    message_service_start: Optional[int] = None

    def __init__(self, l: float) -> None:
        """Construct an OnOffServer

        Args:
            l (float): Lambda parameter for generating service times
        """
        super().__init__()
        self.l = l

    def output_message(self, time: int):
        if self.message is None:
            raise ValueError("No message to output")

        self.message.work_time += self.message_service_length

        print(f"Departing {self.message}")

        self.output(self.message, time)
        self.message = None

    def process_input(self, message: Message, time: int) -> bool:
        # Do not accept messages if the server is off
        if self.state is OnOffServerState.OFF:
            return False

        if (
            self.message
            and time <= self.message_service_start + self.message_service_length
        ):
            # Message is complete
            self.output_message(time)
        elif self.message:
            return False

        self.message = message
        self.message_service_start = time

        # Generate a message service length rounded to the nearest millisecond
        self.message_service_length = round(np.random.exponential(self.l) * 1000)

        print(self.message)
        return True

    def tick(self, time: int):
        if (
            self.message
            and time <= self.message_service_start + self.message_service_length
        ):
            # Message is complete
            self.output_message(time)


if __name__ == "__main__":
    message_generator = MessageGenerator(l=1)
    server = OnOffServer(l=1)
    sink = Sink()

    message_generator.register_output_listener(server)
    server.register_output_listener(sink)

    simulation = Simulation()
    simulation.register_component(message_generator)
    simulation.register_component(server)
    simulation.register_component(sink)

    # Simulate for 10000ms -> 10s
    simulation.simulate(0, 10000)
