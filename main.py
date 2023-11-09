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
    def tick(self, time: int, delta: int):
        """Method called every tick of the simulation"""


class Simulation:
    """Holds the state of the simulation."""

    # Simulation time represented in milliseconds
    time: int = 0

    # Components to orchestrate in the simulation
    components: List[SimulationComponent] = []

    def register_component(self, component: SimulationComponent):
        """Register a component in the simulation runtime.

        When a component is registered, their tick function is called every ms of simulation time

        Args:
            component (SimulationComponent): Component to register
        """
        self.components.append(component)

    def simulate(self, start_time: int = 0, end_time: int = 10000):
        """Simulate from start to end time

        Args:
            start_time (int, optional): Start time of the simulation. Defaults to 0.
            end_time (int, optional): End time of the simulation. Defaults to 10000.
        """
        for i in range(start_time, end_time):
            for component in self.components:
                component.tick(i, 1)


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
        # Message has left the system through a sink.
        message.departure_time = time

        print("Sink", message)

        # A sink will always accept messages
        return True

    def tick(self, time: int, delta: int):
        return super().tick(time, delta)


class MessageGenerator(Generator):
    """Generator for IOT Messages"""

    # Time at which to generate the next message
    next_generation: int = 0

    # Lambda value for poisson distribution
    l: float

    # Create message generator with lambda parameter
    def __init__(self, l: float) -> None:
        """Construct a MessageGenerator that follows a poisson distribution for message generation.

        Args:
            l (float): lambda parameter of the poisson distribution
        """
        super().__init__()
        self.l = l

    def tick(self, time: int, delta: int):
        if self.next_generation <= time:
            # Round the inter arrival time to the nearest millisecond
            # By generating inter arrival times using an exponential distribution, the output of the generator will follow a poisson distribution
            inter_arrival_time = round(np.random.exponential(scale=1 / self.l) * 1000)
            self.next_generation += inter_arrival_time

            # Output the message from the generator
            self.output(Message(arrival_time=time), time)


class OnOffServer(Server):
    """A server that can be ON or OFF"""

    # Message queue
    messages: List[Message] = []

    # The lambda parameter to generate service times
    service_lambda: float

    # How many messages can the process hold in it's queue
    capacity: int

    # Probability of a message being lost if inputted while the server is off
    loss_propability: float

    # Is the server ON or OFF
    state: OnOffServerState = OnOffServerState.OFF

    # Time at which the server state will toggle from ON->OFF or OFF->ON
    state_flop_time: int = 0

    # The lambda parameter to generate state flip flops
    state_flop_lambda: float

    # How long it will take to process the current message
    message_service_length: Optional[int] = None

    # When did message processing start
    message_service_start: Optional[int] = None

    # How much work has been done on the message. This is used to keep track of the server being ON/OFF.
    message_service_effectuated: int = 0

    def __init__(
        self,
        service_lambda: float,
        state_flop_lambda: float,
        capacity: int,
        loss_probability: float,
    ) -> None:
        """Construct an OnOffServer

        Args:
            l (float): Lambda parameter for generating service times
        """
        super().__init__()
        self.service_lambda = service_lambda
        self.state_flop_lambda = state_flop_lambda
        self.capacity = capacity
        self.loss_propability = loss_probability

    def output_message(self, time: int):
        if self.empty():
            raise ValueError("No message to output")

        self.messages[0].work_time += self.message_service_length

        print(f"Departing {self.messages[0]}")

        self.output(self.messages[0], time)

        # Clear data to prepare for the next message
        self.message_service_effectuated = 0
        self.message_service_length = None
        self.message_service_start = None

        self.messages.pop(0)

    def message_processing_complete(self) -> bool:
        """Returns True if the currently processed message is complete.

        Return False if there are no messages to process
        """
        if self.empty() or self.message_service_length is None:
            return False

        return self.message_service_effectuated >= self.message_service_length

    def full(self) -> bool:
        return len(self.messages) >= self.capacity

    def empty(self) -> bool:
        return len(self.messages) == 0

    def process_input(self, message: Message, time: int) -> bool:
        # If the server is off, simulate message loss
        if self.state is OnOffServerState.OFF:
            loss_instance = np.random.uniform(0, 1)
            if loss_instance > self.loss_propability:
                return True

        if self.message_processing_complete():
            self.output_message(time)

        if self.full():
            return False

        self.messages.append(message)
        self.message_service_start = time

        # Generate a message service length rounded to the nearest millisecond
        self.message_service_length = round(
            np.random.exponential(self.service_lambda) * 1000
        )

        print(message)
        return True

    def tick(self, time: int, delta: int):
        processing_completed_messages = True

        # Process all completed messages from the queue
        while (
            processing_completed_messages
            and self.state == OnOffServerState.ON
            and not self.empty()
        ):
            if self.message_processing_complete():
                self.output_message(time)
            else:
                processing_completed_messages = False

        if self.state == OnOffServerState.ON and not self.empty():
            # Increment effectuated work of current message
            self.message_service_effectuated += delta

        if self.state_flop_time <= time:
            self.state_flop_time += round(
                np.random.exponential(self.state_flop_lambda) * 1000
            )

            # print(
            #     f"Toggling server state from {self.state} to {OnOffServerState.ON if self.state == OnOffServerState.OFF else OnOffServerState.OFF} at time {time}"
            # )

            # Toggle the server state
            self.state = (
                OnOffServerState.ON
                if self.state == OnOffServerState.OFF
                else OnOffServerState.OFF
            )


if __name__ == "__main__":
    message_generator = MessageGenerator(l=1)
    server = OnOffServer(
        service_lambda=1, state_flop_lambda=1, capacity=1, loss_probability=0.5
    )
    sink = Sink()

    message_generator.register_output_listener(server)
    server.register_output_listener(sink)

    simulation = Simulation()
    simulation.register_component(message_generator)
    simulation.register_component(server)
    simulation.register_component(sink)

    # Simulate for 10000ms -> 10s
    simulation.simulate(0, 10000)
