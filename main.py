from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from message import Message, MessageState

import numpy as np

from simulation import SimulationParameters, SimulationResults


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
    time: int

    # Components to orchestrate in the simulation
    components: List[SimulationComponent]

    def __init__(self) -> None:
        self.time = 0
        self.components = []

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

    def __str__(self) -> str:
        result = ""
        for component in self.components:
            result += f"{component}\n\n"

        return result


class Input(ABC):
    """Protocol for objects that can take the output of other objects"""

    @abstractmethod
    def process_input(self, message: Message, time: int) -> bool:
        """Returns True if the input was consumed"""


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

    label: str
    # Message State to set messages that enter this sink.
    message_state: Optional[MessageState]
    messages: List[Message]

    # Should print debug messages
    debug: bool

    def __init__(
        self,
        label: str,
        message_state: Optional[MessageState] = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.label = label
        self.message_state = message_state
        self.messages = []
        self.debug = debug

    def process_input(self, message: Message, time: int) -> bool:
        # Message has left the system through a sink.
        message.departure_time = time
        if self.message_state:
            message.state = self.message_state

        if self.debug:
            print(self.label, message)
        self.messages.append(message)

        # A sink will always accept messages
        return True

    def tick(self, time: int, delta: int):
        return super().tick(time, delta)

    def __str__(self) -> str:
        result = f"Sink {self.label}. Messages:\n"
        for message in self.messages:
            result += f"----------\n{message}----------"
        return result


class MessageGenerator(Generator):
    """Generator for IOT Messages"""

    n_messages_generated: int

    # Time at which to generate the next message
    next_generation: int

    # Lambda value for poisson distribution
    l: float

    # Lifetime value for the message, once this amount of time has passed, the message is expired and leaves the system
    lifetime: int

    # Create message generator with lambda parameter
    def __init__(self, l: float, lifetime: int) -> None:
        """Construct a MessageGenerator that follows a poisson distribution for message generation.

        Args:
            l (float): lambda parameter of the poisson distribution
        """
        super().__init__()
        self.l = l
        self.lifetime = lifetime
        self.n_messages_generated = 0
        self.next_generation = 0

    def tick(self, time: int, delta: int):
        if self.l == 0:
            return

        if self.next_generation <= time:
            # Round the inter arrival time to the nearest millisecond
            # By generating inter arrival times using an exponential distribution, the output of the generator will follow a poisson distribution
            inter_arrival_time = round(np.random.exponential(scale=1 / self.l) * 1000)
            self.next_generation += inter_arrival_time

            # Output the message from the generator
            self.output(
                Message(arrival_time=time, expiration_time=time + self.lifetime), time
            )

            self.n_messages_generated += 1

    def __str__(self) -> str:
        return f"Message Generator created {self.n_messages_generated} messages using lambda={self.l} and lifetime={self.lifetime}"


class OnOffServer(Server):
    """A server that can be ON or OFF"""

    # Message queue
    messages: List[Message]

    # The lambda parameter to generate service times
    service_lambda: float

    # How many messages can the process hold in it's queue
    capacity: int

    # Probability of a message being lost if inputted while the server is off
    loss_propability: float

    # Destination for messages that expire or are lost
    lost_messages_output: Optional[Input]

    # Is the server ON or OFF
    state: OnOffServerState

    # Number of times that the state changed
    n_state_toggles: int

    # Time at which the server state will toggle from ON->OFF or OFF->ON
    state_flop_time: int

    # The lambda parameter to generate state flip flops
    state_flop_lambda: float

    # How long it will take to process the current message
    message_service_length: Optional[int]

    # When did message processing start
    message_service_start: Optional[int]

    # How much work has been done on the message. This is used to keep track of the server being ON/OFF.
    message_service_effectuated: int

    # Should print debug messages
    debug: bool

    def __init__(
        self,
        service_lambda: float,
        state_flop_lambda: float,
        capacity: int,
        loss_probability: float,
        debug: bool = False,
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
        self.messages = []
        self.state = OnOffServerState.OFF
        self.n_state_toggles = 0
        self.state_flop_time = 0
        self.message_service_length = None
        self.message_service_start = None
        self.message_service_effectuated = 0
        self.debug = debug

    def remove_processing_message(self):
        self.message_service_effectuated = 0
        self.message_service_length = None
        self.message_service_start = None

        if not self.empty():
            self.messages.pop(0)

    def output_message(self, time: int):
        if self.empty():
            raise ValueError("No message to output")

        self.messages[0].work_time += self.message_service_length

        if self.debug:
            print(f"Departing {self.messages[0]}")

        self.output(self.messages[0], time)

        # Clear data to prepare for the next message
        self.remove_processing_message()

    def lost_message(self, message: Message, time: int):
        message.state = MessageState.LOST
        self.lost_messages_output.process_input(message, time)

    def cleanup_expired_messages(self, time: int):
        if self.empty():
            return

        if self.messages[0].expiration_time <= time:
            if self.debug:
                print("expired", self.messages[0], time)
            self.messages[0].state = MessageState.EXPIRED
            # TODO: This might not be right
            self.messages[0].work_time = self.message_service_effectuated
            self.lost_messages_output.process_input(self.messages[0], time)
            self.remove_processing_message()

        # Traverse in reverse order to remove items without interfering with the rest of the loop
        for i, message in reversed(list(enumerate(self.messages))):
            if message.expiration_time <= time:
                expired_message = self.messages.pop(i)
                expired_message.state = MessageState.EXPIRED
                self.lost_messages_output.process_input(message, time)

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
                self.lost_message(message, time)
                return True

        # Check if any messages are done processing and output them. Doing this before checking if the queue is full is important.
        if self.message_processing_complete():
            self.output_message(time)

        # If the queue is full, the message is lost
        if self.full():
            self.lost_message(message, time)
            return True

        self.messages.append(message)
        self.message_service_start = time

        # Generate a message service length rounded to the nearest millisecond
        self.message_service_length = round(
            np.random.exponential(self.service_lambda) * 1000
        )

        if self.debug:
            print("Server Received", message)
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

        self.cleanup_expired_messages(time)

        if self.state == OnOffServerState.ON and not self.empty():
            # Increment effectuated work of current message
            self.message_service_effectuated += delta

        if self.state_flop_time <= time:
            self.state_flop_time += round(
                np.random.exponential(self.state_flop_lambda) * 1000
            )

            # Toggle the server state
            self.state = (
                OnOffServerState.ON
                if self.state == OnOffServerState.OFF
                else OnOffServerState.OFF
            )
            self.n_state_toggles += 1

    def register_loss_output_listener(self, destination: Input):
        self.lost_messages_output = destination

    def __str__(self) -> str:
        result = f"OnOffServer toggled state {self.n_state_toggles} times. Messages:\n"
        for message in self.messages:
            result += f"----------\n{message}----------"
        return result


def run_simulation(simulation_parameters: SimulationParameters) -> SimulationResults:
    message_generator = MessageGenerator(
        l=simulation_parameters.message_lambda,
        lifetime=simulation_parameters.message_lifetime,
    )
    server = OnOffServer(
        service_lambda=simulation_parameters.service_lambda,
        state_flop_lambda=simulation_parameters.on_off_lambda,
        capacity=simulation_parameters.capacity,
        loss_probability=simulation_parameters.loss_probability,
    )
    sink = Sink("Completed Messages", MessageState.COMPLETED)
    lostSink = Sink("Lost / Expired Messages")

    message_generator.register_output_listener(server)
    server.register_output_listener(sink)
    server.register_loss_output_listener(lostSink)

    simulation = Simulation()
    simulation.register_component(message_generator)
    simulation.register_component(server)
    simulation.register_component(sink)
    simulation.register_component(lostSink)

    # Simulate for 10000ms -> 10s
    simulation.simulate(0, simulation_parameters.simulation_length)

    return SimulationResults(
        n_messages_generated=message_generator.n_messages_generated,
        n_messages_lost=len(
            [
                message
                for message in lostSink.messages
                if message.state is MessageState.LOST
            ]
        ),
        n_messages_expired=len(
            [
                message
                for message in lostSink.messages
                if message.state is MessageState.EXPIRED
            ]
        ),
        n_messages_completed=len(
            [
                message
                for message in sink.messages
                if message.state is MessageState.COMPLETED
            ]
        ),
        averate_response_time=(
            (
                sum([message.response_time() for message in sink.messages])
                / len(sink.messages)
            )
            if len(sink.messages) > 0
            else 0
        ),
    )


if __name__ == "__main__":
    message_generator = MessageGenerator(l=1, lifetime=2000)
    server = OnOffServer(
        service_lambda=1, state_flop_lambda=1, capacity=3, loss_probability=0.5
    )
    sink = Sink("Completed Messages", MessageState.COMPLETED)
    lostSink = Sink("Lost / Expired Messages")

    message_generator.register_output_listener(server)
    server.register_output_listener(sink)
    server.register_loss_output_listener(lostSink)

    simulation = Simulation()
    simulation.register_component(message_generator)
    simulation.register_component(server)
    simulation.register_component(sink)
    simulation.register_component(lostSink)

    # Simulate for 10000ms -> 10s
    simulation.simulate(0, 10000)

    print(
        "##############################\n# Final State:               #\n##############################"
    )
    print(simulation)

    print(
        "##############################\n# Summary:                   #\n##############################"
    )

    print(
        f"{len([message for message in sink.messages if message.state is MessageState.COMPLETED])} messages were completed"
    )
    print(
        f"{len([message for message in server.messages if message.state is MessageState.IN_SYSTEM])} messages were still in the system"
    )
    print(
        f"{len([message for message in lostSink.messages if message.state is MessageState.LOST])} messages were lost"
    )
    print(
        f"{len([message for message in lostSink.messages if message.state is MessageState.EXPIRED])} messages expired"
    )
