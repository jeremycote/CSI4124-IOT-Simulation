from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from message import Message


class OnOffServerState(Enum):
    """The state of an ON OFF Server"""

    OFF = 0
    ON = 1


class SimulationComponent(ABC):
    """Protocol for objects that are orchastrated by the simulation"""

    @abstractmethod
    def tick(time: int):
        """Method called every tick of the simulation"""


class Simulation:
    time: int
    components: List[SimulationComponent] = []

    def register_component(self, component: SimulationComponent):
        self.components.append(component)

    def simulate(self, start_time: int = 0, end_time: int = 10):
        for i in range(start_time, end_time):
            for component in self.components:
                component.tick(i)


class Input(ABC):
    """Protocol for objects that can take the output of other objects"""

    @abstractmethod
    def process_input(message: Message) -> bool:
        """Returns True if the input is accepeted"""


class Output(ABC):
    """Protocol for objects that can output"""

    listeners: List[Input] = []

    def register_output_listener(self, destination):
        self.listeners.append(destination)

    def output(self, message: Message):
        for listener in self.listeners:
            message_accepted = listener.process_input(message)

            if message_accepted:
                return


class Server(Input, Output, SimulationComponent):
    """"""


class Generator(Output, SimulationComponent):
    """"""


class Sink(Input, SimulationComponent):
    """"""

    def process_input(self, message: Message) -> bool:
        return super().process_input()

    def tick(self, time: int):
        return super().tick()


class MessageGenerator(Generator):
    """Generator for IOT Messages"""

    def tick(self, time: int):
        self.output(Message())


class OnOffServer(Server):
    """A server that can be ON or OFF"""

    state: OnOffServerState

    def process_input(self, message: Message) -> bool:
        print(message)
        return super().process_input()

    def tick(self, time: int):
        return super().tick()


if __name__ == "__main__":
    message_generator = MessageGenerator()
    server = OnOffServer()
    sink = Sink()

    message_generator.register_output_listener(server)
    server.register_output_listener(sink)

    simulation = Simulation()
    simulation.register_component(message_generator)
    simulation.register_component(server)
    simulation.register_component(sink)

    simulation.simulate(0, 10)

# class QueueItem(Protocol):
#     def start_service(self, time: int):
#         """Set the time at which the itme started being processed

#         Args:
#             time (int): time at which the item started being processed
#         """

#     def get_arrival_time(self) -> int:
#         """Get the arrival time"""

#     def get_departure_time(self) -> int:
#         """Compute the departure time. Raises a value error if the item has not started being served"""


# class ProcessingQueue:
#     # items that have arrived
#     queue: List[QueueItem] = []

#     # items that will arrive
#     items: List[QueueItem]

#     items_processed = 0

#     def __init__(self, items: List[QueueItem]) -> None:
#         self.items = items

#     def tick(self) -> None:
#         """Step function for the discrete simulation.

#         Each call processes one unit of systme time.
#         """
#         # If an item's absolute arrive time is the current simulation time or before, put them in the queue. Thus, they have "arrived".
#         if len(self.items) > 0 and self.items[0].get_arrival_time() <= self.time:
#             self.queue.append(self.items.pop(0))

#             # If no item is currently being processed, start serving them.
#             if len(self.queue) == 1:
#                 self.queue[0].start_service(self.time)

#         # If the currently processed item's absolute departure time is the current simulation time or before, remove them from the queue. Thus, they have "departed".
#         if len(self.queue) > 0 and self.queue[0].get_departure_time() <= self.time:
#             finished_item = self.queue.pop(0)
#             print(
#                 f'Waiting time for item "{self.items_processed}": {finished_item.start_service_time - finished_item.arrival_time} sec'
#             )
#             print(
#                 f'Total system time for item "{self.items_processed}": {self.time - finished_item.arrival_time} sec\n'
#             )

#             self.items_processed += 1

#             # Start processing the next item if one is waiting.
#             if len(self.queue) > 0:
#                 self.queue[0].start_service(self.time)

#         # Increment the system time by 1
#         self.time += 1

#     def finished(self) -> bool:
#         """Check if the queues are empty

#         Returns:
#             bool: True when the simulation is complete.
#         """
#         return len(self.queue) == 0 and len(self.items) == 0


# class OnOffServer(Server):
#     """Represents a mobile sender with QoS settings for sending messages.

#     It can be in an ON and OFF state for exponentially distributed amounts of time.
#     Messages are processed with a mean service demand.
#     Messages arrive in it's queue with a mean arrival rate using a poisson distribution.

#     """

#     time: int

#     def __init__(self) -> None:
#         time = 0
