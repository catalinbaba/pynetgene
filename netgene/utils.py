from abc import ABC, abstractmethod
from datetime import datetime
from concurrent.futures import wait


class TaskExecutor:
    def __init__(self):
        pass

    @staticmethod
    def run_task(task, executor, *args, **kwargs):
        start = datetime.now()

        try:
            future = executor.submit(task, *args, **kwargs)
            wait([future])  # This should block until the task is done
        except Exception as ex:
            print("An error occurred while executing the task:", ex)
        finish = datetime.now()
        duration = finish - start
        return duration

class Printer(ABC):
    """
    Printer abstract class, equivalent to the Printer interface in Java.
    """
    @abstractmethod
    def print(self, message):
        pass

    @abstractmethod
    def println(self, message=''):
        pass


class ConsolePrinter(Printer):
    """
    Printer used to print on standard console, equivalent to the ConsolePrinter class in Java.
    """
    def print(self, message):
        print(message, end='')

    def println(self, message=''):
        print(message)


CONSOLE_PRINTER = 1
FILE_PRINTER = 2
