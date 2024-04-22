class GaException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class CrossoverException(GaException):
    def __init__(self, message: str):
        super().__init__(message)


class MutatorException(GaException):
    def __init__(self, message: str):
        super().__init__(message)


class SelectionException(GaException):
    def __init__(self, message: str):
        super().__init__(message)
