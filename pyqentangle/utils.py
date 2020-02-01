
class InvalidQuantumStateException(Exception):
    def __init__(self, msg):
        self.message = msg


class OutOfRangeException(Exception):
    def __init__(self, value):
        self.msg = "Out of range: " + str(value)

    def __str__(self):
        return repr(self.msg)


class UnequalLengthException(Exception):
    def __init__(self, array1, array2):
        self.msg = "Unequal length: " + str(len(array1)) + " vs. " + str(len(array2))

    def __str__(self):
        return repr(self.msg)


class InvalidMatrix(Exception):
    def __init__(self, msg = "Invalid matrix"):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

