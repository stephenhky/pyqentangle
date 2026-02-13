
class InvalidQuantumStateException(Exception):
    """Exception raised for invalid quantum states."""
    
    def __init__(self, msg):
        """Initialize the exception.
        
        Args:
            msg (str): Error message.
        """
        self.message = msg


class OutOfRangeException(Exception):
    """Exception raised when a value is outside the expected range."""
    
    def __init__(self, value):
        """Initialize the exception.
        
        Args:
            value: Value that is out of range.
        """
        self.msg = "Out of range: " + str(value)

    def __str__(self):
        """Return string representation of the exception.
        
        Returns:
            str: String representation of the exception.
        """
        return repr(self.msg)


class UnequalLengthException(Exception):
    """Exception raised when arrays have unequal lengths."""
    
    def __init__(self, array1, array2):
        """Initialize the exception.
        
        Args:
            array1: First array.
            array2: Second array.
        """
        self.msg = "Unequal length: " + str(len(array1)) + " vs. " + str(len(array2))

    def __str__(self):
        """Return string representation of the exception.
        
        Returns:
            str: String representation of the exception.
        """
        return repr(self.msg)


class InvalidMatrix(Exception):
    """Exception raised for invalid matrices."""
    
    def __init__(self, msg="Invalid matrix"):
        """Initialize the exception.
        
        Args:
            msg (str, optional): Error message. Defaults to "Invalid matrix".
        """
        self.msg = msg

    def __str__(self):
        """Return string representation of the exception.
        
        Returns:
            str: String representation of the exception.
        """
        return repr(self.msg)

