import numpy as np

from src.utils.helpers import c, p


class Tester:
    """
    Utility class to quickly inspect values during development.
    """

    def __init__( self, obj = "", value = None, precision = 3, color1 = c.GREEN, color2 = c.BLACK, bold = False ):
        """
        Initialize test printer with formatting options.
        
        Args:
            obj: Label or object name to display.
            value: Value to inspect and display.
            precision: Decimal precision for floating point numbers.
            color1: Primary color for labels.
            color2: Secondary color for values.
            bold: Whether to use bold formatting.
        """
        self.obj = obj
        self.value = value
        self.precision = precision
        self.color1 = color1
        self.color2 = color2
        self.bold = bold

    def _print( self ):
        """
        Print the test object and value with configured colors and formatting.
        
        Returns:
            None
        """
        p(self.obj,
          self.value,
          precision = self.precision,
          color1 = self.color1,
          color2 = self.color2,
          bold = self.bold,
          )


def p_test():
    """
    Run comprehensive tests of printer functionality.
    
    Tests all data types (numbers, dicts, lists, arrays, strings) and color options.
    
    Returns:
        None (prints test output to console).
    """
    p("--- Testing Number ---")
    Tester("num", 42)._print()
    Tester("float", 3.14159)._print()

    p("--- Testing Dict ---")
    Tester("dict", { "a": 1, "b": 2 })._print()

    p("--- Testing List ---")
    Tester("list", [10, 20, 30, 40, 50, 60])._print()

    p("--- Testing Tuple ---")
    Tester("tuple", ("x", "y", "z"))._print()

    p("--- Testing Numpy Array ---")
    Tester("array", np.zeros((2, 3)))._print()

    p("--- Testing String ---")
    Tester("string", "hello")._print()

    p("--- Testing Empty ---")
    Tester("", None)._print()

    p("--- Testing Color ---")
    color1 = c.ORANGE
    color2 = c.PURPLE
    c.print_color(color1)
    c.print_color(color2)
    p(f"OBJ color test {color1.name}", f"VALUE color test {color2.name}", color1 = color1, color2 = color2)

    p("--- Testing Title w/blue ---")
    Tester("Title", color1 = c.BLUE)._print()

    p("--- Testing Value w/yellow ---")
    Tester("", "Value", color2 = c.YELLOW)._print()

    p("--- Testing Bold  ---")
    Tester("Bold", "Not bold")._print()
