import numpy as np

from .helpers import c, p


class Tester:
    """
    Utility class to quickly inspect values during development.
    """

    def __init__( self, obj = "", value = None, precision = 3, color1 = c.GREEN, color2 = c.BLACK, bold = False ):
        self.obj = obj
        self.value = value
        self.precision = precision
        self.color1 = color1
        self.color2 = color2
        self.bold = bold

    def _print( self ):
        p(self.obj,
          self.value,
          precision = self.precision,
          color1 = self.color1,
          color2 = self.color2,
          bold = self.bold,
          )


def test_print():
    print("\n--- Testing Number ---")
    Tester("num", 42)._print()
    Tester("float", 3.14159)._print()

    print("\n--- Testing Dict ---")
    Tester("dict", { "a": 1, "b": 2 })._print()

    print("\n--- Testing List ---")
    Tester("list", [10, 20, 30, 40, 50, 60])._print()

    print("\n--- Testing Tuple ---")
    Tester("tuple", ("x", "y", "z"))._print()

    print("\n--- Testing Numpy Array ---")
    Tester("array", np.zeros((2, 3)))._print()

    print("\n--- Testing String ---")
    Tester("string", "hello")._print()

    print("\n--- Testing Empty ---")
    Tester("", None)._print()

    print("\n--- Testing Color ---")
    color1 = c.ORANGE
    color2 = c.PURPLE
    c.print_color(color1)
    c.print_color(color2)
    p(f"OBJ color test {color1.name}", f"VALUE color test {color2.name}", color1 = color1, color2 = color2)

    print("\n--- Testing Title w/blue ---")
    Tester("Title", color1 = c.BLUE)._print()

    print("\n--- Testing Value w/yellow ---")
    Tester("", "Value", color2 = c.YELLOW)._print()

    print("\n--- Testing Bold  ---")
    Tester("Bold", "Not bold")._print()
