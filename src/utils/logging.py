from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.helpers import c, p, t


class Logger:
    """
    Simple logger for console and optional file output.
    Designed for training loops, evaluation runs, and debugging.
    """

    def __init__( self, log_file: Optional[Path] = None ):
        self.log_file = Path(log_file) if log_file is not None else None
        if self.log_file:
            self.log_file.parent.mkdir(parents = True, exist_ok = True)

    def write( self, text: str ) -> None:
        """
        Write a line of text to console and file if configured.
        """
        timestamp = datetime.now().strftime("%m.%d %H:%M:%S")
        line = f"[{timestamp}] {text}"

        if self.log_file is not None:
            with open(self.log_file, "a", encoding = "utf8") as f:
                f.write(line + "\n")

    def header( self, text: str ) -> None:
        """
        Write a visible section header.
        """
        self.write(f"=== {text} ===")
        t(text)

    def info( self, text: str ) -> None:
        """
        Write an informational line.
        """
        self.write(text)
        p("[Info]", text, color1 = c.BLUE, color2 = c.BLACK)

    def warn( self, text: str ) -> None:
        """
        Write a warning line.
        """
        self.write(f"Warning: {text}")
        p("[Warn]", text, color1 = c.ORANGE, color2 = c.BLACK)

    def error( self, text: str ) -> None:
        """
        Write an error line.
        """
        self.write(f"Error: {text}")
        p("[Error]", text, color1 = c.RED, color2 = c.BLACK)
