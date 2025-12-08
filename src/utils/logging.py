from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.helpers import c, p, t


class Logger:
    """
    Simple logger for console and optional file output.
    Designed for training loops, evaluation runs, and debugging.
    """

    def __init__( self, log_file: Optional[Path] = None ,cfg=None):
        """
        Initialize logger with optional file output.
        
        Args:
            log_file (Path, optional): Path to log file. If provided, creates parent directories.
            cfg (optional): Configuration object for model info extraction.
        """
        self.log_file = Path(log_file) if log_file is not None else None
        self.cfg = cfg
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
        Write a visible section header with formatted output.
        
        Args:
            text (str): Header text to display.
        
        Returns:
            None
        """
        self.write(f"=== {text} ===")
        t(text)

    # def info( self, text: str ) -> None:
    #     """
    #     Write an informational line.
    #     """
    #     self.write(text)
    #     p("[Info]", text, color1 = c.BLUE, color2 = c.BLACK)

    def info(self, text):
        """
        Write informational output with special handling for PyTorch models.
        
        If text is a PyTorch model, displays a clean torchinfo summary.
        Otherwise logs as informational text.
        
        Args:
            text: Text to log or PyTorch model to summarize.
        
        Returns:
            None
        """
        from torchinfo import summary
        import torch.nn as nn

        # If the user passes a model
        if isinstance(text, nn.Module):
            try:
                # Try to infer model input channels from common cases
                in_channels = 3
                if hasattr(text, 'in_channels'):
                    in_channels = text.in_channels

                # Default spatial size (can be adjusted)
                img_size = self.cfg.train.image_size

                model_summary = summary(
                        text,
                        input_size=(1, in_channels, img_size, img_size),
                        depth=3,
                        col_names=("input_size", "output_size", "num_params")
                )

                self.write(str(model_summary))
                p("", str(model_summary))
                # p("[Model]", "Summary printed via torchinfo", color1=c.BLUE, color2=c.BLACK)
                return

            except Exception as e:
                # Fallback to normal printing if summary fails
                self.write(str(text))
                p("[Info]", f"torchinfo failed: {e}", color1=c.ORANGE)
                return

        # Normal text logging
        self.write(str(text))
        p("[Info]", str(text), color1=c.BLUE, color2=c.BLACK)


    def warn( self, text: str ) -> None:
        """
        Write a warning line with colored output.
        
        Args:
            text (str): Warning message to display.
        
        Returns:
            None
        """
        self.write(f"Warning: {text}")
        p("[Warn]", text, color1 = c.ORANGE, color2 = c.BLACK)

    def error( self, text: str ) -> None:
        """
        Write an error line with colored output.
        
        Args:
            text (str): Error message to display.
        
        Returns:
            None
        """
        self.write(f"Error: {text}")
        p("[Error]", text, color1 = c.RED, color2 = c.BLACK)

    def exception(self, text: str, exc: Exception) -> None:
        """
        Log an exception with detailed traceback information.
        
        Args:
            text (str): Description of the exception context.
            exc (Exception): The exception object to log.
        
        Returns:
            None
        """
        import traceback

        error_msg = f"Exception: {text}\n{str(exc)}\n{traceback.format_exc()}"
        self.write(error_msg)
        p("[Exception]", text, color1=c.RED, color2=c.RED)
        p("", str(exc), color1=c.ORANGE, color2=c.ORANGE)