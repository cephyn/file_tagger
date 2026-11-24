from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
import os
import subprocess
import sys

def is_dark_color(color):
    """
    Determines if a color is dark (needs white text) or light (needs dark text).
    Uses the luminance formula: 0.299*R + 0.587*G + 0.114*B
    """
    if isinstance(color, str):
        color = QColor(color)
        
    # Calculate luminance (perceived brightness)
    luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) / 255
    
    # If luminance is less than 0.5, color is dark
    return luminance < 0.5

def get_score_color(score: float) -> QColor:
    """Get a color representing the match score (red to green)."""
    if score >= 0.8:
        return QColor(200, 255, 200)  # Light green
    elif score >= 0.6:
        return QColor(255, 255, 200)  # Light yellow
    elif score >= 0.4:
        return QColor(255, 230, 200)  # Light orange
    else:
        return QColor(255, 200, 200)  # Light red

def open_file(file_path: str):
    """Open a file with the system's default application."""
    if os.path.isfile(file_path):
        try:
            if sys.platform == 'win32':
                os.startfile(file_path)
            else:
                subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', file_path))
        except Exception as e:
            raise Exception(f"Could not open file: {str(e)}")

def open_containing_folder(file_path: str):
    """Open the folder containing the file and select it."""
    if os.path.exists(file_path):
        dir_path = os.path.dirname(file_path) if os.path.isfile(file_path) else file_path
        try:
            if sys.platform == 'win32':
                # On Windows, open Explorer and select the file
                subprocess.Popen(f'explorer /select,"{file_path}"')
            elif sys.platform == 'darwin':
                # On macOS, open Finder and select the file
                subprocess.call(['open', '-R', file_path])
            else:
                # On Linux, just open the containing directory
                subprocess.call(['xdg-open', dir_path])
        except Exception as e:
            raise Exception(f"Could not open containing folder: {str(e)}")