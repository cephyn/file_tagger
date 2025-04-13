import time
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QProgressBar
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QColor, QPixmap
from models import init_db
from config import Config
from vector_search import VectorSearch
try:
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
except ImportError:
    # Handle case where it might not exist in some installations,
    # although the error suggests it's expected.
    pass

# Also ensure sentence_transformers is imported somewhere obvious
import sentence_transformers
import onnxruntime # Add this too

class InitializationWorker(QThread):
    progress_signal = Signal(str, int)
    finished_signal = Signal(tuple)
    
    def __init__(self, password):
        super().__init__()
        self.password = password
        
    def run(self):
        try:
            # Initialize database
            self.progress_signal.emit("Initializing database...", 10)
            db_session = init_db()
            
            # Initialize config
            self.progress_signal.emit("Loading configuration...", 30)
            config = Config(self.password)
              # Initialize vector search
            self.progress_signal.emit("Setting up search engine...", 50)
            vector_search = VectorSearch(db_session, config)
            
            # Loading existing files and tags
            self.progress_signal.emit("Loading files and tags...", 70)
            time.sleep(0.5)  # Small delay to make the progress visible
            
            # Finalizing initialization
            self.progress_signal.emit("Finalizing...", 90)
            time.sleep(0.5)  # Small delay to make the progress visible
            
            # Send the results back to the main thread
            self.finished_signal.emit((db_session, config, vector_search))
            
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}", 100)
            self.finished_signal.emit((None, None, None))

class SplashScreen(QPixmap):
    def __init__(self):
        # Create a pixmap for the splash screen background
        splash_pixmap = QPixmap(400, 250)
        splash_pixmap.fill(QColor(45, 45, 48))  # Dark background color
        
        super().__init__(splash_pixmap)
        
        # Set up the layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add title
        title_label = QLabel("File Tagger")
        title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Add subtitle
        subtitle_label = QLabel("Loading application...")
        subtitle_label.setStyleSheet("color: #cccccc; font-size: 14px;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)
        
        # Add spacer
        layout.addSpacing(20)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: #2d2d30;
                color: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                width: 10px;
                margin: 0px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QLabel("Starting up...")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Create a central widget for the layout
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        central_widget.setGeometry(0, 0, 400, 250)
        
        # Set window flags
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        
    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.progress_bar.setValue(value)
        self.repaint()  # Force an update of the splash screen