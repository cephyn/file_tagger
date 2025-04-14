import time
import logging
import traceback
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QProgressBar
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QColor, QPixmap
from models import init_db
from config import Config
from vector_search import VectorSearch

# Set ONNX environment variable here too for the worker thread
os.environ["CHROMADB_DISABLE_ONNX"] = "1"

# Log initialization steps
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('initialization_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Initialization")

# Safely try imports that might be problematic after packaging
logger.debug("Attempting to import potentially problematic modules...")
try:
    logger.debug("Importing sentence_transformers...")
    import sentence_transformers
    logger.debug("Successfully imported sentence_transformers")
except Exception as e:
    logger.error(f"Error importing sentence_transformers: {str(e)}")
    logger.error(traceback.format_exc())

try:
    logger.debug("Importing onnxruntime...")
    import onnxruntime
    logger.debug("Successfully imported onnxruntime")
except Exception as e:
    logger.error(f"Error importing onnxruntime: {str(e)}")
    logger.error(traceback.format_exc())

class InitializationWorker(QThread):
    progress_signal = Signal(str, int)
    finished_signal = Signal(tuple)
    
    def __init__(self, password):
        super().__init__()
        self.password = password
        logger.debug("InitializationWorker created")
        
    def run(self):
        logger.debug("InitializationWorker.run() started")
        try:
            # Initialize database
            logger.debug("Starting database initialization")
            self.progress_signal.emit("Initializing database...", 10)
            db_session = init_db()
            logger.debug("Database initialized successfully")
            
            # Initialize config
            logger.debug(f"Starting config initialization with password (length: {len(self.password)})")
            self.progress_signal.emit("Loading configuration...", 30)
            try:
                config = Config(self.password)
                logger.debug("Config initialized successfully")
            except Exception as config_error:
                logger.error(f"Config initialization failed: {str(config_error)}")
                logger.error(traceback.format_exc())
                raise
            
            # Initialize vector search
            logger.debug("Starting vector search initialization")
            self.progress_signal.emit("Setting up search engine...", 50)
            try:
                vector_search = VectorSearch(db_session, config)
                logger.debug("Vector search initialized successfully")
            except Exception as vs_error:
                logger.error(f"Vector search initialization failed: {str(vs_error)}")
                logger.error(traceback.format_exc())
                raise
            
            # Loading existing files and tags
            logger.debug("Loading existing files and tags")
            self.progress_signal.emit("Loading files and tags...", 70)
            time.sleep(0.5)  # Small delay to make the progress visible
            
            # Finalizing initialization
            logger.debug("Finalizing initialization")
            self.progress_signal.emit("Finalizing...", 90)
            time.sleep(0.5)  # Small delay to make the progress visible
            
            # Send the results back to the main thread
            logger.debug("Initialization completed successfully, emitting signal")
            self.finished_signal.emit((db_session, config, vector_search))
            
        except Exception as e:
            logger.error(f"Initialization failed with error: {str(e)}")
            logger.error(traceback.format_exc())
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