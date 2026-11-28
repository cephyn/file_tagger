import time
import logging
import traceback
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QColor, QPixmap
from .config import Config
from .infrastructure.app_paths import AppPaths
from .migration.legacy_chroma import inspect_legacy_chroma
from .modern_context import initialize_modern_runtime

# Log initialization steps
log_dir = AppPaths.discover().logs_dir
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'initialization_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Initialization")

class InitializationWorker(QThread):
    progress_signal = Signal(str, int)
    finished_signal = Signal(tuple)
    
    def __init__(self, password):
        super().__init__()
        self.password = password
        logger.debug("InitializationWorker created")
        
    def run(self):
        logger.debug("InitializationWorker.run() started")
        config = None
        try:
            # Initialize config
            logger.debug("Starting config initialization")
            self.progress_signal.emit("Loading configuration...", 30)
            try:
                config = Config(self.password)
                logger.debug("Config initialized successfully")
            except Exception as config_error:
                logger.error(f"Config initialization failed: {str(config_error)}")
                logger.error(traceback.format_exc())
                raise

            # Bring the modern local-first sidecar up without blocking on Ollama.
            logger.debug("Starting modern sidecar initialization")
            self.progress_signal.emit("Preparing local indexing database...", 45)
            modern_runtime = initialize_modern_runtime(
                config.paths,
                config,
                self.password,
            )
            logger.debug("Modern sidecar initialization completed")
            modern_factory = getattr(modern_runtime, "session_factory", None)
            db_session = modern_factory() if modern_factory is not None else None
            if db_session is None:
                unavailable_reason = getattr(
                    modern_runtime,
                    "unavailable_reason",
                    "The modern database is unavailable.",
                )
                logger.error(
                    "Modern database initialization unavailable: %s",
                    unavailable_reason,
                )

            # Inventory legacy search data without deleting or reusing it.
            self.progress_signal.emit(
                "Checking legacy migration inventory...", 52
            )
            legacy_chroma = getattr(config.paths, "legacy_chroma", None)
            if legacy_chroma is not None:
                try:
                    inventory = inspect_legacy_chroma(legacy_chroma)
                    setattr(
                        modern_runtime,
                        "legacy_chroma_inventory",
                        inventory,
                    )
                    logger.debug(
                        "Legacy Chroma inventory completed (paths=%d, warnings=%d).",
                        len(inventory.paths),
                        len(inventory.warnings),
                    )
                except Exception as inventory_error:
                    logger.warning(
                        "Legacy Chroma inventory failed (%s).",
                        inventory_error.__class__.__name__,
                    )

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
            self.finished_signal.emit(
                (db_session, config, modern_runtime)
            )
            
        except Exception as e:
            logger.error(f"Initialization failed with error: {str(e)}")
            logger.error(traceback.format_exc())
            self.progress_signal.emit(f"Error: {str(e)}", 100)
            self.finished_signal.emit((None, config, None))

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