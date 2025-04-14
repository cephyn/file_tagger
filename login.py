from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QFrame, QProgressBar, QDialog)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import logging
import traceback
import os
from initialization import InitializationWorker

# Configure additional logging for login process
logger = logging.getLogger("Login")
# Make sure we have a logs directory
os.makedirs('logs', exist_ok=True)
# Add a file handler for login-specific issues
login_log_handler = logging.FileHandler('login_debug.log')
login_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(login_log_handler)
logger.addHandler(logging.StreamHandler())

class AboutDialog(QDialog):
    """Dialog showing information about the application."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("About File Tagger")
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # App title
        title = QLabel("File Tagger")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Version
        version = QLabel("Version 1.0.0")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)
        
        # Description
        desc = QLabel("A feature-rich file management and tagging system with AI-powered search capabilities.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Author info
        author = QLabel("Developed by: Busy Wyvern")
        author.setAlignment(Qt.AlignCenter)
        layout.addWidget(author)
        
        # Website link (clickable)
        website_label = QLabel("<a href='https://busywyvern.com'>Visit Website</a>")
        website_label.setOpenExternalLinks(True)
        website_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(website_label)
        
        # Technologies used
        tech_label = QLabel("Built with Python, PySide6, SQLAlchemy, and Vector Search")
        tech_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(tech_label)
        
        # Copyright
        copyright_label = QLabel("© 2023 All Rights Reserved")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

class LoginScreen(QWidget):
    """A combined login and splash screen that handles password input and shows loading progress."""
    login_successful_signal = Signal(str)  # Signal to emit when login is successful
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("File Tagger - Login")
        self.setFixedSize(450, 400)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
        # Store reference to the worker thread to prevent premature destruction
        self.worker = None
        
        # Set up the main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        self._setup_header(main_layout)
        self._setup_password_section(main_layout)
        self._setup_progress_section(main_layout)
        self._setup_footer(main_layout)
        
        # Set focus to password input
        self.password_input.setFocus()
    
    def _setup_header(self, layout):
        # App title
        title_label = QLabel("File Tagger")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # App subtitle
        subtitle_label = QLabel("File Management & Tagging System")
        subtitle_label.setStyleSheet("font-size: 14px; color: #666666;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)
        
        # Add spacer and separator
        layout.addSpacing(20)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #cccccc;")
        layout.addWidget(separator)
    
    def _setup_password_section(self, layout):
        password_layout = QVBoxLayout()
        
        # Password label
        password_label = QLabel("Please enter your configuration password:")
        password_label.setStyleSheet("font-size: 12px;")
        password_layout.addWidget(password_label)
        
        # Password input
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.submit_password)
        password_layout.addWidget(self.password_input)
        
        # Login button
        self.login_button = QPushButton("Login")
        self.login_button.setStyleSheet("padding: 8px;")
        self.login_button.clicked.connect(self.submit_password)
        password_layout.addWidget(self.login_button)
        
        # Error message label
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #ff0000; font-size: 12px;")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setVisible(False)
        password_layout.addWidget(self.error_label)
        
        layout.addLayout(password_layout)
    
    def _setup_progress_section(self, layout):
        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                width: 10px;
                margin: 0px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.progress_frame)
        self.progress_frame.setVisible(False)
    
    def _setup_footer(self, layout):
        footer_layout = QHBoxLayout()
        
        # About button
        about_button = QPushButton("About")
        about_button.setStyleSheet("padding: 5px 10px;")
        about_button.clicked.connect(self.show_about_dialog)
        footer_layout.addWidget(about_button)
        
        # Copyright label
        footer_label = QLabel("© 2023 File Tagger")
        footer_label.setStyleSheet("font-size: 10px; color: #999999;")
        footer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        footer_layout.addWidget(footer_label)
        
        layout.addLayout(footer_layout)
    
    def show_about_dialog(self):
        """Show the About dialog."""
        dialog = AboutDialog(self)
        dialog.exec()
    
    def submit_password(self):
        """Handle password submission."""
        password = self.password_input.text()
        
        if not password:
            self.show_error("Please enter a password")
            return
        
        # Hide password section and show progress
        self.password_input.setEnabled(False)
        self.login_button.setEnabled(False)
        self.error_label.setVisible(False)
        self.progress_frame.setVisible(True)
        
        # Create and start the initialization worker
        self.worker = InitializationWorker(password)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_initialization_finished)
        self.worker.start()    
    def on_initialization_finished(self, result):
        """Handle completion of the initialization process."""
        logger.debug("on_initialization_finished called with result")
        try:
            db_session, config, vector_search = result
            logger.debug(f"Unpacked result: db_session={db_session is not None}, config={config is not None}, vector_search={vector_search is not None}")
            
            if db_session and config and vector_search:
                logger.debug("Initialization succeeded, preparing to show main window")
                try:
                    # Hide the login window
                    self.hide()
                    logger.debug("Login window hidden")
                    
                    # Import here to avoid circular imports
                    logger.debug("Importing FileTagManager")
                    from file_tag_manager import FileTagManager
                    logger.debug("FileTagManager imported successfully")
                    
                    # Create and show the main window
                    logger.debug("Creating FileTagManager instance")
                    window = FileTagManager(db_session, config, vector_search)
                    logger.debug("FileTagManager instance created")
                    
                    # Save a reference to the window to prevent it from being garbage collected
                    self._main_window = window
                    logger.debug("Main window reference saved")
                    
                    logger.debug("About to show main window")
                    window.show()
                    logger.debug("Main window shown successfully")
                    
                    # Set window attributes to ensure it stays alive
                    window.setAttribute(Qt.WA_DeleteOnClose, False)
                    logger.debug("Window attributes set to prevent automatic deletion")
                    
                    # Close the login screen
                    logger.debug("Closing login screen")
                    self.close()
                    logger.debug("Login screen closed")
                except Exception as e:
                    logger.error(f"Error during main window creation: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.show_error(f"Error launching application: {str(e)}")
                    self.cleanup_worker()
            else:
                logger.error("Initialization failed - missing required components")
                self.show_error("Failed to initialize the application. Incorrect password or corrupted configuration.")
                self.cleanup_worker()
        except Exception as e:
            logger.error(f"Unexpected error in on_initialization_finished: {str(e)}")
            logger.error(traceback.format_exc())
            self.show_error(f"Unexpected error: {str(e)}")
            self.cleanup_worker()
    
    def cleanup_worker(self):
        """Clean up the worker thread safely."""
        if self.worker:
            self.worker.progress_signal.disconnect()
            self.worker.finished_signal.disconnect()
            
            if self.worker.isRunning():
                self.worker.wait()
            
            self.worker = None
    
    def closeEvent(self, event):
        """Handle window close event to properly clean up thread resources."""
        self.cleanup_worker()
        event.accept()
    
    def show_error(self, message):
        """Display error message and reset input state."""
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        self.password_input.setEnabled(True)
        self.login_button.setEnabled(True)
        self.password_input.selectAll()
        self.password_input.setFocus()
    
    def update_progress(self, message, value):
        """Update progress bar and status message."""
        self.status_label.setText(message)
        self.progress_bar.setValue(value)