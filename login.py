from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QFrame, QProgressBar)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from initialization import InitializationWorker

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
        footer_label = QLabel("© 2023 File Tagger")
        footer_label.setStyleSheet("font-size: 10px; color: #999999;")
        footer_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer_label)
    
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
        db_session, config, vector_search = result
        if db_session and config and vector_search:
            self.hide()
            
            # Import here to avoid circular imports
            from file_tag_manager import FileTagManager
            
            # Create and show the main window
            window = FileTagManager(db_session, config, vector_search)
            window.show()
            
            # Close the login screen
            self.close()
        else:
            self.show_error("Failed to initialize the application. Incorrect password or corrupted configuration.")
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