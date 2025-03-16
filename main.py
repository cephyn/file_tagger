import sys
import os
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTreeView, QListWidget, QPushButton, 
                           QInputDialog, QColorDialog, QLabel, QFileSystemModel,
                           QMessageBox, QLineEdit, QRadioButton, QButtonGroup,
                           QComboBox, QHeaderView, QMenuBar, QMenu, QDialog, QFileDialog,
                           QTabWidget, QProgressDialog, QSplashScreen, QProgressBar,
                           QTextEdit, QListWidgetItem, QScrollArea, QFrame)
from PySide6.QtCore import Qt, QDir, QStorageInfo, QThread, Signal
from PySide6.QtGui import QColor, QPixmap, QFont
from sqlalchemy import and_, or_
from models import init_db, File, Tag
from config import Config
from vector_search import VectorSearch
from api_settings import APISettingsDialog
from password_management import PasswordManagementDialog
from tag_suggestion import TagSuggestionDialog
from ai_service import AIService

# Worker thread to handle initialization tasks
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
            vector_search = VectorSearch(db_session)
            
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

class LoginScreen(QWidget):
    """A combined login and splash screen that handles password input and shows loading progress."""
    login_successful_signal = Signal(str)  # Signal to emit when login is successful
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("File Tagger - Login")
        self.setFixedSize(450, 400)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)  # Keep window on top but allow interaction
        
        # Store reference to the worker thread to prevent premature destruction
        self.worker = None
        
        # Set up the main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # App title
        title_label = QLabel("File Tagger")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # App subtitle
        subtitle_label = QLabel("File Management & Tagging System")
        subtitle_label.setStyleSheet("font-size: 14px; color: #666666;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)
        
        # Add spacer
        main_layout.addSpacing(20)
        
        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #cccccc;")
        main_layout.addWidget(separator)
        
        # Password section
        password_layout = QVBoxLayout()
        password_label = QLabel("Please enter your configuration password:")
        password_label.setStyleSheet("font-size: 12px;")
        password_layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.submit_password)
        password_layout.addWidget(self.password_input)
        
        self.login_button = QPushButton("Login")
        self.login_button.setStyleSheet("padding: 8px;")
        self.login_button.clicked.connect(self.submit_password)
        password_layout.addWidget(self.login_button)
        
        # Error message label (initially hidden)
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #ff0000; font-size: 12px;")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setVisible(False)
        password_layout.addWidget(self.error_label)
        
        main_layout.addLayout(password_layout)
        
        # Progress section (initially hidden)
        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
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
        
        main_layout.addWidget(self.progress_frame)
        self.progress_frame.setVisible(False)
        
        # Footer
        footer_label = QLabel("© 2023 File Tagger")
        footer_label.setStyleSheet("font-size: 10px; color: #999999;")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)
        
        # Set focus to password input
        self.password_input.setFocus()
    
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
        
        # Connect signals to update login screen
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_initialization_finished)
        
        # Start the worker
        self.worker.start()
    
    def on_initialization_finished(self, result):
        """Handle completion of the initialization process."""
        db_session, config, vector_search = result
        if db_session and config and vector_search:
            # Signal that login was successful, worker will be cleaned up in closeEvent
            self.hide()
            
            # Create and show the main window
            window = FileTagManager(db_session, config, vector_search)
            window.show()
            
            # Close the login screen after showing the main window
            self.close()
        else:
            self.show_error("Failed to initialize the application. Incorrect password or corrupted configuration.")
            # Clean up worker
            self.cleanup_worker()
    
    def cleanup_worker(self):
        """Clean up the worker thread safely."""
        if self.worker:
            # Disconnect signals to prevent potential issues
            self.worker.progress_signal.disconnect()
            self.worker.finished_signal.disconnect()
            
            # Wait for the thread to finish if it's still running
            if self.worker.isRunning():
                self.worker.wait()
            
            # Delete the worker
            self.worker = None
    
    def closeEvent(self, event):
        """Handle window close event to properly clean up thread resources."""
        self.cleanup_worker()
        event.accept()
    
    def show_error(self, message):
        """Display error message."""
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
        QApplication.processEvents()  # Ensure UI updates

class SplashScreen(QSplashScreen):
    def __init__(self):
        # Create a pixmap for the splash screen background
        # Using a simple colored background if no image is available
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

class ChatWithResultsDialog(QDialog):
    """Dialog for chatting with top search results using AI."""
    def __init__(self, parent, ai_service, top_results, query):
        super().__init__(parent)
        self.ai_service = ai_service
        self.top_results = top_results[:3] if len(top_results) >= 3 else top_results  # Limit to top 3 results
        self.initial_query = query
        self.chat_history = []
        
        self.setWindowTitle("Chat with Search Results")
        self.resize(800, 600)
        self.setup_ui()
        
        # Initialize chat with the search query
        self.start_chat()
    
    def setup_ui(self):
        """Set up the chat interface."""
        layout = QVBoxLayout()
        
        # Result files section
        results_layout = QHBoxLayout()
        results_layout.addWidget(QLabel("Chatting with:"))
        
        # Files being used
        for result in self.top_results:
            file_label = QLabel(os.path.basename(result['path']))
            file_label.setToolTip(result['path'])
            file_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
            results_layout.addWidget(file_label)
        
        layout.addLayout(results_layout)
        
        # Chat history area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.setMaximumHeight(80)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        
        self.setLayout(layout)
    
    def start_chat(self):
        """Initialize the chat with the AI."""
        # Prepare context from the top results
        context = "I'm going to ask questions about the following documents:\n\n"
        
        for i, result in enumerate(self.top_results):
            context += f"Document {i+1}: {os.path.basename(result['path'])}\n"
            context += f"Content snippets:\n"
            for snippet in result.get('snippets', []):
                context += f"- {snippet}\n"
            context += "\n"
        
        # Add the initial query as system message
        system_message = (
            f"You are an assistant helping with questions about documents. "
            f"Base your answers only on the document content snippets provided. "
            f"If the answer is not in the documents, say that you don't know based on the available information."
        )
        
        # Display a welcome message
        self.chat_display.append("<b>AI Assistant:</b> Hello! I can answer questions about the top search results. What would you like to know?")
        
        # Store context and system message in chat history
        self.chat_history = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": context}
        ]
        
        # Process the initial query if there is one
        if self.initial_query:
            self.chat_input.setPlainText(self.initial_query)
            self.send_message()
    
    def send_message(self):
        """Send the user message and get AI response."""
        user_message = self.chat_input.toPlainText().strip()
        if not user_message:
            return
        
        # Display user message
        self.chat_display.append(f"<b>You:</b> {user_message}")
        
        # Add to chat history
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Clear input
        self.chat_input.clear()
        
        # Show typing indicator
        self.chat_display.append("<b>AI Assistant:</b> <i>Thinking...</i>")
        QApplication.processEvents()
        
        # Get AI response
        try:
            # Prepare prompt from chat history
            prompt = self.prepare_prompt()
            
            # Call the AI service
            response = self.get_ai_response(prompt)
            
            # Update the last "typing" message with the actual response
            current_html = self.chat_display.toHtml()
            current_html = current_html.replace("<b>AI Assistant:</b> <i>Thinking...</i>", f"<b>AI Assistant:</b> {response}")
            self.chat_display.setHtml(current_html)
            
            # Add to chat history
            self.chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            # Update the "typing" message with error
            current_html = self.chat_display.toHtml()
            current_html = current_html.replace("<b>AI Assistant:</b> <i>Thinking...</i>", 
                                             f"<b>AI Assistant:</b> Sorry, I encountered an error: {str(e)}")
            self.chat_display.setHtml(current_html)
    
    def prepare_prompt(self) -> str:
        """Prepare the prompt from chat history."""
        prompt = ""
        
        # Add system and context messages first
        for message in self.chat_history:
            if message["role"] == "system":
                prompt += f"{message['content']}\n\n"
        
        # Add conversation history (excluding system messages)
        for message in self.chat_history:
            if message["role"] != "system":
                role_name = "User" if message["role"] == "user" else "Assistant"
                prompt += f"{role_name}: {message['content']}\n\n"
        
        # Add final prompt for the assistant to respond
        prompt += "Assistant: "
        
        return prompt
    
    def get_ai_response(self, prompt: str) -> str:
        """Get a response from the AI service."""
        # Get the AI provider name from the AI service
        provider = self.ai_service.provider
        
        if provider == 'openai':
            return self.get_openai_response(prompt)
        elif provider == 'anthropic':
            return self.get_claude_response(prompt)
        elif provider == 'gemini':
            return self.get_gemini_response(prompt)
        elif provider == 'local':
            return self.get_local_response(prompt)
        else:
            return f"Error: Unsupported AI provider '{provider}'"
    
    def get_openai_response(self, prompt: str) -> str:
        """Get a response from OpenAI."""
        response = self.ai_service.modules['openai'].ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.chat_history[0]["content"]},
                {"role": "system", "content": self.chat_history[1]["content"]},
                *[{"role": msg["role"], "content": msg["content"]} 
                  for msg in self.chat_history[2:] if msg["role"] != "system"]
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def get_claude_response(self, prompt: str) -> str:
        """Get a response from Anthropic Claude."""
        messages = []
        for msg in self.chat_history:
            if msg["role"] != "system":
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = self.ai_service.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system=self.chat_history[0]["content"] + "\n\n" + self.chat_history[1]["content"],
            messages=messages
        )
        return response.content[0].text
    
    def get_gemini_response(self, prompt: str) -> str:
        """Get a response from Google Gemini."""
        messages = []
        for msg in self.chat_history:
            if msg["role"] != "system":
                role = "user" if msg["role"] == "user" else "model"
                messages.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        chat = self.ai_service.model.start_chat(
            history=[],
            system_instruction=self.chat_history[0]["content"] + "\n\n" + self.chat_history[1]["content"],
        )
        
        for msg in messages:
            if msg["role"] == "user":
                chat.send_message(msg["parts"][0]["text"])
        
        response = chat.send_message(messages[-1]["parts"][0]["text"] if messages[-1]["role"] == "user" else "")
        return response.text
    
    def get_local_response(self, prompt: str) -> str:
        """Get a response from the local model."""
        if self.ai_service.local_model_type == 'llama':
            # For LlamaCpp
            full_prompt = f"System: {self.chat_history[0]['content']}\n\n{self.chat_history[1]['content']}\n\n" + prompt
            output = self.ai_service.model(
                full_prompt,
                max_tokens=1000,
                temperature=0.7,
                stop=["</s>", "User:", "user:"]
            )
            return output['choices'][0]['text']
        else:
            # For CTransformers
            full_prompt = f"System: {self.chat_history[0]['content']}\n\n{self.chat_history[1]['content']}\n\n" + prompt
            response = self.ai_service.model(
                full_prompt,
                max_new_tokens=1000,
                temperature=0.7,
                stop_sequences=["</s>", "User:", "user:"]
            )
            return response

class FileTagManager(QMainWindow):
    def __init__(self, db_session=None, config=None, vector_search=None):
        super().__init__()
        
        if (db_session and config and vector_search):
            # If these are provided, use them directly (from the initialization worker)
            self.db_session = db_session
            self.config = config
            self.vector_search = vector_search
        else:
            # Otherwise initialize normally (for backward compatibility)
            self.db_session = init_db()
            # Initialize config with a password dialog
            self.init_config()
            # Initialize vector search
            self.vector_search = VectorSearch(self.db_session)
            
        self.initUI()

    def init_config(self):
        password, ok = QInputDialog.getText(
            self, 'Configuration Password', 
            'Enter password to encrypt/decrypt settings:',
            QLineEdit.Password
        )
        if not ok:
            sys.exit(0)
        self.config = Config(password)
        
    def initUI(self):
        self.setWindowTitle('File Tagger')
        self.setGeometry(100, 100, 1200, 700)
        
        # Create menu bar
        menubar = self.menuBar()
        settings_menu = menubar.addMenu('Settings')
        
        # Add API Settings action
        api_settings_action = settings_menu.addAction('API Settings')
        api_settings_action.triggered.connect(self.show_api_settings)
        
        # Add Set Home Directory action
        home_dir_action = settings_menu.addAction('Set Home Directory')
        home_dir_action.triggered.connect(self.set_home_directory)
        
        # Add Password Management action
        password_action = settings_menu.addAction('Password Management')
        password_action.triggered.connect(self.show_password_management)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create the main tab widget that will contain tagging and search tabs
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)
        
        # Create widgets for both tabs
        tagging_tab = QWidget()
        search_tab = QWidget()
        
        # Add the tabs to the main tab widget
        self.main_tabs.addTab(tagging_tab, "Tagging Interface")
        self.main_tabs.addTab(search_tab, "Search Interface")
        
        # === TAGGING TAB ===
        tagging_layout = QHBoxLayout(tagging_tab)
        
        # File explorer section
        explorer_layout = QVBoxLayout()
        file_label = QLabel("File Explorer")
        explorer_layout.addWidget(file_label)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        # Drive selection
        self.drive_combo = QComboBox()
        
        home_btn = QPushButton("Home")
        home_btn.clicked.connect(self.go_home)
        up_btn = QPushButton("Up")
        up_btn.clicked.connect(self.go_up)
        
        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        
        nav_layout.addWidget(self.drive_combo)
        nav_layout.addWidget(home_btn)
        nav_layout.addWidget(up_btn)
        nav_layout.addWidget(self.path_display)
        
        explorer_layout.addLayout(nav_layout)
        
        # File system model and view
        self.model = QFileSystemModel()
        self.model.setRootPath("")  # Set empty string to allow access to all drives
        
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setSortingEnabled(True)
        self.tree.header().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.tree.header().setSectionsClickable(True)
        
        # Configure column resizing behavior
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.tree.header().setStretchLastSection(False)
        
        # Set initial column sizes
        self.tree.setColumnWidth(0, 250)  # Name column
        self.tree.setColumnWidth(1, 100)  # Size column
        self.tree.setColumnWidth(2, 100)  # Type column
        self.tree.setColumnWidth(3, 150)  # Date Modified column
        
        # Set minimum column widths to prevent columns from becoming too narrow
        self.tree.header().setMinimumSectionSize(50)
        
        # Connect signals
        self.tree.selectionModel().selectionChanged.connect(self.on_file_selected)
        self.tree.doubleClicked.connect(self.on_item_double_clicked)
        explorer_layout.addWidget(self.tree)
        
        # Tag management section
        tag_layout = QVBoxLayout()
        tag_label = QLabel("Tags")
        tag_layout.addWidget(tag_label)
        
        self.tag_list = QListWidget()
        self.tag_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # Enable multi-selection
        self.tag_list.itemSelectionChanged.connect(self.on_tag_selected)
        tag_layout.addWidget(self.tag_list)
        
        tag_buttons = QHBoxLayout()
        add_tag_btn = QPushButton("Add Tag")
        add_tag_btn.clicked.connect(self.add_tag)
        edit_tag_btn = QPushButton("Edit Tag")
        edit_tag_btn.clicked.connect(self.edit_tag)
        delete_tag_btn = QPushButton("Delete Tag")
        delete_tag_btn.clicked.connect(self.delete_tag)
        
        tag_buttons.addWidget(add_tag_btn)
        tag_buttons.addWidget(edit_tag_btn)
        tag_buttons.addWidget(delete_tag_btn)
        tag_layout.addLayout(tag_buttons)
        
        # File tags section
        file_tags_layout = QVBoxLayout()
        file_tags_label = QLabel("File Tags")
        file_tags_layout.addWidget(file_tags_label)
        
        self.file_tags_list = QListWidget()
        self.file_tags_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # Enable multi-selection
        file_tags_layout.addWidget(self.file_tags_list)
        
        file_tag_buttons = QHBoxLayout()
        add_file_tag_btn = QPushButton("Add Tag to File")
        add_file_tag_btn.clicked.connect(self.add_tag_to_file)
        remove_file_tag_btn = QPushButton("Remove Tag from File")
        remove_file_tag_btn.clicked.connect(self.remove_tag_from_file)
        suggest_tags_btn = QPushButton("Suggest Tags (AI)")
        suggest_tags_btn.clicked.connect(self.suggest_tags)
        
        file_tag_buttons.addWidget(add_file_tag_btn)
        file_tag_buttons.addWidget(remove_file_tag_btn)
        file_tag_buttons.addWidget(suggest_tags_btn)
        file_tags_layout.addLayout(file_tag_buttons)
        
        # Add the layouts to the tagging tab
        tagging_layout.addLayout(explorer_layout, stretch=2)
        tagging_layout.addLayout(tag_layout, stretch=1)
        tagging_layout.addLayout(file_tags_layout, stretch=1)
        
        # === SEARCH TAB ===
        search_layout = QVBoxLayout(search_tab)
        
        # Create tab widget for different search types
        search_tabs = QTabWidget()
        
        # Tag search tab
        tag_search_tab = QWidget()
        tag_search_layout = QVBoxLayout(tag_search_tab)
        
        # Tag selection section for search
        tag_select_layout = QVBoxLayout()
        tag_select_label = QLabel("Select Tags to Search:")
        tag_select_layout.addWidget(tag_select_label)
        
        self.search_tag_list = QListWidget()
        self.search_tag_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        tag_select_layout.addWidget(self.search_tag_list)
        
        # Search controls
        search_controls = QHBoxLayout()
        self.and_radio = QRadioButton("AND")
        self.or_radio = QRadioButton("OR")
        self.and_radio.setChecked(True)
        search_controls.addWidget(self.and_radio)
        search_controls.addWidget(self.or_radio)
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_by_tags)
        search_controls.addWidget(search_btn)
        tag_select_layout.addLayout(search_controls)
        
        # Tag search results section
        results_layout = QVBoxLayout()
        results_label = QLabel("Search Results:")
        results_layout.addWidget(results_label)
        
        self.search_results = QListWidget()
        self.search_results.itemDoubleClicked.connect(self.on_search_result_double_clicked)
        self.search_results.setContextMenuPolicy(Qt.CustomContextMenu)
        self.search_results.customContextMenuRequested.connect(self.on_search_result_right_clicked)
        results_layout.addWidget(self.search_results)
        
        # Add layouts to tag search tab
        tag_search_layout.addLayout(tag_select_layout)
        tag_search_layout.addLayout(results_layout)
        
        # RAG search tab
        rag_search_tab = QWidget()
        rag_search_layout = QVBoxLayout(rag_search_tab)
        
        # Query input
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Query:"))
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your search query...")
        query_layout.addWidget(self.query_input)
        
        rag_search_btn = QPushButton("Search")
        rag_search_btn.clicked.connect(self.search_by_content)
        query_layout.addWidget(rag_search_btn)
        
        rag_search_layout.addLayout(query_layout)
        
        # Tag filter section
        filter_layout = QVBoxLayout()
        filter_layout.addWidget(QLabel("Filter by tags:"))
        
        filter_controls = QHBoxLayout()
        self.rag_and_radio = QRadioButton("AND")
        self.rag_or_radio = QRadioButton("OR")
        self.rag_and_radio.setChecked(True)
        filter_controls.addWidget(self.rag_and_radio)
        filter_controls.addWidget(self.rag_or_radio)
        filter_layout.addLayout(filter_controls)
        
        # Tag filter list
        self.tag_filter_list = QListWidget()
        self.tag_filter_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        filter_layout.addWidget(self.tag_filter_list)
        
        # RAG search results
        rag_results_layout = QVBoxLayout()
        rag_results_layout.addWidget(QLabel("Search Results:"))
        
        self.rag_search_results = QListWidget()
        self.rag_search_results.itemDoubleClicked.connect(self.on_search_result_double_clicked)
        self.rag_search_results.setContextMenuPolicy(Qt.CustomContextMenu)
        self.rag_search_results.customContextMenuRequested.connect(self.on_search_result_right_clicked)
        rag_results_layout.addWidget(self.rag_search_results)
        
        # Index management button
        reindex_btn = QPushButton("Reindex Files")
        reindex_btn.clicked.connect(self.reindex_files)
        rag_results_layout.addWidget(reindex_btn)
        
        # Add layouts to RAG search tab
        rag_search_layout.addLayout(filter_layout)
        rag_search_layout.addLayout(rag_results_layout)
        
        # Add tabs
        search_tabs.addTab(tag_search_tab, "Tag Search")
        search_tabs.addTab(rag_search_tab, "Content Search")
        search_layout.addWidget(search_tabs)
        
        # Set initial directory from config
        initial_path = self.config.get_home_directory()
        self.tree.setRootIndex(self.model.index(initial_path))
        
        # Update initial path display
        self.path_display.setText(initial_path)
        
        # First populate the drive list
        self.update_drive_list()
        
        # Now set the correct drive based on initial path
        drive = os.path.splitdrive(initial_path)[0] + os.path.sep
        for i in range(self.drive_combo.count()):
            if self.drive_combo.itemData(i).startswith(drive):
                self.drive_combo.setCurrentIndex(i)
                break
        
        # Connect drive combo change event after setting the initial value
        # to avoid triggering it during initialization
        self.drive_combo.currentIndexChanged.connect(self.on_drive_changed)

        self.refresh_tags()
        self.current_file_path = None

    def refresh_tags(self):
        # Update all tag lists
        self.tag_list.clear()
        self.tag_filter_list.clear()
        self.search_tag_list.clear()
        tags = self.db_session.query(Tag).all()
        for tag in tags:
            # Get tag color and determine appropriate text color
            tag_color = QColor(tag.color)
            text_color = Qt.white if self._is_dark_color(tag_color) else Qt.black
            
            # Add to main tag list in tagging tab
            self.tag_list.addItem(tag.name)
            item = self.tag_list.item(self.tag_list.count() - 1)
            item.setBackground(tag_color)
            item.setForeground(text_color)
            
            # Add to filter list in RAG search
            self.tag_filter_list.addItem(tag.name)
            item = self.tag_filter_list.item(self.tag_filter_list.count() - 1)
            item.setBackground(tag_color)
            item.setForeground(text_color)
            
            # Add to search tag list in tag search
            self.search_tag_list.addItem(tag.name)
            item = self.search_tag_list.item(self.search_tag_list.count() - 1)
            item.setBackground(tag_color)
            item.setForeground(text_color)
            
    def search_by_content(self):
        """Perform RAG-based search with optional tag filtering."""
        query = self.query_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a search query!")
            return
            
        # Debug: Check Shadowdark file specifically
        self.vector_search.debug_check_file("E:/Google Drive/RPG Stuff/Arcane Library/Shadowdark/Shadowdark_RPG_-_V1.pdf")
            
        # Get selected tags for filtering
        selected_tags = [item.text() for item in self.tag_filter_list.selectedItems()]
        use_and = self.rag_and_radio.isChecked()
        
        # Perform search
        results = self.vector_search.search(
            query=query,
            tag_filter=selected_tags if selected_tags else None,
            use_and=use_and
        )
        
        # Clear previous results
        self.rag_search_results.clear()
        
        # Check if results are empty
        if not results:
            self.rag_search_results.addItem("No matching files found")
            return
            
        # Store results for the chat feature
        self._last_search_results = results
        self._last_search_query = query
        
        # Add a "Chat with Results" button at the top of results if we have at least one result
        chat_item = QListWidgetItem("💬 Chat with top results using AI")
        chat_item.setBackground(QColor(230, 240, 255))  # Light blue background
        chat_item.setFont(QFont("Arial", 10, QFont.Bold))
        chat_item.setData(Qt.UserRole, "chat_action")  # Special data to identify this item
        self.rag_search_results.addItem(chat_item)
        
        for result in results:
            # Create result text with score and tags
            score_percent = int(result['score'] * 100)
            tags_text = f" [Tags: {', '.join(result.get('tags', []))}]"
            
            # Create the main result item with filename and score
            display_text = f"{os.path.basename(result['path'])} (Match: {score_percent}%){tags_text}"
            
            self.rag_search_results.addItem(display_text)
            list_item = self.rag_search_results.item(self.rag_search_results.count() - 1)
            
            # Store full path in tooltip
            list_item.setToolTip(result['path'])
            
            # Color code based on match score
            color = self._get_score_color(result['score'])
            list_item.setBackground(color)
            
            # Add snippets as separate items below the main result, indented
            if 'snippets' in result and result['snippets']:
                for i, snippet in enumerate(result['snippets']):
                    # Clean up the snippet for display
                    clean_snippet = snippet.replace('\n', ' ').strip()
                    if len(clean_snippet) > 200:
                        clean_snippet = f"{clean_snippet[:200]}..."
                    
                    # Add snippet with indentation
                    self.rag_search_results.addItem(f"    ↪ {clean_snippet}")
                    snippet_item = self.rag_search_results.item(self.rag_search_results.count() - 1)
                    
                    # Use a lighter shade of the same color for snippets
                    snippet_color = QColor(color)
                    snippet_color.setAlpha(100)  # Make it more transparent
                    snippet_item.setBackground(snippet_color)
                    
                    # Allow selecting the main item when clicking on a snippet
                    snippet_item.setToolTip(result['path'])
        
        # Connect item click signal for the chat action
        self.rag_search_results.itemClicked.connect(self.on_rag_result_clicked)
    
    def _get_score_color(self, score: float) -> QColor:
        """Get a color representing the match score (red to green)."""
        if score >= 0.8:
            return QColor(200, 255, 200)  # Light green
        elif score >= 0.6:
            return QColor(255, 255, 200)  # Light yellow
        elif score >= 0.4:
            return QColor(255, 230, 200)  # Light orange
        else:
            return QColor(255, 200, 200)  # Light red
            
    def reindex_files(self):
        """Reindex all files in the database."""
        progress = QProgressDialog("Reindexing files...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        
        def update_progress(status: str, value: int):
            if progress.wasCanceled():
                return False
            progress.setLabelText(status)
            progress.setValue(value)
            QApplication.processEvents()
            return True
            
        try:
            self.vector_search.reindex_all_files(progress_callback=update_progress)
            if not progress.wasCanceled():
                # Fix metadata after reindexing
                progress.setLabelText("Fixing metadata...")
                self.vector_search.fix_all_metadata()
                QMessageBox.information(self, "Success", "File reindexing complete!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error during reindexing: {str(e)}")
        finally:
            progress.close()
            
    def add_tag(self):
        name, ok = QInputDialog.getText(self, 'Add Tag', 'Enter tag name:')
        if ok and name:
            color = QColorDialog.getColor()
            if color.isValid():
                tag = Tag(name=name, color=color.name())
                try:
                    self.db_session.add(tag)
                    self.db_session.commit()
                    self.refresh_tags()
                except:
                    self.db_session.rollback()
                    QMessageBox.warning(self, "Error", "Tag name must be unique!")
                    
    def edit_tag(self):
        current_item = self.tag_list.currentItem()
        if current_item:
            tag = self.db_session.query(Tag).filter_by(name=current_item.text()).first()
            if tag:
                name, ok = QInputDialog.getText(self, 'Edit Tag', 'Enter new tag name:', text=tag.name)
                if ok and name:
                    color = QColorDialog.getColor(initial=QColor(tag.color))
                    if color.isValid():
                        tag.name = name
                        tag.color = color.name()
                        try:
                            self.db_session.commit()
                            self.refresh_tags()
                            self.refresh_file_tags()
                        except:
                            self.db_session.rollback()
                            QMessageBox.warning(self, "Error", "Tag name must be unique!")
                            
    def delete_tag(self):
        current_item = self.tag_list.currentItem()
        if current_item:
            # tag = self.db_session.query(Tag).filter_by(name=current_item.text()).first()
            tag = self.db_session.query(Tag).filter_by(name=current_item.text()).first()
            if tag:
                self.db_session.delete(tag)
                self.db_session.commit()
                self.refresh_tags()
                self.refresh_file_tags()
                
    def add_tag_to_file(self):
        if not self.current_file_path:
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return
            
        selected_items = self.tag_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select one or more tags to add!")
            return
            
        file_obj = self.db_session.query(File).filter_by(path=self.current_file_path).first()
        if not file_obj:
            file_obj = File(path=self.current_file_path)
            self.db_session.add(file_obj)
            
        for selected_item in selected_items:
            tag = self.db_session.query(Tag).filter_by(name=selected_item.text()).first()
            if tag and tag not in file_obj.tags:
                file_obj.tags.append(tag)
        
        self.db_session.commit()
        self.refresh_file_tags()
        # Update vector search index for the file
        if self.current_file_path and os.path.exists(self.current_file_path):
            content = self.vector_search._extract_file_content(self.current_file_path)
            if content:
                self.vector_search.index_file(self.current_file_path, content)
                
    def remove_tag_from_file(self):
        if not self.current_file_path:
            return
            
        selected_items = self.file_tags_list.selectedItems()
        if selected_items:
            file_obj = self.db_session.query(File).filter_by(path=self.current_file_path).first()
            if file_obj:
                for selected_item in selected_items:
                    tag = self.db_session.query(Tag).filter_by(name=selected_item.text()).first()
                    if tag:
                        file_obj.tags.remove(tag)
                self.db_session.commit()
                self.refresh_file_tags()
                # Update vector search index for the file
                if self.current_file_path and os.path.exists(self.current_file_path):
                    content = self.vector_search._extract_file_content(self.current_file_path)
                    if content:
                        self.vector_search.index_file(self.current_file_path, content)

    def on_file_selected(self, current, previous):
        """Handle file selection changes in the tree view."""
        if current.indexes():
            self.current_file_path = self.model.filePath(current.indexes()[0])
            self.refresh_file_tags()
            
    def refresh_file_tags(self):
        """Update the file tags list for the currently selected file."""
        self.file_tags_list.clear()
        if not self.current_file_path:
            return
            
        file_obj = self.db_session.query(File).filter_by(path=self.current_file_path).first()
        if file_obj:
            for tag in file_obj.tags:
                # Get tag color and determine appropriate text color
                tag_color = QColor(tag.color)
                text_color = Qt.white if self._is_dark_color(tag_color) else Qt.black
                
                self.file_tags_list.addItem(tag.name)
                item = self.file_tags_list.item(self.file_tags_list.count() - 1)
                item.setBackground(tag_color)
                item.setForeground(text_color)

    def update_drive_list(self):
        self.drive_combo.clear()
        # Get all available drives
        for drive in QStorageInfo.mountedVolumes():
            # Skip unmounted volumes
            if not drive.isValid() or not drive.isReady():
                continue
            # Get drive name and path
            name = drive.displayName()
            root_path = drive.rootPath()
            # Add drive to combo box with both name and path
            self.drive_combo.addItem(f"{name} ({root_path})", root_path)

    def on_drive_changed(self, index):
        if index >= 0:
            drive_path = self.drive_combo.itemData(index)
            # Update model and view for the new drive
            model_index = self.model.index(drive_path)
            self.tree.setRootIndex(model_index)
            # Ensure sorting is maintained after drive change
            current_column = self.tree.header().sortIndicatorSection()
            current_order = self.tree.header().sortIndicatorOrder()
            self.tree.sortByColumn(current_column, current_order)
            self.path_display.setText(drive_path)

    def go_home(self):
        """Navigate to the configured home directory."""
        home_path = self.config.get_home_directory()
        self.tree.setRootIndex(self.model.index(home_path))
        self.path_display.setText(home_path)
        # Select the drive containing home directory
        home_drive = os.path.splitdrive(home_path)[0] + os.path.sep
        for i in range(self.drive_combo.count()):
            if self.drive_combo.itemData(i).startswith(home_drive):
                self.drive_combo.setCurrentIndex(i)
                break

    def go_up(self):
        current_path = self.path_display.text()
        parent_path = os.path.dirname(current_path)
        if os.path.exists(parent_path):
            # Check if we're at the root of a drive
            if os.path.splitdrive(current_path)[1] == os.path.sep:
                return  # Don't go up from root of drive
            self.tree.setRootIndex(self.model.index(parent_path))
            self.path_display.setText(parent_path)

    def search_by_tags(self):
        selected_items = self.search_tag_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select at least one tag to search!")
            return
            
        # Get selected tag names
        tag_names = [item.text() for item in selected_items]
        
        # Query tags first
        tags = self.db_session.query(Tag).filter(Tag.name.in_(tag_names)).all()
        if not tags:
            return
            
        # Build the query based on boolean operator
        files_query = self.db_session.query(File)
        
        if self.and_radio.isChecked():
            # AND logic: files must have all selected tags
            for tag in tags:
                files_query = files_query.filter(File.tags.contains(tag))
        else:
            # OR logic: files must have any of the selected tags
            files_query = files_query.filter(File.tags.any(Tag.id.in_([t.id for t in tags])))

        # Get results and display them
        self.search_results.clear()
        files = files_query.all()

        for file in files:
            item = self.search_results.addItem(os.path.basename(file.path))
            # Store full path as item data
            self.search_results.item(self.search_results.count() - 1).setToolTip(file.path)

            # Add tag information to the item display
            tag_text = " [Tags: " + ", ".join(tag.name for tag in file.tags) + "]"
            self.search_results.item(self.search_results.count() - 1).setText(
                os.path.basename(file.path) + tag_text
            )

        if not files:
            self.search_results.addItem("No files found with the selected tags")

    def on_tag_selected(self):
        """Handle tag selection changes."""
        # This method is triggered when tag selection changes
        # Currently used just for enabling/disabling tag operations
        selected_items = self.tag_list.selectedItems()
        # Could add additional functionality here if needed

    def on_search_result_double_clicked(self, item):
        file_path = item.toolTip()  # Get the full path from tooltip
        if os.path.exists(file_path):
            # If it's a file, open it with the default application
            if os.path.isfile(file_path):
                # Use the operating system's default program to open the file
                import subprocess
                try:
                    os.startfile(file_path) if sys.platform == 'win32' else subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', file_path))
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not open file: {str(e)}")
                return
                
            # For directories, navigate to them in the tree view
            # Get the drive path and select it in the combo box
            drive = os.path.splitdrive(file_path)[0] + os.path.sep
            for i in range(self.drive_combo.count()):
                if self.drive_combo.itemData(i).startswith(drive):
                    self.drive_combo.setCurrentIndex(i)
                    break
                    
            # Navigate to the directory containing the file
            dir_path = os.path.dirname(file_path) if os.path.isfile(file_path) else file_path
            self.tree.setRootIndex(self.model.index(dir_path))
            self.path_display.setText(dir_path)
            
            # Select the file in the tree view
            self.tree.setCurrentIndex(self.model.index(file_path))
            self.tree.scrollTo(self.model.index(file_path))

    def on_item_double_clicked(self, index):
        """Handle double-click on tree view items."""
        path = self.model.filePath(index)
        if os.path.isdir(path):
            self.tree.setRootIndex(index)
            self.path_display.setText(path)
        else:
            # If it's a file, open it with the default application
            import subprocess
            try:
                os.startfile(path) if sys.platform == 'win32' else subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', path))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open file: {str(e)}")

    def show_api_settings(self):
        dialog = APISettingsDialog(self.config, self)
        dialog.exec()

    def show_password_management(self):
        dialog = PasswordManagementDialog(self.config, self)
        dialog.exec()

    def suggest_tags(self):
        """Open the tag suggestion dialog for the current file."""
        if not self.current_file_path:
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return
            
        if os.path.isdir(self.current_file_path):
            QMessageBox.warning(self, "Error", "Please select a file, not a directory!")
            return
            
        dialog = TagSuggestionDialog(self.config, self.db_session, self.current_file_path, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_tags()
            self.refresh_file_tags()

    def set_home_directory(self):
        """Open dialog to set home directory."""
        current_dir = self.config.get_home_directory()
        new_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Home Directory",
            current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if new_dir:
            if self.config.set_home_directory(new_dir):
                QMessageBox.information(
                    self,
                    "Success",
                    "Home directory updated successfully.\nNew location will be used next time you start the application."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Failed to set home directory. Please ensure the selected directory exists and is accessible."
                )

    def on_search_result_right_clicked(self, position):
        """Handle right-click on search results items to show context menu."""
        # Get the item that was clicked
        item = self.sender().itemAt(position)
        if not item:
            return
            
        # Create context menu
        context_menu = QMenu(self)
        
        # Get file path from item tooltip
        file_path = item.toolTip()
        if not file_path or not os.path.exists(file_path):
            return
            
        # Add open action
        open_action = context_menu.addAction("Open File")
        open_in_folder_action = context_menu.addAction("Open Containing Folder")
        
        # Add remove from vector DB action
        remove_action = context_menu.addAction("Remove from Search Index")
        
        # Show the context menu
        action = context_menu.exec_(self.sender().mapToGlobal(position))
        
        # Handle menu actions
        if action == open_action:
            # Open the file with default application
            self._open_file(file_path)
        elif action == open_in_folder_action:
            # Open the containing folder and select the file
            self._open_containing_folder(file_path)
        elif action == remove_action:
            # Remove the file from vector database
            self._remove_from_vector_db(file_path)
            
    def _open_file(self, file_path):
        """Open a file with the system's default application."""
        if os.path.isfile(file_path):
            import subprocess
            try:
                os.startfile(file_path) if sys.platform == 'win32' else subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', file_path))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open file: {str(e)}")
                
    def _open_containing_folder(self, file_path):
        """Open the folder containing the file and select it."""
        if os.path.exists(file_path):
            dir_path = os.path.dirname(file_path) if os.path.isfile(file_path) else file_path
            import subprocess
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
                QMessageBox.warning(self, "Error", f"Could not open containing folder: {str(e)}")
                
    def _remove_from_vector_db(self, file_path):
        """Remove a file from the vector database."""
        # Ask for confirmation
        reply = QMessageBox.question(
            self, 
            "Remove from Search Index",
            f"Are you sure you want to remove\n{file_path}\nfrom the search index?\n\nNote: This only removes the file from the search index, not from your computer or the tag database.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove from vector database
            success = self.vector_search.remove_file(file_path)
            
            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"File removed from search index successfully.\n\nThe file is still on your computer and in the tag database."
                )
                
                # If the item is in the current search results, remove it
                self._remove_item_from_results(file_path)
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Could not remove file from search index. See console for details."
                )
                
    def _remove_item_from_results(self, file_path):
        """Remove items with the given file path from search results lists."""
        # Check which tab is active and remove from appropriate list
        current_tab = self.main_tabs.currentIndex()
        
        if current_tab == 1:  # Search tab
            search_tabs = self.main_tabs.widget(1).layout().itemAt(0).widget()
            current_search_tab = search_tabs.currentIndex()
            
            if current_search_tab == 0:  # Tag search tab
                # Remove from tag search results
                for i in range(self.search_results.count()):
                    item = self.search_results.item(i)
                    if item and item.toolTip() == file_path:
                        self.search_results.takeItem(i)
                        break
            elif current_search_tab == 1:  # RAG search tab
                # Remove from RAG search results
                for i in range(self.rag_search_results.count()):
                    item = self.rag_search_results.item(i)
                    if item and item.toolTip() == file_path:
                        self.rag_search_results.takeItem(i)
                        # Also remove any snippet items that follow
                        j = i + 1
                        while j < self.rag_search_results.count():
                            next_item = self.rag_search_results.item(j)
                            if next_item and next_item.text().startswith("    ↪"):
                                self.rag_search_results.takeItem(j)
                            else:
                                break
                        break

    def _is_dark_color(self, color):
        """
        Determines if a color is dark (needs white text) or light (needs dark text).
        Uses the luminance formula: 0.299*R + 0.587*G + 0.114*B
        
        Args:
            color (QColor): The color to check
            
        Returns:
            bool: True if the color is dark, False if it's light
        """
        if isinstance(color, str):
            color = QColor(color)
            
        # Calculate luminance (perceived brightness)
        luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) / 255
        
        # If luminance is less than 0.5, color is dark
        return luminance < 0.5

    def on_rag_result_clicked(self, item):
        """Handle click events on RAG search results."""
        # Check if this is the "Chat with Results" button
        if item.data(Qt.UserRole) == "chat_action":
            # Check if we have search results and an AI service available
            if not hasattr(self, '_last_search_results') or not self._last_search_results:
                QMessageBox.warning(self, "Error", "No search results available to chat with.")
                return
                
            # Get the API config
            provider = self.config.get_selected_provider()
            api_key = self.config.get_api_key(provider)  # Now correctly passing the provider
            local_model_path = self.config.get_local_model_path()
            local_model_type = self.config.get_local_model_type()
            
            if not api_key and provider != 'local':
                QMessageBox.warning(self, "Error", "AI service not configured. Please set up API credentials in Settings.")
                return
            
            if provider == 'local' and not local_model_path:
                QMessageBox.warning(self, "Error", "Local model not configured. Please set up the model path in Settings.")
                return
            
            # Initialize AI service
            try:
                ai_service = AIService(
                    provider=provider,
                    api_key=api_key,
                    db_session=self.db_session,
                    local_model_path=local_model_path,
                    local_model_type=local_model_type
                )
                
                # Open chat dialog with top results
                chat_dialog = ChatWithResultsDialog(
                    self,
                    ai_service=ai_service,
                    top_results=self._last_search_results[:3],  # Use top 3 results
                    query=self._last_search_query
                )
                chat_dialog.exec()
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to initialize AI service: {str(e)}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show the login screen
    # The login screen now handles all thread management internally
    login_screen = LoginScreen()
    login_screen.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()