from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QComboBox, QMessageBox,
                            QFileDialog, QGroupBox, QRadioButton, QButtonGroup,
                            QTextEdit)
from PySide6.QtCore import Qt
from config import Config
import os

class APISettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('API Settings')
        self.setModal(True)
        layout = QVBoxLayout(self)
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_label = QLabel('AI Provider:')
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(['OpenAI', 'Gemini', 'Claude', 'Local Model'])
        
        # Set current provider
        current_provider = self.config.get_selected_provider()
        index_map = {'openai': 0, 'gemini': 1, 'anthropic': 2, 'local': 3}
        index = index_map.get(current_provider, 0)
        self.provider_combo.setCurrentIndex(index)
        
        self.provider_combo.currentIndexChanged.connect(self.on_provider_changed)
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_combo)
        
        # API key input (for cloud providers)
        self.api_key_group = QGroupBox("API Key Settings")
        api_key_layout = QVBoxLayout()
        
        key_layout = QHBoxLayout()
        key_label = QLabel('API Key:')
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setText(self.get_current_api_key())
        
        # Add toggle visibility button
        self.toggle_visibility_btn = QPushButton('👁')  # Eye emoji as icon
        self.toggle_visibility_btn.setFixedWidth(30)  # Make the button compact
        self.toggle_visibility_btn.setCheckable(True)  # Make it toggleable
        self.toggle_visibility_btn.clicked.connect(self.toggle_key_visibility)
        
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.key_input)
        key_layout.addWidget(self.toggle_visibility_btn)
        api_key_layout.addLayout(key_layout)
        self.api_key_group.setLayout(api_key_layout)
        
        # Local model settings
        self.local_model_group = QGroupBox("Local Model Settings")
        local_model_layout = QVBoxLayout()
        
        # Model path selection
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("Model Path:")
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(self.config.get_local_model_path())
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_input)
        model_path_layout.addWidget(browse_btn)
        local_model_layout.addLayout(model_path_layout)
        
        # Model type selection
        model_type_label = QLabel("Model Type:")
        local_model_layout.addWidget(model_type_label)
        
        model_type_group = QButtonGroup(self)
        self.llama_radio = QRadioButton("LLama.cpp")
        self.ctransformers_radio = QRadioButton("CTransformers (Gemma, GPT-J, etc.)")
        
        # Set the current model type
        current_model_type = self.config.get_local_model_type()
        if current_model_type == "ctransformers":
            self.ctransformers_radio.setChecked(True)
        else:
            self.llama_radio.setChecked(True)
            
        model_type_group.addButton(self.llama_radio)
        model_type_group.addButton(self.ctransformers_radio)
        
        local_model_layout.addWidget(self.llama_radio)
        local_model_layout.addWidget(self.ctransformers_radio)
        
        # Add help text
        help_text = QLabel(
            "Note: You need to install the appropriate package for your model:\n"
            "• For Llama models: pip install llama-cpp-python\n"
            "• For other models: pip install ctransformers"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #666; font-size: 10px;")
        local_model_layout.addWidget(help_text)
        
        self.local_model_group.setLayout(local_model_layout)
        
        # System message settings
        self.system_message_group = QGroupBox("AI System Message")
        system_message_layout = QVBoxLayout()
        
        # System message explanation
        system_message_info = QLabel(
            "This message instructs the AI how to behave when tagging files."
        )
        system_message_info.setWordWrap(True)
        system_message_layout.addWidget(system_message_info)
        
        # System message input
        self.system_message_input = QTextEdit()
        self.system_message_input.setPlainText(self.config.get_system_message())
        self.system_message_input.setMinimumHeight(100)
        system_message_layout.addWidget(self.system_message_input)
        
        # Reset button
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_system_message)
        system_message_layout.addWidget(reset_btn)
        
        self.system_message_group.setLayout(system_message_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton('Save')
        save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        
        # Add all layouts
        layout.addLayout(provider_layout)
        layout.addWidget(self.api_key_group)
        layout.addWidget(self.local_model_group)
        layout.addWidget(self.system_message_group)
        layout.addLayout(button_layout)
        
        # Initialize visibility
        self.on_provider_changed()
        
        # Set a reasonable dialog size
        self.resize(600, 650)
        
    def get_current_api_key(self):
        provider = ['openai', 'gemini', 'anthropic', 'local'][self.provider_combo.currentIndex()]
        return self.config.get_api_key(provider)
        
    def on_provider_changed(self):
        index = self.provider_combo.currentIndex()
        is_local = index == 3  # 'Local Model' is at index 3
        
        self.api_key_group.setVisible(not is_local)
        self.local_model_group.setVisible(is_local)
        
        if not is_local:
            self.key_input.setText(self.get_current_api_key())
        
    def save_settings(self):
        index = self.provider_combo.currentIndex()
        provider = ['openai', 'gemini', 'anthropic', 'local'][index]
        
        # For cloud providers, validate and save API key
        if provider != 'local':
            api_key = self.key_input.text().strip()
            
            if not api_key:
                QMessageBox.warning(self, 'Error', 'Please enter an API key')
                return
                
            self.config.set_api_key(provider, api_key)
        else:
            # For local models, validate and save model settings
            model_path = self.model_path_input.text().strip()
            
            if not model_path or not os.path.exists(model_path):
                QMessageBox.warning(self, 'Error', 
                                  'Please select a valid model file')
                return
            
            # Save model path and type
            self.config.set_local_model_path(model_path)
            
            model_type = "ctransformers" if self.ctransformers_radio.isChecked() else "llama"
            self.config.set_local_model_type(model_type)
        
        # Save selected provider
        self.config.set_selected_provider(provider)
        
        # Save system message
        system_message = self.system_message_input.toPlainText().strip()
        self.config.set_system_message(system_message)
        
        self.accept()
        
    def toggle_key_visibility(self):
        if self.toggle_visibility_btn.isChecked():
            self.key_input.setEchoMode(QLineEdit.Normal)
            self.toggle_visibility_btn.setText('👁️‍🗨️')  # Different eye icon when visible
        else:
            self.key_input.setEchoMode(QLineEdit.Password)
            self.toggle_visibility_btn.setText('👁')
    
    def browse_model(self):
        """Open file dialog to select a model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select AI Model File",
            "",  # Start in the default directory
            "Model Files (*.bin *.gguf *.ggml *.pth);;All Files (*)"
        )
        
        if file_path:
            self.model_path_input.setText(file_path)
            
            # Try to auto-detect model type from filename
            filename = os.path.basename(file_path).lower()
            if any(name in filename for name in ["gemma", "gpt", "phi", "mistral"]):
                self.ctransformers_radio.setChecked(True)
            elif "llama" in filename:
                self.llama_radio.setChecked(True)
                
    def reset_system_message(self):
        """Reset system message to default."""
        self.config.reset_system_message()
        self.system_message_input.setPlainText(self.config.get_system_message())