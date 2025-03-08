from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QComboBox, QMessageBox)
from PySide6.QtCore import Qt
from config import Config

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
        self.provider_combo.addItems(['OpenAI', 'Gemini', 'Claude'])
        
        # Set current provider
        current_provider = self.config.get_selected_provider()
        index = ['openai', 'gemini', 'anthropic'].index(current_provider)
        self.provider_combo.setCurrentIndex(index)
        
        self.provider_combo.currentIndexChanged.connect(self.on_provider_changed)
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_combo)
        
        # API key input
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
        layout.addLayout(key_layout)
        layout.addLayout(button_layout)
        
    def get_current_api_key(self):
        provider = ['openai', 'gemini', 'anthropic'][self.provider_combo.currentIndex()]
        return self.config.get_api_key(provider)
        
    def on_provider_changed(self):
        self.key_input.setText(self.get_current_api_key())
        
    def save_settings(self):
        provider = ['openai', 'gemini', 'anthropic'][self.provider_combo.currentIndex()]
        api_key = self.key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, 'Error', 'Please enter an API key')
            return
            
        self.config.set_api_key(provider, api_key)
        self.config.set_selected_provider(provider)
        self.accept()
        
    def toggle_key_visibility(self):
        if self.toggle_visibility_btn.isChecked():
            self.key_input.setEchoMode(QLineEdit.Normal)
            self.toggle_visibility_btn.setText('👁️‍🗨️')  # Different eye icon when visible
        else:
            self.key_input.setEchoMode(QLineEdit.Password)
            self.toggle_visibility_btn.setText('👁')