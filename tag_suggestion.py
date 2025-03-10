from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QListWidget, QMessageBox, QProgressDialog,
                            QListWidgetItem, QProgressBar, QWidget, QApplication)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor
from sqlalchemy.orm import Session
from datetime import datetime
from models import Tag, File, TagSuggestionCache
from ai_service import AIService
from config import Config

class ConfidenceWidget(QWidget):
    def __init__(self, tag: str, confidence: float, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        
        # Tag label
        label = QLabel(tag)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        
        # Progress bar for confidence
        progress = QProgressBar()
        progress.setMinimum(0)
        progress.setMaximum(100)
        progress.setValue(int(confidence * 100))
        progress.setFormat(f"{confidence:.2f}")
        progress.setTextVisible(True)
        progress.setMinimumWidth(100)
        
        # Set color based on confidence
        style = f"""
            QProgressBar {{
                text-align: center;
                border: 1px solid gray;
                border-radius: 2px;
                background: white;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {'red' if confidence < 0.5 else 'yellow'},
                    stop:1 {'yellow' if confidence < 0.5 else 'green'});
            }}
        """
        progress.setStyleSheet(style)
        layout.addWidget(progress)
        
        # Store the original tag name for later use
        self.tag_name = tag

class TagSuggestionDialog(QDialog):
    def __init__(self, config: Config, db_session: Session, file_path: str, parent=None):
        super().__init__(parent)
        self.config = config
        self.db_session = db_session
        self.file_path = file_path
        self.using_cache = False
        self.initUI()
        self.analyze_file()
        
    def initUI(self):
        self.setWindowTitle('AI Tag Suggestions')
        self.setModal(True)
        layout = QVBoxLayout(self)
        
        # File info
        self.file_label = QLabel(f"Selected file: {self.file_path}")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        
        # Progress section
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label, stretch=1)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)
        
        # Cache status
        self.cache_label = QLabel("")
        layout.addWidget(self.cache_label)
        
        # Create split views for existing and new tags
        tag_layout = QHBoxLayout()
        
        # Existing tags section
        existing_section = QVBoxLayout()
        existing_label = QLabel("Suggested Existing Tags:")
        existing_section.addWidget(existing_label)
        
        self.existing_list = QListWidget()
        self.existing_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        existing_section.addWidget(self.existing_list)
        
        # New tags section
        new_section = QVBoxLayout()
        new_label = QLabel("Suggested New Tags:")
        new_section.addWidget(new_label)
        
        self.new_list = QListWidget()
        self.new_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        new_section.addWidget(self.new_list)
        
        # Add sections to tag layout
        tag_layout.addLayout(existing_section)
        tag_layout.addLayout(new_section)
        layout.addLayout(tag_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply Selected Tags")
        apply_btn.clicked.connect(self.apply_tags)
        
        retry_btn = QPushButton("Force Refresh")
        retry_btn.clicked.connect(lambda: self.analyze_file(force_refresh=True))
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(retry_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        # Set minimum dialog size
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
    def _add_tag_item(self, tag: str, confidence: float, list_widget: QListWidget):
        """Add a tag item with confidence widget to a list widget."""
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 40))  # Set appropriate height for the widget
        
        confidence_widget = ConfidenceWidget(tag, confidence)
        
        # Store the widget reference to prevent garbage collection
        item.setData(Qt.UserRole, confidence_widget)
        
        list_widget.addItem(item)
        list_widget.setItemWidget(item, confidence_widget)
        
    def _update_progress(self, status: str, progress: int):
        """Update progress bar and status label."""
        self.status_label.setText(status)
        self.progress_bar.setValue(progress)
        QApplication.processEvents()  # Ensure UI updates
        
    def analyze_file(self, force_refresh=False):
        """Analyze the file using the configured AI provider."""
        try:
            # Get all existing tags
            existing_tags = [tag.name for tag in self.db_session.query(Tag).all()]
            
            # Get current provider and API key
            provider = self.config.get_selected_provider()
            api_key = self.config.get_api_key(provider)
            
            if not api_key:
                QMessageBox.warning(self, "Error", 
                                  "No API key configured for the selected provider.\n"
                                  "Please configure it in Settings > API Settings.")
                return
            
            # Clear cache if force refresh
            if (force_refresh):
                self.db_session.query(TagSuggestionCache).filter_by(
                    file_path=self.file_path
                ).delete()
                self.db_session.commit()
            
            # Create AI service with progress callback
            service = AIService(
                provider, 
                api_key, 
                self.db_session,
                progress_callback=self._update_progress
            )
            
            # Analyze file
            existing_matches, new_suggestions = service.analyze_file(
                self.file_path, 
                existing_tags
            )
            
            # Update lists
            self.existing_list.clear()
            self.new_list.clear()
            
            # Add items with confidence scores
            for tag, confidence in existing_matches:
                self._add_tag_item(tag, confidence, self.existing_list)
                
            for tag, confidence in new_suggestions:
                self._add_tag_item(tag, confidence, self.new_list)
            
            # Update cache status
            cache_entry = self.db_session.query(TagSuggestionCache).filter_by(
                file_path=self.file_path
            ).first()
            
            if cache_entry:
                age = datetime.utcnow() - cache_entry.timestamp
                self.cache_label.setText(
                    f"Using cached suggestions from {age.days}d {age.seconds//3600}h ago"
                )
                self.cache_label.setStyleSheet("color: blue;")
            else:
                self.cache_label.setText("Generated new suggestions")
                self.cache_label.setStyleSheet("color: green;")
            
            self._update_progress("Ready", 100)
            
        except Exception as e:
            self._update_progress("Error", 0)
            QMessageBox.warning(self, "Error", f"Failed to analyze file: {str(e)}")
            
    def apply_tags(self):
        """Apply the selected tags to the file."""
        try:
            # Get selected tags from both lists using the stored widgets
            selected_existing = [
                self.existing_list.itemWidget(item).tag_name
                for item in self.existing_list.selectedItems()
            ]
            selected_new = [
                self.new_list.itemWidget(item).tag_name
                for item in self.new_list.selectedItems()
            ]
            
            if not (selected_existing or selected_new):
                QMessageBox.information(self, "Info", "Please select some tags to apply")
                return
            
            # Get or create file record
            file_obj = self.db_session.query(File).filter_by(path=self.file_path).first()
            if not file_obj:
                file_obj = File(path=self.file_path)
                self.db_session.add(file_obj)
            
            # Add existing tags
            for tag_name in selected_existing:
                tag = self.db_session.query(Tag).filter_by(name=tag_name).first()
                if tag and tag not in file_obj.tags:
                    file_obj.tags.append(tag)
            
            # Create and add new tags
            for tag_name in selected_new:
                tag = self.db_session.query(Tag).filter_by(name=tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name, color='#808080')  # Default gray color
                    self.db_session.add(tag)
                if tag not in file_obj.tags:
                    file_obj.tags.append(tag)
            
            self.db_session.commit()
            self.accept()
            
        except Exception as e:
            self.db_session.rollback()
            QMessageBox.warning(self, "Error", f"Failed to apply tags: {str(e)}")