from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QListWidget, QMessageBox, QProgressDialog,
                            QListWidgetItem, QProgressBar, QWidget, QApplication,
                            QTextEdit)
from PySide6.QtCore import Qt, QSize, Signal, QEvent
from PySide6.QtGui import QColor
from sqlalchemy.orm import Session
from collections.abc import Sequence
import mimetypes
import os
import random
from typing import Protocol

from .domain.contracts import TagDecision
from .models import Tag, File
from .tagging.decision_service import (
    MAX_SYSTEM_ONE_QUESTIONS,
    TagDecisionResult,
    decide_tags,
)
from .vector_search.content_extractor import ContentExtractor


MAX_DOCUMENT_STATE_SAMPLE_BYTES = 16 * 1024
_TRUNCATION_MARKER = "\n[truncated]"
_STRUCTURED_TEXT_SUBTYPES = frozenset(
    {"calendar", "csv", "html", "rtf", "sgml", "xml"}
)


class TagSuggestionConfig(Protocol):
    def get_pdf_extractor(self) -> str: ...


def _document_state(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    stat = os.stat(file_path)
    content = ""
    if _is_plain_text_mime(mime_type):
        try:
            with open(file_path, "rb") as source:
                sample = source.read(MAX_DOCUMENT_STATE_SAMPLE_BYTES)
            content = sample.decode("utf-8", errors="replace").strip()
        except OSError:
            pass
    else:
        content = ContentExtractor.extract_file_content(file_path) or ""
    state = (
        f"Filename: {os.path.basename(file_path)}\n"
        f"Type: {mime_type or 'application/octet-stream'}\n"
        f"Size: {stat.st_size} bytes"
    )
    if content:
        state += f"\nContent sample:\n{content}"
    return _truncate_utf8(state, MAX_DOCUMENT_STATE_SAMPLE_BYTES)


def _is_plain_text_mime(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    media_type, separator, subtype = mime_type.partition("/")
    return (
        bool(separator)
        and media_type == "text"
        and subtype not in _STRUCTURED_TEXT_SUBTYPES
    )


def _truncate_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value

    marker = _TRUNCATION_MARKER.encode("utf-8")
    if limit <= len(marker):
        return marker[:limit].decode("utf-8", errors="ignore")
    prefix = encoded[: limit - len(marker)].decode("utf-8", errors="ignore").rstrip()
    return prefix + _TRUNCATION_MARKER


def _decide_tags_in_batches(
    document_state: str,
    candidates: Sequence[str],
) -> TagDecisionResult:
    decisions: list[TagDecision] = []
    for start in range(0, len(candidates), MAX_SYSTEM_ONE_QUESTIONS):
        batch = candidates[start : start + MAX_SYSTEM_ONE_QUESTIONS]
        decisions.extend(decide_tags(document_state, batch).decisions)
    return TagDecisionResult(decisions=tuple(decisions))


class TagSuggester:
    """Helper class to suggest tags for multiple files."""
    
    def __init__(self, config: TagSuggestionConfig):
        """
        Initialize the tag suggester with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
    def suggest_tags_for_file(self, file_path: str) -> dict[str, float]:
        """
        Get AI tag suggestions for a single file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary mapping tag names to confidence scores
        """
        db_session = None
        try:
            from .models import init_db

            db_session = init_db()
            existing_tags = [str(tag.name) for tag in db_session.query(Tag).all()]
            if not existing_tags:
                return {}
            result = _decide_tags_in_batches(
                _document_state(file_path),
                existing_tags,
            )
            return {
                decision.tag: decision.confidence
                for decision in result.accepted
            }
        except Exception as e:
            print(f"Error suggesting tags for {file_path}: {str(e)}")
            return {}
        finally:
            if db_session is not None:
                db_session.close()
        
# Rest of the file remains unchanged
class TagExplanationDialog(QDialog):
    """Dialog to display explanations for tag suggestions."""
    def __init__(self, tag_name, confidence, explanation, parent=None):
        super().__init__(parent)
        self.tag_name = tag_name
        self.confidence = confidence
        self.explanation = explanation
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(f'Tag Explanation: {self.tag_name}')
        self.setModal(True)
        layout = QVBoxLayout(self)
        
        # Tag name and confidence
        header_layout = QHBoxLayout()
        tag_label = QLabel(f"<b>Tag:</b> {self.tag_name}")
        confidence_label = QLabel(f"<b>Confidence:</b> {self.confidence:.2f}")
        header_layout.addWidget(tag_label)
        header_layout.addWidget(confidence_label)
        layout.addLayout(header_layout)
        
        # Explanation text
        explanation_label = QLabel("<b>Explanation:</b>")
        layout.addWidget(explanation_label)
        
        explanation_text = QTextEdit()
        explanation_text.setReadOnly(True)
        explanation_text.setPlainText(self.explanation)
        explanation_text.setMinimumHeight(200)
        layout.addWidget(explanation_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        
        # Set dialog size
        self.setMinimumWidth(400)
        self.resize(500, 300)

class ConfidenceWidget(QWidget):
    # Signal emitted when widget is clicked
    clicked = Signal(str, float)
    
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
        
        # Store the original tag name and confidence for later use
        self.tag_name = tag
        self.confidence = confidence
        self.progress_bar = progress
        
        # Only make the progress bar use hand cursor to indicate it's clickable
        self.progress_bar.setCursor(Qt.PointingHandCursor)
        
        # Add tooltip only to the progress bar
        self.progress_bar.setToolTip(f"Click for explanation of {tag} tag")
        
        # Install event filter on progress bar to catch clicks
        self.progress_bar.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        """Filter events to capture clicks on the progress bar."""
        if obj == self.progress_bar and event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
            self.clicked.emit(self.tag_name, self.confidence)
            return True
        return super().eventFilter(obj, event)
        
    def mousePressEvent(self, event):
        """Handle mouse click events but don't trigger explanation."""
        # No longer emit signal for the whole widget
        super().mousePressEvent(event)

class TagSuggestionDialog(QDialog):
    def __init__(
        self,
        config: TagSuggestionConfig,
        db_session: Session,
        file_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.db_session = db_session
        self.file_path = file_path
        self.using_cache = False
        self.explanation_text = ""
        self.tag_explanations = {}  # Dictionary to store tag-specific explanations
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
        
        # Info label about clicking for explanations
        info_label = QLabel("Click on any tag confidence bar for an explanation of why it was suggested")
        info_label.setStyleSheet("color: blue; font-style: italic;")
        layout.addWidget(info_label)
        
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
        confidence_widget.clicked.connect(self.show_tag_explanation)
        
        # Store the widget reference to prevent garbage collection
        item.setData(Qt.UserRole, confidence_widget)
        
        list_widget.addItem(item)
        list_widget.setItemWidget(item, confidence_widget)
        
    def _update_progress(self, status: str, progress: int):
        """Update progress bar and status label."""
        self.status_label.setText(status)
        self.progress_bar.setValue(progress)
        QApplication.processEvents()  # Ensure UI updates
    
    def show_tag_explanation(self, tag_name: str, confidence: float):
        """Show explanation dialog for the selected tag."""
        # Get the general explanation
        explanation = self.explanation_text
        
        # Show the explanation dialog
        dialog = TagExplanationDialog(tag_name, confidence, explanation, self)
        dialog.exec()
        
    def analyze_file(self, force_refresh=False):
        """Analyze the file with the local System One decision model."""
        try:
            existing_tags = [
                str(tag.name) for tag in self.db_session.query(Tag).all()
            ]
            if not existing_tags:
                QMessageBox.information(
                    self,
                    "No Tag Candidates",
                    "Create at least one tag before requesting typed suggestions.",
                )
                return

            self._update_progress(
                "Refreshing local tag decisions" if force_refresh else "Requesting tag decisions",
                25,
            )
            result = _decide_tags_in_batches(
                _document_state(self.file_path),
                existing_tags,
            )
            self.explanation_text = (
                "Suggestions are based on typed local System One decisions. "
                "Review confidence before applying a tag."
            )
            self.existing_list.clear()
            self.new_list.clear()
            for decision in result.accepted + result.review:
                self._add_tag_item(
                    decision.tag,
                    decision.confidence,
                    self.existing_list,
                )
            self.cache_label.setText("Generated new local decisions")
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
            is_new_file = False
            if not file_obj:
                file_obj = File(path=self.file_path)
                self.db_session.add(file_obj)
                is_new_file = True
            
            # Track if file had tags before this operation
            had_tags_before = len(file_obj.tags) > 0
            
            # Add existing tags
            for tag_name in selected_existing:
                tag = self.db_session.query(Tag).filter_by(name=tag_name).first()
                if tag and tag not in file_obj.tags:
                    file_obj.tags.append(tag)
            
            # Create and add new tags with random colors
            for tag_name in selected_new:
                tag = self.db_session.query(Tag).filter_by(name=tag_name).first()
                if not tag:
                    # Generate random color with good saturation and brightness
                    hue = random.randint(0, 359)
                    saturation = random.randint(128, 255)  # Medium to high saturation
                    value = random.randint(180, 255)  # Medium to high brightness
                    random_color = QColor.fromHsv(hue, saturation, value).name()
                    tag = Tag(name=tag_name, color=random_color)
                    self.db_session.add(tag)
                if tag not in file_obj.tags:
                    file_obj.tags.append(tag)
            
            # Commit changes to the database
            self.db_session.commit()
              # Check if this is the first time the file has been tagged
            # If so, add it to the vector database for search
            if is_new_file or not had_tags_before:
                try:
                    print(f"Indexing newly tagged file (via AI suggestions): {self.file_path}")
                    # Get vector search from parent window if possible
                    from .vector_search.content_extractor import ContentExtractor
                    from .vector_search import VectorSearch
                    
                    # Get vector search instance (either from parent or create new)
                    vector_search = None
                    if hasattr(self.parent(), 'vector_search'):
                        vector_search = self.parent().vector_search
                    else:
                        vector_search = VectorSearch(self.db_session)
                    
                    # Create progress dialog for extraction
                    progress = QProgressDialog("Extracting content...", "Cancel", 0, 0, self)
                    progress.setWindowTitle("Extracting Content")
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.setValue(0)
                    progress.setAutoClose(False)
                    progress.setAutoReset(False)
                    progress.setCancelButton(None)  # No cancel button
                    progress.setRange(0, 0)  # Show busy indicator (spinner)
                    progress.show()
                    
                    # Variable to track completion state for the accept() call
                    self.content_extraction_complete = False
                    
                    # Define callbacks for the async extraction
                    def on_progress(message):
                        progress.setLabelText(message)
                        QApplication.processEvents()
                    
                    def on_extraction_complete(result):
                        progress.close()
                        content = result.get('content', '')
                        
                        if content:
                            try:
                                # Add file to vector database
                                vector_search.index_file(self.file_path, content)
                                print(f"Added newly tagged file to vector search index: {self.file_path}")
                            except Exception as e:
                                import traceback
                                print(f"Error indexing file: {str(e)}")
                                traceback.print_exc()
                        else:
                            print(f"Warning: No content could be extracted from file: {self.file_path}")
                        
                        # Mark extraction as complete and accept the dialog
                        self.content_extraction_complete = True
                        self.accept()
                      # Get the PDF extractor preference from config
                    pdf_extractor = self.config.get_pdf_extractor()
                    print(f"Starting async content extraction using {pdf_extractor} mode")
                    
                    # Start async content extraction with the chosen extractor mode
                    ContentExtractor.extract_file_content_async(
                        self.file_path, 
                        on_extraction_complete, 
                        on_progress,
                        pdf_extractor=pdf_extractor
                    )
                    
                    # Return without accepting the dialog - the callback will handle it
                    return
                    
                except Exception as e:
                    import traceback
                    print(f"Error adding file to vector search: {str(e)}")
                    traceback.print_exc()
            else:
                # If the file was already tagged before, just update the tags metadata
                try:
                    # Get vector search from parent window if possible
                    vector_search = None
                    if hasattr(self.parent(), 'vector_search'):
                        vector_search = self.parent().vector_search
                    else:
                        from .vector_search import VectorSearch
                        vector_search = VectorSearch(self.db_session)
                    
                    vector_search.update_metadata(self.file_path)
                    print(f"Updated tags metadata for file in vector search: {self.file_path}")
                except Exception as e:
                    import traceback
                    print(f"Error updating vector search metadata: {str(e)}")
                    traceback.print_exc()
                    traceback.print_exc()
                    
            self.accept()
            
        except Exception as e:
            self.db_session.rollback()
            QMessageBox.warning(self, "Error", f"Failed to apply tags: {str(e)}")