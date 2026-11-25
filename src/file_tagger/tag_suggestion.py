from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QListWidget, QMessageBox,
                            QListWidgetItem, QProgressBar, QWidget, QApplication,
                            QTextEdit)
from PySide6.QtCore import Qt, QSize, Signal, QEvent
from PySide6.QtGui import QColor
from sqlalchemy.orm import Session
from hashlib import sha256
import mimetypes
import os
import random
from typing import Any, Protocol

from .domain.contracts import ModelProfile
from .infrastructure.app_paths import AppPaths
from .infrastructure.database import create_session_factory, upgrade_database
from .models import File, Tag, TagSuggestionCache as LegacySuggestionCache
from .search.hybrid import HybridSearch
from .tagging.cache import (
    CachedTagRecommendations,
    DatabaseTagRecommendationCache,
    TagRecommendationCache,
)
from .tagging.candidate_service import CandidateService
from .extraction.docling_adapter import extract_document


MAX_DOCUMENT_STATE_SAMPLE_BYTES = 16 * 1024
_MISSING_RUNTIME_VALUE = object()
_TRUNCATION_MARKER = "\n[truncated]"
_STRUCTURED_TEXT_SUBTYPES = frozenset(
    {"calendar", "csv", "html", "rtf", "sgml", "xml"}
)
TAG_SUGGESTION_EXTRACTION_VERSION = "tag-suggestion-state-v1"


class TagSuggestionConfig(Protocol):
    pass


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
        content = extract_document(file_path).full_markdown
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


class TagSuggester:
    """Helper class to suggest tags for multiple files."""

    def __init__(
        self,
        config: TagSuggestionConfig,
        db_session: Session | None = None,
        *,
        candidate_service: CandidateService | None = None,
        cache: TagRecommendationCache | None = None,
        gateway: Any | None = None,
        decision_gateway: Any | None = None,
        modern_runtime: Any | None = None,
    ):
        """
        Initialize the tag suggester with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.db_session = db_session
        self._candidate_service = candidate_service
        self._cache = cache
        self._gateway = gateway
        self._decision_gateway = decision_gateway
        self.modern_runtime = modern_runtime

    def recommend_tags_for_file(
        self,
        file_path: str,
        *,
        force_refresh: bool = False,
    ) -> CachedTagRecommendations:
        """Return typed recommendations without applying any tags."""
        document_state = _document_state(file_path)
        canonical_path = os.path.abspath(os.path.normpath(file_path))
        file_hash = _file_hash(file_path)
        if self._candidate_service is not None:
            return self._candidate_service.recommend(
                document_state,
                file_hash=file_hash,
                extraction_version=TAG_SUGGESTION_EXTRACTION_VERSION,
                file_path=canonical_path,
                force_refresh=force_refresh,
            )

        modern_factory = getattr(
            getattr(self, "modern_runtime", None),
            "session_factory",
            None,
        )
        if modern_factory is None:
            paths = AppPaths.discover()
            paths.ensure_directories()
            upgrade_database(paths.database)
            modern_factory = create_session_factory(paths.database)
        search_session = modern_factory()
        cache_session = None
        selected_cache = self._cache
        try:
            if selected_cache is None:
                cache_session = search_session
                selected_cache = DatabaseTagRecommendationCache(cache_session)
            embedding_profile = _runtime_embedding_profile(self.modern_runtime)
            candidate_model = _runtime_selected_model(
                self.modern_runtime,
                "generation",
            )
            decision_model = _runtime_selected_model(
                self.modern_runtime,
                "decision",
            )
            if self.modern_runtime is None:
                service = CandidateService(
                    HybridSearch(
                        search_session,
                        embedding_profile=embedding_profile,
                    ),
                    gateway=self._gateway,
                    decision_gateway=self._decision_gateway,
                    cache=selected_cache,
                )
            elif candidate_model is None or decision_model is None:
                raise ValueError(
                    "Modern runtime model selection is incomplete."
                )
            else:
                service = CandidateService(
                    HybridSearch(
                        search_session,
                        embedding_profile=embedding_profile,
                    ),
                    gateway=self._gateway,
                    decision_gateway=self._decision_gateway,
                    cache=selected_cache,
                    candidate_model=candidate_model,
                    decision_model=decision_model,
                )
            result = service.recommend(
                document_state,
                file_hash=file_hash,
                extraction_version=TAG_SUGGESTION_EXTRACTION_VERSION,
                file_path=canonical_path,
                force_refresh=force_refresh,
                embedding_profile=embedding_profile,
            )
            if cache_session is not None:
                cache_session.commit()
            return result
        except Exception:
            if cache_session is not None:
                cache_session.rollback()
            raise
        finally:
            if cache_session is not None:
                cache_session.close()
            search_session.close()

    def suggest_tags_for_file(
        self,
        file_path: str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, float]:
        """
        Get AI tag suggestions for a single file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary mapping tag names to confidence scores
        """
        try:
            result = self.recommend_tags_for_file(
                file_path,
                force_refresh=force_refresh,
            )
            return {
                decision.tag: decision.confidence
                for decision in (*result.accepted, *result.review)
            }
        except Exception as e:
            print(f"Error suggesting tags for {file_path}: {str(e)}")
            return {}


def _runtime_embedding_profile(runtime: Any | None) -> ModelProfile | None:
    if runtime is None:
        return None
    profile = getattr(
        runtime,
        "embedding_profile",
        _MISSING_RUNTIME_VALUE,
    )
    if profile is _MISSING_RUNTIME_VALUE:
        report = getattr(runtime, "readiness", None)
        if report is not None:
            profile = report.selected_profile("embedding")
        else:
            profile = None
    if profile is None:
        return None
    if not isinstance(profile, ModelProfile):
        raise ValueError("The active embedding profile is invalid.")
    if profile.dimensions not in {256, 512, 768}:
        raise ValueError(
            "The active embedding profile does not support modern search."
        )
    return profile


def _runtime_selected_model(runtime: Any | None, role: str) -> str | None:
    if runtime is None:
        return None
    property_name = {
        "generation": "candidate_model",
        "decision": "decision_model",
    }.get(role)
    if property_name is not None:
        model = getattr(
            runtime,
            property_name,
            _MISSING_RUNTIME_VALUE,
        )
        if model is not _MISSING_RUNTIME_VALUE:
            if model is None:
                return None
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"The active {role} model is invalid.")
            return model
    report = getattr(runtime, "readiness", None)
    if report is None:
        return None
    profile = report.selected_profile(role)
    return profile.model if profile is not None else None


class _LegacyTagRecommendationCache:
    """Persistent recommendation cache for the legacy desktop database."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def lookup(
        self,
        file_hash: str,
        *,
        file_id: int | None = None,
        file_path: str | None = None,
        document_state_digest: str | None = None,
        extraction_version: str | None = None,
        candidate_model_digest: str | None = None,
        decision_model_digest: str | None = None,
        candidate_set_digest: str | None = None,
        max_candidates: int | None = None,
        confirmation_threshold: float | None = None,
        model_digest: str | None = None,
    ) -> CachedTagRecommendations | None:
        del file_id
        query = self._session.query(LegacySuggestionCache).filter_by(
            file_hash=file_hash
        )
        if file_path is not None:
            query = query.filter_by(file_path=file_path)
        row = query.first()
        if row is None or not isinstance(row.suggestions, dict):
            return None
        try:
            stored = CachedTagRecommendations.model_validate(row.suggestions)
        except (TypeError, ValueError):
            return None
        selected_candidate = candidate_model_digest or model_digest
        selected_decision = decision_model_digest or model_digest
        expected = (
            (file_path, stored.file_path),
            (document_state_digest, stored.document_state_digest),
            (extraction_version, stored.extraction_version),
            (selected_candidate, stored.effective_candidate_model_digest),
            (selected_decision, stored.effective_decision_model_digest),
            (candidate_set_digest, stored.candidate_set_digest),
            (max_candidates, stored.max_candidates),
            (confirmation_threshold, stored.confirmation_threshold),
        )
        if any(
            requested is not None and requested != actual
            for requested, actual in expected
        ):
            return None
        return stored

    def store(self, stored: CachedTagRecommendations) -> None:
        if not stored.file_path:
            raise ValueError("Persistent legacy cache entries require a file path.")
        row = (
            self._session.query(LegacySuggestionCache)
            .filter_by(file_path=stored.file_path)
            .first()
        )
        if row is None:
            row = LegacySuggestionCache(file_path=stored.file_path)
            self._session.add(row)
        setattr(row, "file_hash", stored.file_hash)
        setattr(row, "timestamp", stored.cached_at.replace(tzinfo=None))
        setattr(
            row,
            "suggestions",
            stored.model_dump(mode="json", exclude_none=True),
        )
        setattr(row, "provider", "ollama-tag-recommendations")
        self._session.flush()


def _file_hash(file_path: str) -> str:
    digest = sha256()
    with open(file_path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
        
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
        self.progress_bar.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
        
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
        *,
        modern_runtime: Any | None = None,
    ):
        super().__init__(parent)
        self.config = config
        self.db_session = db_session
        self.file_path = file_path
        self.using_cache = False
        self.explanation_text = ""
        self.tag_explanations: dict[str, str] = {}
        self.modern_runtime = modern_runtime
        self.tag_suggester = TagSuggester(
            config,
            db_session,
            modern_runtime=modern_runtime,
        )
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
        item.setData(Qt.UserRole, confidence_widget)  # type: ignore[attr-defined]
        
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
        explanation = self.tag_explanations.get(tag_name, self.explanation_text)
        
        # Show the explanation dialog
        dialog = TagExplanationDialog(tag_name, confidence, explanation, self)
        dialog.exec()
        
    def analyze_file(self, force_refresh: bool = False) -> None:
        """Analyze the file with the local candidate and decision services."""
        try:
            self._update_progress(
                "Refreshing local tag decisions" if force_refresh else "Requesting tag decisions",
                25,
            )
            result = self.tag_suggester.recommend_tags_for_file(
                self.file_path,
                force_refresh=force_refresh,
            )
            self.explanation_text = (
                "Suggestions are based on typed local System One decisions. "
                "Review confidence before applying a tag."
            )
            self.existing_list.clear()
            self.new_list.clear()
            for decision in result.accepted + result.review:
                target = self.new_list if decision.is_new else self.existing_list
                self._add_tag_item(
                    decision.tag,
                    decision.confidence,
                    target,
                )
                if decision.rationale:
                    self.tag_explanations[decision.tag] = decision.rationale
            self.cache_label.setText(
                "Refreshed local recommendations"
                if force_refresh
                else "Recommendations checked against persistent cache"
            )
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
            runtime = getattr(self, "modern_runtime", None)
            if runtime is not None:
                try:
                    runtime.sync_file(
                        self.file_path,
                        [(tag.name, tag.color) for tag in file_obj.tags],
                    )
                except Exception:
                    pass
            parent = self.parent()
            if parent is not None and hasattr(parent, "_index_modern_file"):
                parent._index_modern_file(self.file_path)

            self.accept()
            
        except Exception as e:
            self.db_session.rollback()
            QMessageBox.warning(self, "Error", f"Failed to apply tags: {str(e)}")