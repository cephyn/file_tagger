import os
import logging
import traceback
from collections.abc import Iterable
from typing import Any, cast
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeView,
    QListWidget,
    QPushButton,
    QInputDialog,
    QColorDialog,
    QLabel,
    QFileSystemModel,
    QMessageBox,
    QLineEdit,
    QRadioButton,
    QComboBox,
    QHeaderView,
    QMenu,
    QFileDialog,
    QTabWidget,
    QProgressDialog,
    QFrame,
    QListWidgetItem,
    QDialog,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QApplication,
    QStyle,
)
from PySide6.QtCore import Qt, QStorageInfo
from PySide6.QtGui import (
    QColor,
    QTextDocument,
    QAction,
    QActionGroup,
)
from PySide6.QtCore import QSize
from .models import File, Tag
from .ollama.readiness import ReadinessReport
from .password_management import PasswordManagementDialog
from .tag_suggestion import TagSuggestionDialog
from .ui.indexing_status import IndexingStatus
from .ui.ollama_settings import OllamaSettingsDialog
from .migration.legacy_chroma import cleanup_legacy_chroma, inspect_legacy_chroma
from .utils import is_dark_color, get_score_color, open_file, open_containing_folder
from .infrastructure.app_paths import AppPaths
from .modern_context import ModernIndexingUnavailable


class HTMLDelegate(QStyledItemDelegate):
    """Custom delegate to render HTML in QListWidgetItems with text wrapping"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc = QTextDocument()

    def paint(self, painter, option, index):
        """Paint the item using HTML rendering"""
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        if options.widget is None:
            style = QApplication.style()
        else:
            style = options.widget.style()
            # Get text and check if it has HTML content
        text = index.data(Qt.DisplayRole)
        if text and ("<" in text and ">" in text):  # Check for any HTML tags
            # Save painter state
            painter.save()

            # Prepare document
            self.doc.setHtml(text)

            # Calculate available width - use full width minus margins
            available_width = options.rect.width() - 10  # Subtract margins
            self.doc.setTextWidth(available_width)

            # Clear text to avoid default rendering
            options.text = ""
            style.drawControl(QStyle.CE_ItemViewItem, options, painter)

            # Set up the text position with a small margin
            textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)
            painter.translate(textRect.topLeft())

            # Draw text with HTML formatting
            self.doc.drawContents(painter)

            # Restore painter
            painter.restore()
        else:
            # Default rendering for non-HTML text
            style.drawControl(QStyle.CE_ItemViewItem, options, painter)

    def sizeHint(self, option, index):
        """Calculate the size needed for the HTML content with word wrapping"""
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        text = index.data(Qt.DisplayRole)
        if text and ("<" in text and ">" in text):  # Check for any HTML tags
            self.doc.setHtml(text)

            # Set text width to the available width in the view
            available_width = option.rect.width()
            if available_width <= 0:
                # If we don't have a valid width yet, use a reasonable default
                # This happens during initial layout
                available_width = 500

            self.doc.setTextWidth(available_width)

            # Return a size that accommodates the wrapped text
            return QSize(available_width, self.doc.size().height())

        return super().sizeHint(option, index)


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
        version = QLabel("Version 0.0.1")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)

        # Description
        desc = QLabel(
            "A feature-rich file management and tagging system with AI-powered search capabilities."
        )
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
        website_label = QLabel("<a href='https://www.busywyvern.com'>Busy Wyvern</a>")
        website_label.setOpenExternalLinks(True)
        website_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(website_label)

        # Technologies used
        tech_label = QLabel("Built with Python, PySide6, SQLAlchemy, and Vector Search")
        tech_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(tech_label)

        # Copyright
        copyright_label = QLabel("© 2025 All Rights Reserved")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class FileTagManager(QMainWindow):
    """Main window for the File Tagger application."""

    def __init__(self, db_session, config, vector_search=None, modern_runtime=None):
        # Set up logging for the main window
        self.logger = logging.getLogger("FileTagManager")
        # Add a file handler for main window-specific issues
        log_dir = AppPaths.discover().logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "main_window_debug.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(logging.StreamHandler())

        self.logger.debug("FileTagManager initialization started")

        try:
            super().__init__()
            self.logger.debug("QMainWindow initialized")

            try:
                self.logger.debug(f"Received db_session: {db_session is not None}")
                self.db_session = db_session

                self.logger.debug(f"Received config: {config is not None}")
                self.config = config

                self.logger.debug(
                    f"Received vector_search: {vector_search is not None}"
                )
                self.vector_search = vector_search
                self.modern_runtime = modern_runtime
                self.modern_index_status_text = ""

                self.current_file_path = None
                self.current_search_results = []  # Initialize search results storage

                self.logger.debug("About to call init_ui")
                self.init_ui()
                self.logger.debug("init_ui completed successfully")
            except Exception as e:
                self.logger.error(
                    f"Error during FileTagManager initialization: {str(e)}"
                )
                self.logger.error(traceback.format_exc())
                raise

        except Exception as e:
            self.logger.critical(f"Fatal error in FileTagManager constructor: {str(e)}")
            self.logger.critical(traceback.format_exc())
            # Show error dialog since this is a critical failure
            QMessageBox.critical(
                None,
                "Critical Error",
                f"Failed to initialize the main window: {str(e)}\n\n"
                "Please check the logs for more details.",
            )
            raise

    def _set_modern_index_status(self, message: str) -> None:
        self.modern_index_status_text = message
        status_label = getattr(self, "modern_index_status_label", None)
        if status_label is not None:
            status_label.setText(message)

    def _index_modern_file(self, file_path: str) -> bool:
        indexing_status = getattr(self, "indexing_status", None)
        if indexing_status is None:
            self._set_modern_index_status(
                "Modern indexing unavailable: indexing status UI is not initialized."
            )
            return False
        try:
            if callable(getattr(indexing_status, "setVisible", None)):
                indexing_status.setVisible(True)
            indexing_status.start(file_path)
            self._set_modern_index_status("Modern indexing started.")
            return True
        except ModernIndexingUnavailable as exc:
            self._set_modern_index_status(
                f"Modern indexing unavailable: {exc}"
            )
            return False
        except Exception as exc:
            self._set_modern_index_status(
                f"Modern indexing failed: {exc.__class__.__name__}."
            )
            self.logger.warning(
                "Modern indexing failed for a requested file (%s).",
                exc.__class__.__name__,
            )
            return False

    def _index_modern_files(self, file_paths: Iterable[str]) -> bool:
        paths = list(file_paths)
        indexing_status = getattr(self, "indexing_status", None)
        if indexing_status is None:
            self._set_modern_index_status(
                "Modern indexing unavailable: indexing status UI is not initialized."
            )
            return False
        try:
            if callable(getattr(indexing_status, "setVisible", None)):
                indexing_status.setVisible(True)
            indexing_status.start_many(paths)
            self._set_modern_index_status(
                f"Modern indexing queued for {len(paths)} file(s)."
            )
            return True
        except ModernIndexingUnavailable as exc:
            self._set_modern_index_status(
                f"Modern indexing unavailable: {exc}"
            )
            return False
        except Exception as exc:
            self._set_modern_index_status(
                f"Modern indexing failed: {exc.__class__.__name__}."
            )
            self.logger.warning(
                "Modern indexing failed for requested files (%s).",
                exc.__class__.__name__,
            )
            return False

    def _commit_legacy_changes(
        self,
        *,
        tag_upserts: Iterable[tuple[str, str | None]] = (),
        tag_deletes: Iterable[str] = (),
        file_paths: Iterable[tuple[str, Iterable[tuple[str, str | None]]]] = (),
        file_deletes: Iterable[str] = (),
    ) -> bool:
        """Commit legacy UI state first, then best-effort mirror it to modern."""
        self.db_session.commit()
        runtime = getattr(self, "modern_runtime", None)
        if runtime is None:
            return False
        try:
            synchronize = getattr(runtime, "sync_legacy_changes", None)
            if synchronize is not None:
                synchronize(
                    tag_upserts=list(tag_upserts),
                    tag_deletes=list(tag_deletes),
                    file_paths=[(path, list(tags)) for path, tags in file_paths],
                    file_deletes=list(file_deletes),
                )
            else:
                for name in tag_deletes:
                    runtime.delete_tag(name)
                for name, color in tag_upserts:
                    runtime.sync_tag(name, color)
                for path, tags in file_paths:
                    runtime.sync_file(path, list(tags))
                for path in file_deletes:
                    runtime.delete_file(path)
        except Exception as exc:
            self.logger.warning(
                "Modern database synchronization failed (%s); legacy state is safe.",
                exc.__class__.__name__,
            )
            return False
        return True

    def showEvent(self, event):
        """Override showEvent to log when window is shown"""
        self.logger.debug("Main window show event received")
        super().showEvent(event)
        self.logger.debug("Main window is now visible")

    def closeEvent(self, event):
        """Override closeEvent to log when window is closed"""
        self.logger.debug(
            f"Main window close event received, reason: {event.spontaneous()}"
        )
        self.logger.debug("Stack trace at close event:")
        self.logger.debug("".join(traceback.format_stack()))

        # Call parent class method to proceed with normal closing
        super().closeEvent(event)
        self.logger.debug("Main window closed")

    def init_ui(self):
        self.logger.debug("init_ui started")
        try:
            self.setWindowTitle("File Tagger")
            self.setGeometry(100, 100, 1200, 700)
            self.logger.debug("Window title and geometry set")

            # Create menu bar and menus
            self.logger.debug("Setting up menus")
            self.setup_menus()
            self.logger.debug("Menus set up successfully")

            # Create central widget and main layout
            self.logger.debug("Creating central widget")
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            self.logger.debug("Central widget created")

            # Create the main tab widget
            self.logger.debug("Creating main tabs")
            self.main_tabs = QTabWidget()
            main_layout.addWidget(self.main_tabs)
            self.logger.debug("Main tabs created")

            # Create and add the tagging and search tabs
            self.logger.debug("Creating tagging tab")
            tagging_tab = self.create_tagging_tab()
            self.logger.debug("Tagging tab created")

            self.logger.debug("Creating search tab")
            search_tab = self.create_search_tab()
            self.logger.debug("Search tab created")

            self.main_tabs.addTab(tagging_tab, "Tagging Interface")
            self.main_tabs.addTab(search_tab, "Search Interface")
            self.logger.debug("Added tabs to main tabs widget")

            runtime = getattr(self, "modern_runtime", None)
            self.modern_index_status_label = QLabel(
                "Modern index: initializing."
                if runtime is not None
                else "Modern index: unavailable."
            )
            main_layout.addWidget(self.modern_index_status_label)
            self.indexing_status = IndexingStatus(runtime, self)
            self.indexing_status.status_changed.connect(
                lambda message, _value: self._set_modern_index_status(message)
            )
            self.indexing_status.setVisible(runtime is not None)
            main_layout.addWidget(self.indexing_status)

            # Set initial directory from config
            self.logger.debug("Setting initial directory")
            try:
                initial_path = self.config.get_home_directory()
                self.logger.debug(f"Got home directory from config: {initial_path}")
                self.tree.setRootIndex(self.model.index(initial_path))
                self.logger.debug("Set root index of tree to home directory")

                # Update initial path display
                self.path_display.setText(initial_path)
                self.logger.debug("Updated path display")
            except Exception as e:
                self.logger.error(f"Error setting initial directory: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise

            # Update drive list and select current drive
            self.logger.debug("Updating drive list")
            try:
                self.update_drive_list()
                self.logger.debug("Drive list updated")
                drive = os.path.splitdrive(initial_path)[0] + os.path.sep
                for i in range(self.drive_combo.count()):
                    if self.drive_combo.itemData(i).startswith(drive):
                        self.drive_combo.setCurrentIndex(i)
                        break
                self.logger.debug(f"Selected drive in combo box: {drive}")
            except Exception as e:
                self.logger.error(f"Error updating drive list: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise

            # Connect signals after initialization
            self.logger.debug("Connecting signals")
            try:
                self.drive_combo.currentIndexChanged.connect(self.on_drive_changed)
                self.logger.debug("Connected drive_combo signal")
                self.refresh_tags()
                self.logger.debug("Tags refreshed")
            except Exception as e:
                self.logger.error(
                    f"Error connecting signals or refreshing tags: {str(e)}"
                )
                self.logger.error(traceback.format_exc())
                raise

            self.logger.debug("init_ui completed successfully")
        except Exception as e:
            self.logger.error(f"Error in init_ui: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def setup_menus(self):
        """Set up the application menus."""
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        indexing_menu = menubar.addMenu("Indexing")
        help_menu = menubar.addMenu("Help")

        # Add settings menu actions
        ollama_settings_action = settings_menu.addAction("Ollama Readiness")
        ollama_settings_action.triggered.connect(self.show_ollama_settings)

        home_dir_action = settings_menu.addAction("Set Home Directory")
        home_dir_action.triggered.connect(self.set_home_directory)

        password_action = settings_menu.addAction("Password Management")
        password_action.triggered.connect(self.show_password_management)

        # Add PDF extractor settings submenu
        pdf_extractor_menu = settings_menu.addMenu("PDF Extractor")

        # Create action group to ensure only one option is selected at a time
        self.pdf_extractor_group = QActionGroup(self)
        self.pdf_extractor_group.setExclusive(True)

        # Create actions for PDF extractor options
        self.fast_action = QAction("Fast (markitdown)", self)
        self.fast_action.setCheckable(True)
        self.fast_action.setActionGroup(self.pdf_extractor_group)
        self.fast_action.triggered.connect(lambda: self.set_pdf_extractor("fast"))

        self.accurate_action = QAction("Slower, more accurate (docling)", self)
        self.accurate_action.setCheckable(True)
        self.accurate_action.setActionGroup(self.pdf_extractor_group)
        self.accurate_action.triggered.connect(
            lambda: self.set_pdf_extractor("accurate")
        )

        # Add actions to the menu
        pdf_extractor_menu.addAction(self.fast_action)
        pdf_extractor_menu.addAction(self.accurate_action)

        # Set initial state based on configuration
        pdf_extractor = self.config.get_pdf_extractor()
        self.fast_action.setChecked(pdf_extractor == "fast")
        self.accurate_action.setChecked(pdf_extractor == "accurate")

        # Add scan directory action
        scan_action = menubar.addAction("Scan Directory")
        scan_action.triggered.connect(self.scan_directory_for_untagged)

        reindex_action = indexing_menu.addAction("Reindex All Files")
        reindex_action.triggered.connect(self.reindex_files)
        resume_action = indexing_menu.addAction("Resume Indexing")
        resume_action.triggered.connect(self.resume_indexing)
        migration_action = indexing_menu.addAction("Migration Inventory")
        migration_action.triggered.connect(self.show_migration_inventory)
        cleanup_action = indexing_menu.addAction("Clean Legacy Search Data")
        cleanup_action.triggered.connect(self.cleanup_legacy_search_data)

        # Add help menu actions
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about_dialog)

    def show_about_dialog(self):
        """Show the About dialog."""
        dialog = AboutDialog(self)
        dialog.exec()

    def create_tagging_tab(self):
        """Create and return the tagging interface tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # File explorer section
        explorer_layout = self.create_file_explorer_section()

        # Tag management section
        tag_layout = self.create_tag_management_section()

        # File tags section
        file_tags_layout = self.create_file_tags_section()

        # Add sections to tab layout
        layout.addLayout(explorer_layout, stretch=2)
        layout.addLayout(tag_layout, stretch=1)
        layout.addLayout(file_tags_layout, stretch=1)

        return tab

    def create_search_tab(self):
        """Create and return the search interface tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Create tab widget for different search types
        search_tabs = QTabWidget()

        # Create and add tag search and RAG search tabs
        tag_search_tab = self.create_tag_search_tab()
        rag_search_tab = self.create_rag_search_tab()

        search_tabs.addTab(tag_search_tab, "Tag Search")
        search_tabs.addTab(rag_search_tab, "Content Search")

        layout.addWidget(search_tabs)
        return tab

    def create_file_explorer_section(self):
        """Create and return the file explorer section layout."""
        explorer_layout = QVBoxLayout()
        explorer_layout.addWidget(QLabel("File Explorer"))

        # Navigation controls
        nav_layout = QHBoxLayout()

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
        self.model.setRootPath("")

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setSortingEnabled(True)
        self.tree.header().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.tree.header().setSectionsClickable(True)

        # Configure column resizing
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.tree.header().setStretchLastSection(False)
        self.tree.setColumnWidth(0, 250)  # Name
        self.tree.setColumnWidth(1, 100)  # Size
        self.tree.setColumnWidth(2, 100)  # Type
        self.tree.setColumnWidth(3, 150)  # Date Modified
        self.tree.header().setMinimumSectionSize(50)

        # Connect signals
        self.tree.selectionModel().selectionChanged.connect(self.on_file_selected)
        self.tree.doubleClicked.connect(self.on_item_double_clicked)
        explorer_layout.addWidget(self.tree)

        return explorer_layout

    def create_tag_management_section(self):
        """Create and return the tag management section layout."""
        tag_layout = QVBoxLayout()
        tag_layout.addWidget(QLabel("Tags"))

        self.tag_list = QListWidget()
        self.tag_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
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

        return tag_layout

    def create_file_tags_section(self):
        """Create and return the file tags section layout."""
        file_tags_layout = QVBoxLayout()
        file_tags_layout.addWidget(QLabel("File Tags"))

        self.file_tags_list = QListWidget()
        self.file_tags_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        file_tags_layout.addWidget(self.file_tags_list)

        buttons = QHBoxLayout()
        add_file_tag_btn = QPushButton("Add Tag to File")
        add_file_tag_btn.clicked.connect(self.add_tag_to_file)
        remove_file_tag_btn = QPushButton("Remove Tag from File")
        remove_file_tag_btn.clicked.connect(self.remove_tag_from_file)
        suggest_tags_btn = QPushButton("Suggest Tags (AI)")
        suggest_tags_btn.clicked.connect(self.suggest_tags)

        buttons.addWidget(add_file_tag_btn)
        buttons.addWidget(remove_file_tag_btn)
        buttons.addWidget(suggest_tags_btn)
        file_tags_layout.addLayout(buttons)

        return file_tags_layout

    def create_tag_search_tab(self):
        """Create and return the tag search tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Tag selection section
        tag_select_layout = QVBoxLayout()
        tag_select_layout.addWidget(QLabel("Select Tags to Search:"))

        self.search_tag_list = QListWidget()
        self.search_tag_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        tag_select_layout.addWidget(self.search_tag_list)

        # Search controls
        controls = QHBoxLayout()
        self.and_radio = QRadioButton("AND")
        self.or_radio = QRadioButton("OR")
        self.and_radio.setChecked(True)
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_by_tags)

        controls.addWidget(self.and_radio)
        controls.addWidget(self.or_radio)
        controls.addWidget(search_btn)
        tag_select_layout.addLayout(controls)

        # Results section
        results_layout = QVBoxLayout()
        results_layout.addWidget(QLabel("Search Results:"))

        self.search_results = QListWidget()
        self.search_results.itemDoubleClicked.connect(
            self.on_search_result_double_clicked
        )
        self.search_results.setContextMenuPolicy(Qt.CustomContextMenu)
        self.search_results.customContextMenuRequested.connect(
            self.on_search_result_right_clicked
        )
        results_layout.addWidget(self.search_results)

        # Set custom delegate for HTML rendering
        self.search_results.setItemDelegate(HTMLDelegate(self.search_results))

        layout.addLayout(tag_select_layout)
        layout.addLayout(results_layout)

        return tab

    def create_rag_search_tab(self):
        """Create and return the RAG search tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Query input section
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Query:"))

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your search query...")
        query_layout.addWidget(self.query_input)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_by_content)
        query_layout.addWidget(search_btn)

        layout.addLayout(query_layout)

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

        self.tag_filter_list = QListWidget()
        self.tag_filter_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        filter_layout.addWidget(self.tag_filter_list)

        layout.addLayout(filter_layout)

        # Results section
        results_layout = QVBoxLayout()
        results_layout.addWidget(QLabel("Search Results:"))

        self.rag_search_results = QListWidget()
        self.rag_search_results.itemDoubleClicked.connect(
            self.on_search_result_double_clicked
        )
        self.rag_search_results.setContextMenuPolicy(Qt.CustomContextMenu)
        self.rag_search_results.customContextMenuRequested.connect(
            self.on_search_result_right_clicked
        )
        results_layout.addWidget(self.rag_search_results)

        # Set custom delegate for HTML rendering
        self.rag_search_results.setItemDelegate(HTMLDelegate(self.rag_search_results))

        # Add buttons layout
        buttons_layout = QHBoxLayout()

        # Reindex button
        reindex_btn = QPushButton("Reindex Files")
        reindex_btn.clicked.connect(self.reindex_files)
        buttons_layout.addWidget(reindex_btn)

        # Chat with results button
        chat_btn = QPushButton("Chat with Results")
        chat_btn.setEnabled(False)
        self.chat_results_btn = chat_btn
        buttons_layout.addWidget(chat_btn)

        results_layout.addLayout(buttons_layout)

        layout.addLayout(results_layout)

        return tab

    def update_drive_list(self):
        """Update the list of available drives in the combo box."""
        self.drive_combo.clear()
        for drive in QStorageInfo.mountedVolumes():
            if not drive.isValid() or not drive.isReady():
                continue
            name = drive.displayName()
            root_path = drive.rootPath()
            self.drive_combo.addItem(f"{name} ({root_path})", root_path)

    def on_drive_changed(self, index):
        """Handle drive selection changes."""
        if index >= 0:
            drive_path = self.drive_combo.itemData(index)
            model_index = self.model.index(drive_path)
            self.tree.setRootIndex(model_index)
            current_column = self.tree.header().sortIndicatorSection()
            current_order = self.tree.header().sortIndicatorOrder()
            self.tree.sortByColumn(current_column, current_order)
            self.path_display.setText(drive_path)

    def go_home(self):
        """Navigate to the configured home directory."""
        home_path = self.config.get_home_directory()
        self.tree.setRootIndex(self.model.index(home_path))
        self.path_display.setText(home_path)
        home_drive = os.path.splitdrive(home_path)[0] + os.path.sep
        for i in range(self.drive_combo.count()):
            if self.drive_combo.itemData(i).startswith(home_drive):
                self.drive_combo.setCurrentIndex(i)
                break

    def go_up(self):
        """Navigate to the parent directory."""
        current_path = self.path_display.text()
        parent_path = os.path.dirname(current_path)
        if os.path.exists(parent_path):
            if os.path.splitdrive(current_path)[1] == os.path.sep:
                return  # Don't go up from root of drive
            self.tree.setRootIndex(self.model.index(parent_path))
            self.path_display.setText(parent_path)

    def refresh_tags(self):
        """Update all tag lists in the UI."""
        self.tag_list.clear()
        self.tag_filter_list.clear()
        self.search_tag_list.clear()

        tags = self.db_session.query(Tag).all()
        for tag in tags:
            tag_color = QColor(tag.color)
            text_color = Qt.white if is_dark_color(tag_color) else Qt.black

            # Add to main tag list
            self.tag_list.addItem(tag.name)
            item = self.tag_list.item(self.tag_list.count() - 1)
            item.setBackground(tag_color)
            item.setForeground(text_color)

            # Add to filter list
            self.tag_filter_list.addItem(tag.name)
            item = self.tag_filter_list.item(self.tag_filter_list.count() - 1)
            item.setBackground(tag_color)
            item.setForeground(text_color)

            # Add to search tag list
            self.search_tag_list.addItem(tag.name)
            item = self.search_tag_list.item(self.search_tag_list.count() - 1)
            item.setBackground(tag_color)
            item.setForeground(text_color)

    def refresh_file_tags(self):
        """Update the file tags list for the currently selected file."""
        self.file_tags_list.clear()
        if not self.current_file_path:
            return

        file_obj = (
            self.db_session.query(File).filter_by(path=self.current_file_path).first()
        )
        if file_obj:
            for tag in file_obj.tags:
                tag_color = QColor(tag.color)
                text_color = Qt.white if is_dark_color(tag_color) else Qt.black

                self.file_tags_list.addItem(tag.name)
                item = self.file_tags_list.item(self.file_tags_list.count() - 1)
                item.setBackground(tag_color)
                item.setForeground(text_color)

    def on_file_selected(self, current, previous):
        """Handle file selection changes in the tree view."""
        if current.indexes():
            self.current_file_path = self.model.filePath(current.indexes()[0])
            self.refresh_file_tags()

    def on_item_double_clicked(self, index):
        """Handle double-click on tree view items."""
        path = self.model.filePath(index)
        if os.path.isdir(path):
            self.tree.setRootIndex(index)
            self.path_display.setText(path)
        else:
            try:
                open_file(path)
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def on_search_result_double_clicked(self, item):
        """Handle double-click on search result items."""
        file_path = item.toolTip()
        if not file_path or not os.path.exists(file_path):
            return

        try:
            if os.path.isfile(file_path):
                open_file(file_path)
            else:
                # For directories, navigate to them in the tree view
                drive = os.path.splitdrive(file_path)[0] + os.path.sep
                for i in range(self.drive_combo.count()):
                    if self.drive_combo.itemData(i).startswith(drive):
                        self.drive_combo.setCurrentIndex(i)
                        break

                dir_path = (
                    os.path.dirname(file_path)
                    if os.path.isfile(file_path)
                    else file_path
                )
                self.tree.setRootIndex(self.model.index(dir_path))
                self.path_display.setText(dir_path)
                self.tree.setCurrentIndex(self.model.index(file_path))
                self.tree.scrollTo(self.model.index(file_path))
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def on_search_result_right_clicked(self, position):
        """Handle right-click on search result items to show context menu."""
        item = self.sender().itemAt(position)
        if not item:
            return

        context_menu = QMenu(self)
        file_path = item.toolTip()
        if not file_path or not os.path.exists(file_path):
            return

        open_action = context_menu.addAction("Open File")
        open_in_folder_action = context_menu.addAction("Open Containing Folder")
        reindex_action = context_menu.addAction("Force Reindex")
        remove_action = context_menu.addAction("Remove from Search Index")

        action = context_menu.exec_(self.sender().mapToGlobal(position))

        try:
            if action == open_action:
                open_file(file_path)
            elif action == open_in_folder_action:
                open_containing_folder(file_path)
            elif action == reindex_action:
                self.force_reindex_file(file_path)
            elif action == remove_action:
                self._remove_from_vector_db(file_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _remove_from_vector_db(self, file_path):
        """Remove a file from the modern index and tag database."""
        reply = QMessageBox.question(
            self,
            "Remove File",
            f"Are you sure you want to remove\n{file_path}\nfrom the system?\n\n"
            "This will remove the file from both the search index and the tag database.\n"
            "Note: This will not delete the actual file from your computer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            success_tags = False
            success_modern = False
            try:
                file_obj = self.db_session.query(File).filter_by(path=file_path).first()
                if file_obj:
                    self.db_session.delete(file_obj)
                    success_modern = self._commit_legacy_changes(
                        file_deletes=[file_path]
                    )
                    success_tags = True
            except Exception as e:
                print(f"Error removing file from tag database: {str(e)}")
                import traceback

                traceback.print_exc()

            if success_modern and success_tags:
                QMessageBox.information(
                    self,
                    "Success",
                    "File removed from the modern index and tag database successfully.\n\n"
                    "The file is still on your computer.",
                )
                self._remove_item_from_results(file_path)
            elif success_tags:
                QMessageBox.warning(
                    self,
                    "Partial Success",
                    "File was removed from the tag database but the modern index could not be updated.",
                )
                self._remove_item_from_results(file_path)
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Could not remove the file from the tag database. See console for details.",
                )

    def _remove_item_from_results(self, file_path):
        """Remove items with the given file path from search results lists."""
        current_tab = self.main_tabs.currentIndex()

        if current_tab == 1:  # Search tab
            search_tabs = self.main_tabs.widget(1).layout().itemAt(0).widget()
            current_search_tab = search_tabs.currentIndex()

            if current_search_tab == 0:  # Tag search tab
                for i in range(self.search_results.count()):
                    item = self.search_results.item(i)
                    if item and item.toolTip() == file_path:
                        self.search_results.takeItem(i)
                        break
            elif current_search_tab == 1:  # RAG search tab
                for i in range(self.rag_search_results.count()):
                    item = self.rag_search_results.item(i)
                    if item and item.toolTip() == file_path:
                        self.rag_search_results.takeItem(i)
                        j = i + 1
                        while j < self.rag_search_results.count():
                            next_item = self.rag_search_results.item(j)
                            if next_item and next_item.text().startswith("    ↪"):
                                self.rag_search_results.takeItem(j)
                            else:
                                break
                        break

    def add_tag(self):
        """Add a new tag."""
        tag_name, ok = QInputDialog.getText(self, "Add Tag", "Enter tag name:")
        if ok and tag_name:
            # Check if tag already exists
            existing = self.db_session.query(Tag).filter_by(name=tag_name).first()
            if existing:
                QMessageBox.warning(
                    self, "Error", "A tag with this name already exists!"
                )
                return

            # Get color for the tag
            color = QColorDialog.getColor()
            if color.isValid():
                # Create new tag
                new_tag = Tag(name=tag_name, color=color.name())
                self.db_session.add(new_tag)
                self._commit_legacy_changes(
                    tag_upserts=[(tag_name, color.name())]
                )

                # Refresh tag lists
                self.refresh_tags()

    def edit_tag(self):
        """Edit the selected tag."""
        selected_items = self.tag_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select a tag to edit!")
            return

        # Get the selected tag
        tag_name = selected_items[0].text()
        tag = self.db_session.query(Tag).filter_by(name=tag_name).first()
        if not tag:
            return

        # Get new name
        old_name = tag.name
        new_name, ok = QInputDialog.getText(
            self, "Edit Tag", "Enter new tag name:", text=tag.name
        )
        if not ok or not new_name:
            return

        # Check if new name already exists (if different from current)
        if new_name != tag.name:
            existing = self.db_session.query(Tag).filter_by(name=new_name).first()
            if existing:
                QMessageBox.warning(
                    self, "Error", "A tag with this name already exists!"
                )
                return

        # Get new color
        initial_color = QColor(tag.color)
        color = QColorDialog.getColor(initial=initial_color)
        if color.isValid():
            # Update tag
            tag.name = new_name
            tag.color = color.name()
            self._commit_legacy_changes(
                tag_upserts=[(new_name, color.name())],
                tag_deletes=[old_name] if old_name != new_name else [],
                file_paths=[
                    (
                        file.path,
                        [(file_tag.name, file_tag.color) for file_tag in file.tags],
                    )
                    for file in tag.files
                ],
            )

            # Refresh tag lists
            self.refresh_tags()
            self.refresh_file_tags()

    def delete_tag(self):
        """Delete the selected tag(s)."""
        selected_items = self.tag_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select tag(s) to delete!")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {len(selected_items)} tag(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            deleted_names: list[str] = []
            for item in selected_items:
                tag = self.db_session.query(Tag).filter_by(name=item.text()).first()
                if tag:
                    deleted_names.append(tag.name)
                    self.db_session.delete(tag)

            self._commit_legacy_changes(tag_deletes=deleted_names)
            self.refresh_tags()
            self.refresh_file_tags()

    def add_tag_to_file(self):
        """Add selected tag(s) to the current file."""
        if not self.current_file_path:
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return

        selected_items = self.tag_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select tag(s) to add!")
            return

        # DEBUGGING: Add console output to track what's happening
        print(f"\n=== DEBUGGING: Adding tags to file: {self.current_file_path} ===")

        # Get or create file record
        file_obj = (
            self.db_session.query(File).filter_by(path=self.current_file_path).first()
        )
        is_new_file = False
        if not file_obj:
            print("DEBUGGING: File not found in database, creating new record")
            file_obj = File(path=self.current_file_path)
            self.db_session.add(file_obj)
            is_new_file = True
        else:
            print(
                f"DEBUGGING: File found in database with {len(file_obj.tags)} existing tags"
            )

        # Track if file had tags before this operation
        had_tags_before = len(file_obj.tags) > 0
        print(
            f"DEBUGGING: had_tags_before = {had_tags_before}, is_new_file = {is_new_file}"
        )

        # Add selected tags
        tags_added = []
        for item in selected_items:
            tag = self.db_session.query(Tag).filter_by(name=item.text()).first()
            if tag and tag not in file_obj.tags:
                file_obj.tags.append(tag)
                tags_added.append(tag.name)

        print(f"DEBUGGING: Added tags: {tags_added}")

        # Perform database commit - this will generate ID for new files
        self._commit_legacy_changes(
            file_paths=[
                (
                    self.current_file_path,
                    [(tag.name, tag.color) for tag in file_obj.tags],
                )
            ]
        )
        print("DEBUGGING: Database commit completed")

        self._index_modern_file(self.current_file_path)

        # Refresh file tags display
        self.refresh_file_tags()
        print("DEBUGGING: UI refreshed with updated tags")

    def remove_tag_from_file(self):
        """Remove selected tag(s) from the current file."""
        if not self.current_file_path:
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return

        selected_items = self.file_tags_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select tag(s) to remove!")
            return

        file_obj = (
            self.db_session.query(File).filter_by(path=self.current_file_path).first()
        )
        if file_obj:
            for item in selected_items:
                tag = self.db_session.query(Tag).filter_by(name=item.text()).first()
                if tag in file_obj.tags:
                    file_obj.tags.remove(tag)

            self._commit_legacy_changes(
                file_paths=[
                    (
                        self.current_file_path,
                        [(tag.name, tag.color) for tag in file_obj.tags],
                    )
                ]
            )
            self.refresh_file_tags()

    def suggest_tags(self):
        """Open the tag suggestion dialog for the current file."""
        if not self.current_file_path:
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return

        dialog = TagSuggestionDialog(
            self.config,
            self.db_session,
            self.current_file_path,
            self,
            modern_runtime=getattr(self, "modern_runtime", None),
        )
        if dialog.exec():
            self.refresh_file_tags()

    def search_by_tags(self):
        """Search for files with selected tags."""
        selected_items = self.search_tag_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select tag(s) to search for!")
            return

        # Build query
        query = self.db_session.query(File).distinct()
        tag_names = [item.text() for item in selected_items]

        if self.and_radio.isChecked():
            # Files must have ALL selected tags
            for tag_name in tag_names:
                tag = self.db_session.query(Tag).filter_by(name=tag_name).first()
                if tag:
                    query = query.filter(File.tags.contains(tag))
        else:
            # Files must have ANY of the selected tags
            tags = self.db_session.query(Tag).filter(Tag.name.in_(tag_names)).all()
            query = query.filter(File.tags.any(Tag.id.in_([t.id for t in tags])))

        # Display results
        self.search_results.clear()
        for file in query.all():
            if os.path.exists(file.path):
                item = QListWidgetItem(os.path.basename(file.path))
                item.setToolTip(file.path)
                self.search_results.addItem(item)

    def search_by_content(self):
        """Search for files using semantic search."""
        query = self.query_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a search query!")
            return

        # Get selected tags for filtering
        tag_filters = []
        for i in range(self.tag_filter_list.count()):
            item = self.tag_filter_list.item(i)
            if item.isSelected():
                tag_filters.append(item.text())

        # Perform search
        try:
            vector_search = self.vector_search
            if vector_search is None:
                from .vector_search import VectorSearch

                legacy_search = cast(Any, VectorSearch)
                vector_search = legacy_search(self.db_session, self.config)
                self.vector_search = vector_search
            results = vector_search.search(
                query,
                tag_filter=tag_filters,
                use_and=self.rag_and_radio.isChecked(),
                limit=20,
            )

            # Display results
            self.rag_search_results.clear()

            if not results or len(results) == 0:
                self.chat_results_btn.setEnabled(False)
                self.current_search_results = []
                return

            # Store all results for reference
            self.current_search_results = results

            # Keep track of which list items correspond to documents (not snippets)
            self.result_document_items = {}

            for result in results:
                if os.path.exists(result["path"]):
                    # Add file item with checkbox
                    score = result.get("score", 0)

                    # Create main file item
                    file_item = QListWidgetItem(
                        f"{os.path.basename(result['path'])} ({score:.2f})"
                    )
                    file_item.setToolTip(result["path"])
                    file_item.setBackground(get_score_color(score))
                    file_item.setFlags(
                        file_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    file_item.setCheckState(Qt.CheckState.Unchecked)
                    self.rag_search_results.addItem(file_item)

                    # Store reference to this item for selection tracking
                    self.result_document_items[result["path"]] = file_item
                    # Display document summary if available
                    summary = result.get("summary", "")
                    if summary:
                        # Wrap summary in HTML div to ensure proper text wrapping
                        formatted_summary = f"<div style='text-align:left; margin-left:4px; margin-right:4px;'>    📝 {summary}</div>"
                        summary_item = QListWidgetItem()
                        summary_item.setText(formatted_summary)
                        summary_item.setToolTip(result["path"])
                        # Use lighter background to differentiate from main result
                        bg_color = get_score_color(score)
                        # Make the background slightly lighter for summary
                        lighter_bg = QColor(
                            min(bg_color.red() + 15, 255),
                            min(bg_color.green() + 15, 255),
                            min(bg_color.blue() + 15, 255),
                        )
                        summary_item.setBackground(lighter_bg)
                        self.rag_search_results.addItem(summary_item)

                    # Display tags on a separate line with colored backgrounds for each tag
                    tags = result.get("tags", [])
                    if tags:
                        tag_display = "    ⚑ Tags: "
                        tags_item = QListWidgetItem(tag_display)
                        tags_item.setToolTip(result["path"])
                        # Use lighter background to differentiate from main result
                        bg_color = get_score_color(score)
                        # Make the background slightly lighter
                        lighter_bg = QColor(
                            min(bg_color.red() + 20, 255),
                            min(bg_color.green() + 20, 255),
                            min(bg_color.blue() + 20, 255),
                        )
                        tags_item.setBackground(lighter_bg)
                        self.rag_search_results.addItem(tags_item)

                        # Add each tag with its proper color
                        for tag_name in tags:
                            # Look up the tag color from the database
                            tag = (
                                self.db_session.query(Tag)
                                .filter_by(name=tag_name)
                                .first()
                            )
                            if tag:
                                tag_color = QColor(tag.color)
                                text_color = (
                                    Qt.white if is_dark_color(tag_color) else Qt.black
                                )

                                # Create a tag item with spacing for visual separation
                                tag_item = QListWidgetItem(f"        • {tag_name}")
                                tag_item.setToolTip(result["path"])
                                tag_item.setBackground(tag_color)
                                tag_item.setForeground(text_color)
                                self.rag_search_results.addItem(tag_item)

                    # Add snippet items if available
                    for snippet in result.get("snippets", []):
                        # Create a rich text item that can show bold formatting
                        snippet_item = QListWidgetItem()
                        snippet_item.setToolTip(result["path"])

                        # Check if this is already a formatted snippet with a context
                        if (
                            isinstance(snippet, str)
                            and snippet.startswith("[")
                            and "]" in snippet
                        ):
                            # Get context and text parts
                            context_end = snippet.find("]")
                            context = snippet[1:context_end]
                            text = snippet[context_end + 1 :].strip()

                            # Create formatted text with context prefix in italics
                            formatted_text = f"    ↪ <i>{context}:</i> {text}"
                        else:
                            # Just use the snippet text as-is
                            formatted_text = f"    ↪ {snippet}"
                        # Set the text with HTML formatting that preserves bold highlighting
                        # (the ** marks from markdown are converted to HTML <b> tags)
                        formatted_text = formatted_text.replace("**", "<b>", 1)
                        while "**" in formatted_text:
                            formatted_text = formatted_text.replace("**", "</b>", 1)

                        # Add div with styling for better text wrapping
                        formatted_text = f"<div style='text-align:left; margin-left:4px; margin-right:4px;'>{formatted_text}</div>"

                        snippet_item.setText(formatted_text)

                        # Apply lighter background color for snippet items
                        lighter_bg = QColor(
                            min(bg_color.red() + 40, 255),
                            min(bg_color.green() + 40, 255),
                            min(bg_color.blue() + 40, 255),
                        )
                        snippet_item.setBackground(lighter_bg)

                        # Add to results list
                        self.rag_search_results.addItem(snippet_item)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Search failed: {str(e)}")
            self.chat_results_btn.setEnabled(False)
            self.current_search_results = []
            self.result_document_items = {}

    def reindex_files(self):
        """Queue all known files through the resumable modern indexing service."""
        reply = QMessageBox.question(
            self,
            "Reindex Files",
            "Are you sure you want to reindex all files?\n\n"
            "This will rebuild the search index for all files in the database.\n"
            "This operation may take a while depending on the number of files.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            file_paths = [
                file_obj.path for file_obj in self.db_session.query(File).all()
            ]
            if not self._index_modern_files(file_paths):
                QMessageBox.warning(
                    self,
                    "Error",
                    self.modern_index_status_text,
                )

    def resume_indexing(self) -> bool:
        """Resume the most recent failed or cancelled modern indexing job."""
        indexing_status = getattr(self, "indexing_status", None)
        if indexing_status is None:
            self._set_modern_index_status(
                "Modern indexing resume is unavailable."
            )
            return False
        try:
            if callable(getattr(indexing_status, "setVisible", None)):
                indexing_status.setVisible(True)
            return indexing_status.resume() is not None
        except Exception as exc:
            logger = getattr(self, "logger", logging.getLogger(__name__))
            logger.warning(
                "Could not resume modern indexing (%s).",
                exc.__class__.__name__,
            )
            return False

    def chat_with_results(self):
        """Open the chat dialog with selected documents (up to 3)."""
        try:
            # Get selected documents
            selected_docs = []
            checked_count = 0

            # Check if we have document items and search results
            if hasattr(self, "result_document_items") and hasattr(
                self, "current_search_results"
            ):
                # Count number of checked items and get their corresponding results
                for file_path, item in self.result_document_items.items():
                    if item.checkState() == Qt.CheckState.Checked:
                        checked_count += 1
                        # Find the matching result in current_search_results
                        for result in self.current_search_results:
                            if result["path"] == file_path:
                                selected_docs.append(result)
                                break

            # If no documents are specifically selected, use the first 3 from search results
            if (
                not selected_docs
                and hasattr(self, "current_search_results")
                and self.current_search_results
            ):
                # Limit to first 3 documents if none are selected
                selected_docs = self.current_search_results[
                    : min(3, len(self.current_search_results))
                ]
                message = "No documents selected. Using top search results."
                QMessageBox.information(self, "Information", message)

            if not selected_docs:
                QMessageBox.warning(
                    self,
                    "No Results",
                    "No search results available. Please perform a search first.",
                )
                return

            # Check if more than 3 documents are selected
            if checked_count > 3:
                QMessageBox.warning(
                    self,
                    "Too Many Documents",
                    "You can select up to 3 documents. Please uncheck some documents.",
                )
                return

            QMessageBox.information(
                self,
                "Local Search Chat Unavailable",
                "Search-result chat is not available in the current local-first "
                "workflow. Tag decisions use the installed Ollama System One model.",
            )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not start chat: {str(e)}\nPlease check Ollama readiness and try again.",
            )

    def migration_inventory(self) -> dict[str, object]:
        """Return a safe inventory of migrated and still-legacy search data."""
        runtime = getattr(self, "modern_runtime", None)
        paths = getattr(runtime, "paths", None) or getattr(
            getattr(self, "config", None),
            "paths",
            None,
        )
        legacy_chroma = getattr(paths, "legacy_chroma", None)
        chroma_found = bool(
            legacy_chroma is not None and legacy_chroma.exists()
        )
        chroma_paths: tuple[str, ...] = ()
        chroma_warnings: tuple[str, ...] = ()
        if legacy_chroma is not None:
            try:
                plan = inspect_legacy_chroma(legacy_chroma)
                chroma_found = plan.found
                chroma_paths = plan.paths
                chroma_warnings = plan.warnings
            except (OSError, ValueError) as exc:
                chroma_warnings = (
                    f"Legacy search inventory failed: {exc.__class__.__name__}.",
                )

        sqlite = getattr(runtime, "sqlite_migration", None)
        config = getattr(runtime, "config_migration", None)
        return {
            "legacy_chroma_found": chroma_found,
            "legacy_chroma_path_count": len(chroma_paths),
            "legacy_chroma_warnings": chroma_warnings,
            "requires_full_reindex": bool(chroma_paths),
            "files_seen": int(getattr(sqlite, "files_seen", 0)),
            "files_migrated": int(getattr(sqlite, "files_migrated", 0)),
            "tags_migrated": int(getattr(sqlite, "tags_migrated", 0)),
            "associations_migrated": int(
                getattr(sqlite, "associations_migrated", 0)
            ),
            "suggestions_preserved": int(
                getattr(sqlite, "suggestions_preserved", 0)
            ),
            "cloud_fields_removed": int(
                getattr(config, "cloud_fields_removed", 0)
            ),
            "migration_errors": (
                tuple(getattr(sqlite, "errors", ()))
                + tuple(getattr(config, "errors", ()))
            ),
        }

    def show_migration_inventory(self) -> None:
        inventory = self.migration_inventory()
        warnings = cast(tuple[str, ...], inventory["legacy_chroma_warnings"])
        errors = cast(tuple[str, ...], inventory["migration_errors"])
        details = [
            (
                f"Legacy search data found: "
                f"{'yes' if inventory['legacy_chroma_found'] else 'no'}"
            ),
            (
                "Legacy paths requiring full reindex: "
                f"{inventory['legacy_chroma_path_count']}"
            ),
            f"Files inspected: {inventory['files_seen']}",
            f"Files migrated: {inventory['files_migrated']}",
            f"Tags migrated: {inventory['tags_migrated']}",
            (
                "File/tag associations migrated: "
                f"{inventory['associations_migrated']}"
            ),
            (
                "Tag suggestions preserved: "
                f"{inventory['suggestions_preserved']}"
            ),
            (
                "Removed legacy cloud fields: "
                f"{inventory['cloud_fields_removed']}"
            ),
        ]
        if warnings:
            details.append("Legacy search warnings: " + "; ".join(warnings))
        if errors:
            details.append("Migration warnings: " + "; ".join(errors))
        QMessageBox.information(
            self,
            "Migration Inventory",
            "\n".join(details),
        )

    def cleanup_legacy_search_data(self) -> bool:
        """Delete only the inspected legacy Chroma directory after confirmation."""
        runtime = getattr(self, "modern_runtime", None)
        paths = getattr(runtime, "paths", None) or getattr(
            getattr(self, "config", None),
            "paths",
            None,
        )
        legacy_chroma = getattr(paths, "legacy_chroma", None)
        if legacy_chroma is None or not legacy_chroma.exists():
            QMessageBox.information(
                self,
                "Legacy Search Cleanup",
                "No legacy search directory is available to remove.",
            )
            return False

        inventory = self.migration_inventory()
        reply = QMessageBox.question(
            self,
            "Clean Legacy Search Data",
            "Remove the legacy .chroma search data?\n\n"
            f"{inventory['legacy_chroma_path_count']} file path(s) were inventoried. "
            "Legacy embeddings are not reused, and the modern index must be rebuilt. "
            "Migration backups remain untouched.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            QMessageBox.information(
                self,
                "Legacy Search Cleanup",
                "Legacy search cleanup was cancelled; no data was removed.",
            )
            return False
        try:
            removed = cleanup_legacy_chroma(legacy_chroma, confirmed=True)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(
                self,
                "Legacy Search Cleanup",
                f"Cleanup failed safely: {exc}",
            )
            return False
        QMessageBox.information(
            self,
            "Legacy Search Cleanup",
            "Legacy search data was removed."
            if removed
            else "The legacy search data was already absent.",
        )
        return removed

    def show_ollama_settings(self) -> None:
        """Show the local Ollama readiness and installed-model dialog."""
        runtime = getattr(self, "modern_runtime", None)
        report = getattr(runtime, "readiness", None)
        if not isinstance(report, ReadinessReport):
            reason = (
                runtime.unavailable_reason
                if runtime is not None
                else "the modern Ollama runtime is not initialized."
            )
            report = ReadinessReport(
                ready=False,
                blocked_roles=("indexing", "search", "tagging"),
                instructions=(f"Ollama readiness unavailable: {reason}",),
                errors=(str(reason),),
            )
        dialog = OllamaSettingsDialog(
            report,
            runtime_refresh_provider=runtime,
            parent=self,
        )
        result = dialog.exec()
        if (
            result == QDialog.DialogCode.Accepted
            and runtime is not None
            and callable(getattr(runtime, "apply_model_selections", None))
        ):
            try:
                runtime.apply_model_selections(dialog.selected_models())
            except (TypeError, ValueError) as exc:
                self.logger.warning("Could not apply Ollama model selections: %s", exc)
                QMessageBox.warning(
                    self,
                    "Ollama Models",
                    f"Could not apply the selected models: {exc}",
                )

    def show_password_management(self):
        """Show the password management dialog."""
        dialog = PasswordManagementDialog(self, self.config)
        dialog.exec()

    def set_home_directory(self):
        """Set the home directory for file browsing."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Home Directory",
            self.config.get_home_directory(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if dir_path:
            self.config.set_home_directory(dir_path)
            self.go_home()

    def scan_directory_for_untagged(self):
        """Scan a directory for untagged files."""
        # Use the current directory from the file explorer
        dir_path = self.path_display.text()

        if not dir_path or not os.path.isdir(dir_path):
            QMessageBox.warning(
                self, "Error", "Please navigate to a valid directory first!"
            )
            return

        # Import threading here to avoid circular imports
        from PySide6.QtCore import QThread, Signal

        class ScanThread(QThread):
            """Thread to scan directory for untagged files without blocking UI."""

            scan_progress = Signal(int, int)  # files_processed, total_files
            scan_finished = Signal(
                dict
            )  # dictionary mapping file paths to tag suggestions
            scan_error = Signal(str)  # error message

            def __init__(self, directory, db_session, config, modern_runtime):
                super().__init__()
                self.directory = directory
                self.db_session = db_session
                self.config = config
                self.modern_runtime = modern_runtime
                self.stop_requested = False

            def run(self):
                try:
                    # List all files in directory (not subdirectories)
                    files = []
                    with os.scandir(self.directory) as entries:
                        for entry in entries:
                            if entry.is_file() and not entry.name.startswith("."):
                                files.append(entry.path)

                    # Initialize progress
                    total_files = len(files)
                    if total_files == 0:
                        self.scan_finished.emit({})
                        return

                    # Check which files are not in database or have no tags
                    # and generate suggestions for each file
                    from .tag_suggestion import TagSuggester

                    tag_suggester = TagSuggester(
                        self.config,
                        modern_runtime=getattr(self, "modern_runtime", None),
                    )

                    untagged_files = {}
                    for idx, file_path in enumerate(files):
                        if self.stop_requested:
                            return

                        file_obj = (
                            self.db_session.query(File)
                            .filter_by(path=file_path)
                            .first()
                        )
                        if not file_obj or not file_obj.tags:
                            # Get tag suggestions for this specific file
                            suggestions = tag_suggester.suggest_tags_for_file(file_path)
                            untagged_files[file_path] = suggestions

                        # Emit progress
                        self.scan_progress.emit(idx + 1, total_files)

                    # Emit result with file-specific suggestions
                    self.scan_finished.emit(untagged_files)

                except Exception as e:
                    self.scan_error.emit(str(e))

        # Create progress dialog
        progress = QProgressDialog(
            "Scanning directory for untagged files...", "Cancel", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Create and configure scan thread
        self.scan_thread = ScanThread(
            dir_path,
            self.db_session,
            self.config,
            getattr(self, "modern_runtime", None),
        )

        # Connect signals
        self.scan_thread.scan_progress.connect(
            lambda current, total: progress.setValue(int(current / total * 100))
        )
        self.scan_thread.scan_finished.connect(
            lambda files: self.show_untagged_files_dialog(files, dir_path)
        )
        self.scan_thread.scan_error.connect(
            lambda error: QMessageBox.critical(
                self, "Scan Error", f"Error scanning directory: {error}"
            )
        )
        progress.canceled.connect(
            lambda: setattr(self.scan_thread, "stop_requested", True)
        )

        # Start the scan
        self.scan_thread.start()

    def show_untagged_files_dialog(self, untagged_files, directory_path):
        """Show dialog with untagged files and provide options to tag them."""
        if not untagged_files:
            QMessageBox.information(
                self,
                "Scan Complete",
                f"No untagged files found in directory:\n{directory_path}",
            )
            return

        # Create the dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Untagged Files")
        dialog.setMinimumWidth(900)
        dialog.setMinimumHeight(600)
        layout = QVBoxLayout(dialog)

        # Add header information
        header_label = QLabel(
            f"Found {len(untagged_files)} untagged files in:\n{directory_path}"
        )
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        # Create main splitter for files and tags
        main_layout = QVBoxLayout()

        # File list section (top section)
        file_layout = QVBoxLayout()
        file_label = QLabel("Select a file to view its suggestions:")
        file_layout.addWidget(file_label)

        self.untagged_files_list = QListWidget()
        # Store file paths and their suggestions
        self.file_suggestions_map = untagged_files

        for file_path in untagged_files.keys():
            item = QListWidgetItem(os.path.basename(file_path))
            item.setToolTip(file_path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.untagged_files_list.addItem(item)

        file_layout.addWidget(self.untagged_files_list)

        # Selection buttons
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        selection_layout.addWidget(select_all_btn)
        selection_layout.addWidget(select_none_btn)
        file_layout.addLayout(selection_layout)

        # Connect selection buttons
        select_all_btn.clicked.connect(
            lambda: [
                self.untagged_files_list.item(i).setCheckState(Qt.CheckState.Checked)
                for i in range(self.untagged_files_list.count())
            ]
        )
        select_none_btn.clicked.connect(
            lambda: [
                self.untagged_files_list.item(i).setCheckState(Qt.CheckState.Unchecked)
                for i in range(self.untagged_files_list.count())
            ]
        )

        main_layout.addLayout(file_layout)

        # Create a horizontal layout for the file-specific suggestions and general tag selection
        tags_container_layout = QHBoxLayout()

        # Tag suggestions section (left side)
        suggestions_layout = QVBoxLayout()
        suggestions_label = QLabel("File-Specific Tag Suggestions:")
        suggestions_layout.addWidget(suggestions_label)

        self.file_suggestions_list = QListWidget()
        self.file_suggestions_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        suggestions_layout.addWidget(self.file_suggestions_list)

        # Connect file list selection to update suggestions
        self.untagged_files_list.currentItemChanged.connect(
            self.update_file_specific_suggestions
        )

        tags_container_layout.addLayout(suggestions_layout)

        # General tag selection section (right side)
        tag_layout = QVBoxLayout()
        tag_layout.addWidget(QLabel("Available Tags:"))

        tag_list = QListWidget()
        tag_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        # Add existing tags to the list
        tags = self.db_session.query(Tag).all()
        for tag in tags:
            tag_color = QColor(tag.color)
            text_color = Qt.white if is_dark_color(tag_color) else Qt.black

            tag_list.addItem(tag.name)
            item = tag_list.item(tag_list.count() - 1)
            item.setBackground(tag_color)
            item.setForeground(text_color)

        tag_layout.addWidget(tag_list)

        # Add new tag button
        add_tag_btn = QPushButton("Create New Tag")
        add_tag_btn.clicked.connect(lambda: self.add_tag_from_dialog(tag_list))
        tag_layout.addWidget(add_tag_btn)

        tags_container_layout.addLayout(tag_layout)
        main_layout.addLayout(tags_container_layout)

        layout.addLayout(main_layout)

        # Add buttons at the bottom
        btn_layout = QHBoxLayout()
        apply_selected_btn = QPushButton("Apply Selected Tags to Selected Files")
        apply_all_suggestions_btn = QPushButton(
            "Apply All Suggestions to Selected Files"
        )
        cancel_btn = QPushButton("Close")
        btn_layout.addWidget(apply_selected_btn)
        btn_layout.addWidget(apply_all_suggestions_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # Connect buttons
        apply_selected_btn.clicked.connect(
            lambda: self.apply_tags_to_files(
                self.get_selected_files_from_list(self.untagged_files_list),
                [item.text() for item in tag_list.selectedItems()]
                + [
                    self.get_tag_from_suggestion_item(item)
                    for item in self.file_suggestions_list.selectedItems()
                ],
            )
        )

        apply_all_suggestions_btn.clicked.connect(
            lambda: self.apply_all_suggestions_to_files(
                self.get_selected_files_from_list(self.untagged_files_list)
            )
        )

        cancel_btn.clicked.connect(dialog.accept)

        # Show dialog
        if self.untagged_files_list.count() > 0:
            self.untagged_files_list.setCurrentRow(0)

        dialog.exec()

    def get_tag_from_suggestion_item(self, item):
        """Extract the tag name from a suggestion list item."""
        text = item.text()
        # Remove the confidence score part if present, e.g., "tag (0.95)" -> "tag"
        if "(" in text:
            return text.split("(")[0].strip()
        return text

    def update_file_specific_suggestions(self, current, previous):
        """Update the file-specific suggestions list based on the selected file."""
        if not current:
            return

        self.file_suggestions_list.clear()
        file_path = current.toolTip()

        if file_path in self.file_suggestions_map:
            suggestions = self.file_suggestions_map[file_path]

            # Sort suggestions by confidence score
            sorted_suggestions = dict(
                sorted(suggestions.items(), key=lambda item: item[1], reverse=True)
            )

            # Add suggestions to the list
            for tag_name, confidence in sorted_suggestions.items():
                # Format item text with confidence
                text = f"{tag_name} ({confidence:.2f})"

                # Create the item
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, tag_name)  # Store actual tag name

                # Set background color based on confidence
                item.setBackground(get_score_color(confidence))

                # Add to list
                self.file_suggestions_list.addItem(item)

    def apply_all_suggestions_to_files(self, file_paths):
        """Apply all AI suggestions to the selected files."""
        if not file_paths:
            QMessageBox.warning(self, "Error", "Please select files to tag!")
            return

        applied_count = 0
        total_files = len(file_paths)

        applied_paths = []
        for file_path in file_paths:
            if file_path in self.file_suggestions_map:
                suggestions = self.file_suggestions_map[file_path]

                if suggestions:
                    # Filter suggestions with good confidence (above 0.7)
                    good_suggestions = {
                        tag: score for tag, score in suggestions.items() if score > 0.7
                    }

                    if good_suggestions:
                        # Apply these tags
                        self.apply_tags_to_file(file_path, good_suggestions.keys())
                        applied_paths.append(file_path)
                        applied_count += 1

        if applied_paths:
            self._commit_legacy_changes(
                file_paths=[
                    (
                        file_path,
                        [
                            (tag.name, tag.color)
                            for tag in (
                                self.db_session.query(File)
                                .filter_by(path=file_path)
                                .one()
                                .tags
                            )
                        ],
                    )
                    for file_path in applied_paths
                ]
            )
            self._index_modern_files(applied_paths)

        # Show success message
        if applied_count > 0:
            QMessageBox.information(
                self,
                "Tags Applied",
                f"Applied high-confidence suggestions to {applied_count} of {total_files} selected files.",
            )
        else:
            QMessageBox.information(
                self,
                "No Tags Applied",
                "No high-confidence suggestions were found for the selected files.",
            )

    def apply_tags_to_file(self, file_path, tag_names):
        """Apply tags to a single file."""
        if not tag_names:
            return

        # Get or create file record
        file_obj = self.db_session.query(File).filter_by(path=file_path).first()

        if not file_obj:
            file_obj = File(path=file_path)
            self.db_session.add(file_obj)

        # Add tags (create them if they don't exist)
        for tag_name in tag_names:
            tag = self.db_session.query(Tag).filter_by(name=tag_name).first()
            if not tag:
                # Create a new tag with a random color
                import random

                hue = random.randint(0, 359)
                saturation = random.randint(128, 255)  # Medium to high saturation
                value = random.randint(180, 255)  # Medium to high brightness
                random_color = QColor.fromHsv(hue, saturation, value).name()

                tag = Tag(name=tag_name, color=random_color)
                self.db_session.add(tag)
                self.db_session.flush()  # Generate ID without committing transaction

            if tag not in file_obj.tags:
                file_obj.tags.append(tag)

        # The caller commits the legacy snapshot and queues modern indexing.

    def apply_tags_to_files(self, file_paths, tag_names):
        """Apply selected tags to multiple files."""
        if not file_paths:
            QMessageBox.warning(self, "Error", "Please select files to tag!")
            return

        if not tag_names:
            QMessageBox.warning(self, "Error", "Please select tags to apply!")
            return

        # Apply tags to each file
        for file_path in file_paths:
            self.apply_tags_to_file(file_path, tag_names)

        # Commit all changes at once
        self._commit_legacy_changes(
            file_paths=[
                (
                    file_path,
                    [
                        (tag.name, tag.color)
                        for tag in (
                            self.db_session.query(File)
                            .filter_by(path=file_path)
                            .one()
                            .tags
                        )
                    ],
                )
                for file_path in file_paths
            ]
        )
        self._index_modern_files(file_paths)

        # Refresh tag lists
        self.refresh_tags()
        self.refresh_file_tags()

        # Show success message
        QMessageBox.information(
            self,
            "Tags Applied",
            f"Applied {len(tag_names)} tag(s) to {len(file_paths)} file(s).",
        )

    def get_selected_files_from_list(self, list_widget):
        """Get file paths from checked items in a list widget."""
        selected_files = []

        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                file_path = item.toolTip()
                selected_files.append(file_path)

        return selected_files

    def force_reindex_file(self, file_path):
        """Queue one file for atomic, resumable modern indexing."""
        if not file_path or not os.path.isfile(file_path):
            QMessageBox.warning(
                self,
                "Error",
                f"The file {file_path} does not exist or is not a valid file.",
            )
            return False

        try:
            # Get the file from database
            file_obj = self.db_session.query(File).filter_by(path=file_path).first()
            if not file_obj:
                QMessageBox.warning(
                    self, "Error", f"The file {file_path} is not in the tag database."
                )
                return False

            return self._index_modern_file(file_path)

        except Exception as e:
            print(f"Error during force reindex: {str(e)}")
            import traceback

            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to reindex file: {str(e)}")
            return False

    def set_pdf_extractor(self, preference):
        """Set the PDF extractor preference."""
        try:
            current = self.config.get_pdf_extractor()
            if current == preference:
                return  # No change needed

            # Update configuration
            self.config.set_pdf_extractor(preference)

            # Update action states - use the stored action references
            # instead of trying to access the menu directly
            self.fast_action.setChecked(preference == "fast")
            self.accurate_action.setChecked(preference == "accurate")

            # Show confirmation message
            if preference == "fast":
                QMessageBox.information(
                    self,
                    "PDF Extractor Changed",
                    "PDF extractor set to Fast mode (using markitdown).\n\n"
                    "This setting will be used for all future PDF and Word document extraction.",
                )
            else:
                QMessageBox.information(
                    self,
                    "PDF Extractor Changed",
                    "PDF extractor set to Accurate mode (using docling).\n\n"
                    "This setting will be used for all future PDF and Word document extraction.",
                )

        except Exception as e:
            self.logger.error(f"Error setting PDF extractor preference: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.warning(
                self, "Error", f"Failed to change PDF extractor preference: {str(e)}"
            )
