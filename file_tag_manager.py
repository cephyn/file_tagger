import os
import sys
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTreeView, QListWidget, QPushButton, QInputDialog,
                             QColorDialog, QLabel, QFileSystemModel, QMessageBox,
                             QLineEdit, QRadioButton, QComboBox, QHeaderView,
                             QMenuBar, QMenu, QFileDialog, QTabWidget, QProgressDialog,
                             QFrame, QListWidgetItem, QDialog)
from PySide6.QtCore import Qt, QDir, QStorageInfo
from PySide6.QtGui import QColor, QFont, QDesktopServices
from PySide6.QtCore import QUrl
from sqlalchemy import and_, or_
from models import File, Tag
from config import Config
from vector_search import VectorSearch
from api_settings import APISettingsDialog
from password_management import PasswordManagementDialog
from tag_suggestion import TagSuggestionDialog
from search import ChatWithResultsDialog
from utils import is_dark_color, get_score_color, open_file, open_containing_folder

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
    def __init__(self, db_session, config, vector_search):
        super().__init__()
        self.db_session = db_session
        self.config = config
        self.vector_search = vector_search
        self.current_file_path = None
        self.current_search_results = []  # Initialize search results storage
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('File Tagger')
        self.setGeometry(100, 100, 1200, 700)
        
        # Create menu bar and menus
        self.setup_menus()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create the main tab widget
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)
        
        # Create and add the tagging and search tabs
        tagging_tab = self.create_tagging_tab()
        search_tab = self.create_search_tab()
        
        self.main_tabs.addTab(tagging_tab, "Tagging Interface")
        self.main_tabs.addTab(search_tab, "Search Interface")
        
        # Set initial directory from config
        initial_path = self.config.get_home_directory()
        self.tree.setRootIndex(self.model.index(initial_path))
        
        # Update initial path display
        self.path_display.setText(initial_path)
        
        # Update drive list and select current drive
        self.update_drive_list()
        drive = os.path.splitdrive(initial_path)[0] + os.path.sep
        for i in range(self.drive_combo.count()):
            if self.drive_combo.itemData(i).startswith(drive):
                self.drive_combo.setCurrentIndex(i)
                break
        
        # Connect signals after initialization
        self.drive_combo.currentIndexChanged.connect(self.on_drive_changed)
        self.refresh_tags()

    def setup_menus(self):
        """Set up the application menus."""
        menubar = self.menuBar()
        settings_menu = menubar.addMenu('Settings')
        help_menu = menubar.addMenu('Help')
        
        # Add settings menu actions
        api_settings_action = settings_menu.addAction('API Settings')
        api_settings_action.triggered.connect(self.show_api_settings)
        
        home_dir_action = settings_menu.addAction('Set Home Directory')
        home_dir_action.triggered.connect(self.set_home_directory)
        
        password_action = settings_menu.addAction('Password Management')
        password_action.triggered.connect(self.show_password_management)
        
        # Add scan directory action
        scan_action = menubar.addAction('Scan Directory')
        scan_action.triggered.connect(self.scan_directory_for_untagged)
        
        # Add help menu actions
        about_action = help_menu.addAction('About')
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
        self.file_tags_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
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
        self.search_tag_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
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
        self.search_results.itemDoubleClicked.connect(self.on_search_result_double_clicked)
        self.search_results.setContextMenuPolicy(Qt.CustomContextMenu)
        self.search_results.customContextMenuRequested.connect(self.on_search_result_right_clicked)
        results_layout.addWidget(self.search_results)
        
        # Add layouts to tab
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
        self.rag_search_results.itemDoubleClicked.connect(self.on_search_result_double_clicked)
        self.rag_search_results.setContextMenuPolicy(Qt.CustomContextMenu)
        self.rag_search_results.customContextMenuRequested.connect(self.on_search_result_right_clicked)
        results_layout.addWidget(self.rag_search_results)
        
        # Add buttons layout
        buttons_layout = QHBoxLayout()
        
        # Reindex button
        reindex_btn = QPushButton("Reindex Files")
        reindex_btn.clicked.connect(self.reindex_files)
        buttons_layout.addWidget(reindex_btn)
        
        # Chat with results button
        chat_btn = QPushButton("Chat with Results")
        chat_btn.clicked.connect(self.chat_with_results)
        chat_btn.setEnabled(False)  # Disabled until results are available
        self.chat_results_btn = chat_btn  # Store reference to enable/disable
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
            
        file_obj = self.db_session.query(File).filter_by(path=self.current_file_path).first()
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
                
                dir_path = os.path.dirname(file_path) if os.path.isfile(file_path) else file_path
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
        """Remove a file from the vector database."""
        reply = QMessageBox.question(
            self, 
            "Remove from Search Index",
            f"Are you sure you want to remove\n{file_path}\nfrom the search index?\n\n"
            "Note: This only removes the file from the search index, not from your computer or the tag database.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.vector_search.remove_file(file_path)
            
            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    "File removed from search index successfully.\n\n"
                    "The file is still on your computer and in the tag database."
                )
                self._remove_item_from_results(file_path)
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Could not remove file from search index. See console for details."
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
                QMessageBox.warning(self, "Error", "A tag with this name already exists!")
                return
            
            # Get color for the tag
            color = QColorDialog.getColor()
            if color.isValid():
                # Create new tag
                new_tag = Tag(name=tag_name, color=color.name())
                self.db_session.add(new_tag)
                self.db_session.commit()
                
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
        new_name, ok = QInputDialog.getText(
            self, "Edit Tag", "Enter new tag name:", text=tag.name
        )
        if not ok or not new_name:
            return
            
        # Check if new name already exists (if different from current)
        if new_name != tag.name:
            existing = self.db_session.query(Tag).filter_by(name=new_name).first()
            if existing:
                QMessageBox.warning(self, "Error", "A tag with this name already exists!")
                return
        
        # Get new color
        initial_color = QColor(tag.color)
        color = QColorDialog.getColor(initial=initial_color)
        if color.isValid():
            # Update tag
            tag.name = new_name
            tag.color = color.name()
            self.db_session.commit()
            
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
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for item in selected_items:
                tag = self.db_session.query(Tag).filter_by(name=item.text()).first()
                if tag:
                    self.db_session.delete(tag)
            
            self.db_session.commit()
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
        file_obj = self.db_session.query(File).filter_by(path=self.current_file_path).first()
        is_new_file = False
        if not file_obj:
            print("DEBUGGING: File not found in database, creating new record")
            file_obj = File(path=self.current_file_path)
            self.db_session.add(file_obj)
            is_new_file = True
        else:
            print(f"DEBUGGING: File found in database with {len(file_obj.tags)} existing tags")
        
        # Track if file had tags before this operation
        had_tags_before = len(file_obj.tags) > 0
        print(f"DEBUGGING: had_tags_before = {had_tags_before}, is_new_file = {is_new_file}")
        
        # Add selected tags
        tags_added = []
        for item in selected_items:
            tag = self.db_session.query(Tag).filter_by(name=item.text()).first()
            if tag and tag not in file_obj.tags:
                file_obj.tags.append(tag)
                tags_added.append(tag.name)
        
        print(f"DEBUGGING: Added tags: {tags_added}")
        
        # Perform database commit - this will generate ID for new files
        self.db_session.commit()
        print("DEBUGGING: Database commit completed")
        
        # IMPORTANT: Check if this triggers vector indexing
        print("DEBUGGING: Checking if file should be indexed")
        if is_new_file or not had_tags_before:
            print("DEBUGGING: File should be indexed (new file or first tags)")
            try:
                from vector_search.content_extractor import ContentExtractor
                # Extract content from the file
                print(f"DEBUGGING: Extracting content from: {self.current_file_path}")
                content = ContentExtractor.extract_file_content(self.current_file_path)
                
                if content:
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"DEBUGGING: Content extracted, length: {len(content)} characters")
                    print(f"DEBUGGING: Content preview: {content_preview}")
                    
                    # Add file to vector database - this is the critical step!
                    print(f"DEBUGGING: Calling vector_search.index_file for: {self.current_file_path}")
                    try:
                        self.vector_search.index_file(self.current_file_path, content)
                        print(f"DEBUGGING: Successfully indexed file in vector search: {self.current_file_path}")
                        
                        # Verify the file was actually indexed
                        try:
                            results = self.vector_search.collection.get(
                                ids=[self.current_file_path], include=['metadatas']
                            )
                            if results and results['ids'] and len(results['ids']) > 0:
                                print(f"DEBUGGING: Verification successful - file found in vector store")
                                print(f"DEBUGGING: File metadata: {results['metadatas'][0]}")
                            else:
                                print(f"DEBUGGING: VERIFICATION FAILED - file not found in vector store after indexing!")
                        except Exception as verify_err:
                            print(f"DEBUGGING: Error verifying file in vector store: {str(verify_err)}")
                    except Exception as index_err:
                        print(f"DEBUGGING: Error in index_file method: {str(index_err)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"DEBUGGING: WARNING - No content could be extracted from file: {self.current_file_path}")
            except Exception as e:
                import traceback
                print(f"DEBUGGING: Error adding file to vector search: {str(e)}")
                print("DEBUGGING: Detailed error information:")
                traceback.print_exc()
        else:
            # If the file was already tagged before, just update the tags metadata
            print("DEBUGGING: File already had tags, updating metadata only")
            try:
                self.vector_search.update_metadata(self.current_file_path)
                print(f"DEBUGGING: Updated tags metadata for file in vector search: {self.current_file_path}")
            except Exception as e:
                import traceback
                print(f"DEBUGGING: Error updating vector search metadata: {str(e)}")
                traceback.print_exc()
        
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
        
        file_obj = self.db_session.query(File).filter_by(path=self.current_file_path).first()
        if file_obj:
            for item in selected_items:
                tag = self.db_session.query(Tag).filter_by(name=item.text()).first()
                if tag in file_obj.tags:
                    file_obj.tags.remove(tag)
            
            self.db_session.commit()
            self.refresh_file_tags()
    
    def suggest_tags(self):
        """Open the tag suggestion dialog for the current file."""
        if not self.current_file_path:
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return
        
        dialog = TagSuggestionDialog(self.config, self.db_session, self.current_file_path, self)
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
            results = self.vector_search.search(
                query,
                tag_filter=tag_filters,
                use_and=self.rag_and_radio.isChecked(),
                limit=20
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
                if os.path.exists(result['path']):
                    # Add file item with checkbox
                    score = result.get('score', 0)
                    file_item = QListWidgetItem(f"{os.path.basename(result['path'])} ({score:.2f})")
                    file_item.setToolTip(result['path'])
                    file_item.setBackground(get_score_color(score))
                    file_item.setFlags(file_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    file_item.setCheckState(Qt.CheckState.Unchecked)
                    self.rag_search_results.addItem(file_item)
                    
                    # Store reference to this item for selection tracking
                    self.result_document_items[result['path']] = file_item
                    
                    # Add snippet items if available
                    for snippet in result.get('snippets', []):
                        snippet_item = QListWidgetItem(f"    ↪ {snippet}")
                        snippet_item.setToolTip(result['path'])
                        self.rag_search_results.addItem(snippet_item)
            
            # Enable chat button if results are available
            self.chat_results_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Search failed: {str(e)}")
            self.chat_results_btn.setEnabled(False)
            self.current_search_results = []
            self.result_document_items = {}
    
    def reindex_files(self):
        """Reindex all files in the vector database."""
        reply = QMessageBox.question(
            self,
            "Reindex Files",
            "Are you sure you want to reindex all files?\n\n"
            "This will rebuild the search index for all files in the database.\n"
            "This operation may take a while depending on the number of files.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            progress = QProgressDialog("Reindexing files...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            
            try:
                self.vector_search.reindex_all(
                    progress_callback=lambda msg, p: progress.setValue(p)
                )
                QMessageBox.information(self, "Success", "Files reindexed successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Reindexing failed: {str(e)}")
            finally:
                progress.close()
    
    def chat_with_results(self):
        """Open the chat dialog with selected documents (up to 3)."""
        try:
            # Get selected documents
            selected_docs = []
            checked_count = 0
            
            # Check if we have document items and search results
            if hasattr(self, 'result_document_items') and hasattr(self, 'current_search_results'):
                # Count number of checked items and get their corresponding results
                for file_path, item in self.result_document_items.items():
                    if item.checkState() == Qt.CheckState.Checked:
                        checked_count += 1
                        # Find the matching result in current_search_results
                        for result in self.current_search_results:
                            if result['path'] == file_path:
                                selected_docs.append(result)
                                break
            
            # If no documents are specifically selected, use the first 3 from search results
            if not selected_docs and hasattr(self, 'current_search_results') and self.current_search_results:
                # Limit to first 3 documents if none are selected
                selected_docs = self.current_search_results[:min(3, len(self.current_search_results))]
                message = "No documents selected. Using top search results."
                QMessageBox.information(self, "Information", message)
            
            if not selected_docs:
                QMessageBox.warning(
                    self,
                    "No Results",
                    "No search results available. Please perform a search first."
                )
                return
                
            # Check if more than 3 documents are selected
            if checked_count > 3:
                QMessageBox.warning(
                    self,
                    "Too Many Documents",
                    "You can select up to 3 documents. Please uncheck some documents."
                )
                return
                
            # Get AI service
            ai_service = self.config.get_ai_service()
            if ai_service is None:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Could not initialize AI service. Please check your API settings and try again."
                )
                return
            
            # Open the chat dialog with selected documents
            dialog = ChatWithResultsDialog(self, ai_service, selected_docs, self.query_input.text().strip())
            dialog.exec()
                
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not start chat: {str(e)}\nPlease check your API settings and try again."
            )
    
    def show_api_settings(self):
        """Show the API settings dialog."""
        dialog = APISettingsDialog(config=self.config, parent=self)
        dialog.exec()
    
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
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if dir_path:
            self.config.set_home_directory(dir_path)
            self.go_home()
    
    def scan_directory_for_untagged(self):
        """Scan a directory for untagged files."""
        # Use the current directory from the file explorer
        dir_path = self.path_display.text()
        
        if not dir_path or not os.path.isdir(dir_path):
            QMessageBox.warning(self, "Error", "Please navigate to a valid directory first!")
            return
        
        # Import threading here to avoid circular imports
        from PySide6.QtCore import QThread, Signal
        
        class ScanThread(QThread):
            """Thread to scan directory for untagged files without blocking UI."""
            scan_progress = Signal(int, int)  # files_processed, total_files
            scan_finished = Signal(dict)  # dictionary mapping file paths to tag suggestions
            scan_error = Signal(str)  # error message
            
            def __init__(self, directory, db_session, config):
                super().__init__()
                self.directory = directory
                self.db_session = db_session
                self.config = config
                self.stop_requested = False
            
            def run(self):
                try:
                    # List all files in directory (not subdirectories)
                    files = []
                    with os.scandir(self.directory) as entries:
                        for entry in entries:
                            if entry.is_file() and not entry.name.startswith('.'):
                                files.append(entry.path)
                    
                    # Initialize progress
                    total_files = len(files)
                    if total_files == 0:
                        self.scan_finished.emit({})
                        return
                    
                    # Check which files are not in database or have no tags
                    # and generate suggestions for each file
                    from tag_suggestion import TagSuggester
                    tag_suggester = TagSuggester(self.config)
                    
                    untagged_files = {}
                    for idx, file_path in enumerate(files):
                        if self.stop_requested:
                            return
                        
                        file_obj = self.db_session.query(File).filter_by(path=file_path).first()
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
        progress = QProgressDialog("Scanning directory for untagged files...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Create and configure scan thread
        self.scan_thread = ScanThread(dir_path, self.db_session, self.config)
        
        # Connect signals
        self.scan_thread.scan_progress.connect(
            lambda current, total: progress.setValue(int(current / total * 100))
        )
        self.scan_thread.scan_finished.connect(lambda files: self.show_untagged_files_dialog(files, dir_path))
        self.scan_thread.scan_error.connect(
            lambda error: QMessageBox.critical(self, "Scan Error", f"Error scanning directory: {error}")
        )
        progress.canceled.connect(lambda: setattr(self.scan_thread, 'stop_requested', True))
        
        # Start the scan
        self.scan_thread.start()
    
    def show_untagged_files_dialog(self, untagged_files, directory_path):
        """Show dialog with untagged files and provide options to tag them."""
        if not untagged_files:
            QMessageBox.information(
                self, 
                "Scan Complete", 
                f"No untagged files found in directory:\n{directory_path}"
            )
            return
        
        # Create the dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Untagged Files")
        dialog.setMinimumWidth(900)
        dialog.setMinimumHeight(600)
        layout = QVBoxLayout(dialog)
        
        # Add header information
        header_label = QLabel(f"Found {len(untagged_files)} untagged files in:\n{directory_path}")
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
            lambda: [self.untagged_files_list.item(i).setCheckState(Qt.CheckState.Checked) 
                    for i in range(self.untagged_files_list.count())]
        )
        select_none_btn.clicked.connect(
            lambda: [self.untagged_files_list.item(i).setCheckState(Qt.CheckState.Unchecked) 
                    for i in range(self.untagged_files_list.count())]
        )
        
        main_layout.addLayout(file_layout)
        
        # Create a horizontal layout for the file-specific suggestions and general tag selection
        tags_container_layout = QHBoxLayout()
        
        # Tag suggestions section (left side)
        suggestions_layout = QVBoxLayout()
        suggestions_label = QLabel("File-Specific Tag Suggestions:")
        suggestions_layout.addWidget(suggestions_label)
        
        self.file_suggestions_list = QListWidget()
        self.file_suggestions_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        suggestions_layout.addWidget(self.file_suggestions_list)
        
        # Connect file list selection to update suggestions
        self.untagged_files_list.currentItemChanged.connect(self.update_file_specific_suggestions)
        
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
        apply_all_suggestions_btn = QPushButton("Apply All Suggestions to Selected Files")
        cancel_btn = QPushButton("Close")
        btn_layout.addWidget(apply_selected_btn)
        btn_layout.addWidget(apply_all_suggestions_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        # Connect buttons
        apply_selected_btn.clicked.connect(lambda: self.apply_tags_to_files(
            self.get_selected_files_from_list(self.untagged_files_list),
            [item.text() for item in tag_list.selectedItems()] +
            [self.get_tag_from_suggestion_item(item) for item in self.file_suggestions_list.selectedItems()]
        ))
        
        apply_all_suggestions_btn.clicked.connect(lambda: self.apply_all_suggestions_to_files(
            self.get_selected_files_from_list(self.untagged_files_list)
        ))
        
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
            sorted_suggestions = dict(sorted(
                suggestions.items(), 
                key=lambda item: item[1], 
                reverse=True
            ))
            
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
        
        for file_path in file_paths:
            if file_path in self.file_suggestions_map:
                suggestions = self.file_suggestions_map[file_path]
                
                if suggestions:
                    # Filter suggestions with good confidence (above 0.7)
                    good_suggestions = {tag: score for tag, score in suggestions.items() if score > 0.7}
                    
                    if good_suggestions:
                        # Apply these tags
                        self.apply_tags_to_file(file_path, good_suggestions.keys())
                        applied_count += 1
        
        # Show success message
        if applied_count > 0:
            QMessageBox.information(
                self,
                "Tags Applied",
                f"Applied high-confidence suggestions to {applied_count} of {total_files} selected files."
            )
        else:
            QMessageBox.information(
                self,
                "No Tags Applied",
                "No high-confidence suggestions were found for the selected files."
            )
            
    def apply_tags_to_file(self, file_path, tag_names):
        """Apply tags to a single file."""
        if not tag_names:
            return
            
        # Get or create file record
        file_obj = self.db_session.query(File).filter_by(path=file_path).first()
        is_new_file = False
        
        if not file_obj:
            file_obj = File(path=file_path)
            self.db_session.add(file_obj)
            is_new_file = True
        
        # Check if file already had tags before
        had_tags_before = len(file_obj.tags) > 0
        
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
        
        # Don't commit here - we commit in the calling function
        
        # If this is the first time the file has been tagged, add it to the vector database
        if is_new_file or not had_tags_before:
            try:
                from vector_search.content_extractor import ContentExtractor
                # Extract content from the file
                print(f"Attempting to extract content from: {file_path}")
                content = ContentExtractor.extract_file_content(file_path)
                
                if content:
                    print(f"Content extracted, length: {len(content)} characters")
                    # Add file to vector database
                    print(f"Sending to vector search index: {file_path}")
                    result = self.vector_search.index_file(file_path, content)
                    print(f"Added newly tagged file to vector search index: {file_path}")
                else:
                    print(f"Warning: No content could be extracted from file: {file_path}")
            except Exception as e:
                import traceback
                print(f"Error adding file to vector search: {str(e)}")
                print("Detailed error information:")
                traceback.print_exc()
        else:
            # If the file was already tagged before, just update the tags metadata
            try:
                self.vector_search.update_metadata(file_path)
                print(f"Updated tags metadata for file in vector search: {file_path}")
            except Exception as e:
                import traceback
                print(f"Error updating vector search metadata: {str(e)}")
                traceback.print_exc()

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
        self.db_session.commit()
        
        # Refresh tag lists
        self.refresh_tags()
        self.refresh_file_tags()
        
        # Show success message
        QMessageBox.information(
            self,
            "Tags Applied",
            f"Applied {len(tag_names)} tag(s) to {len(file_paths)} file(s)."
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
        """Force a file to be reindexed in the vector search database."""
        if not file_path or not os.path.isfile(file_path):
            QMessageBox.warning(self, "Error", f"The file {file_path} does not exist or is not a valid file.")
            return False
            
        try:
            # Get the file from database
            file_obj = self.db_session.query(File).filter_by(path=file_path).first()
            if not file_obj:
                QMessageBox.warning(self, "Error", f"The file {file_path} is not in the tag database.")
                return False
            
            from vector_search.content_extractor import ContentExtractor
            # Extract content from the file
            content = ContentExtractor.extract_file_content(file_path)
            if not content:
                QMessageBox.warning(
                    self, 
                    "Error", 
                    f"Could not extract content from {os.path.basename(file_path)}.\n\n"
                    "This file type may not be supported for content extraction."
                )
                return False
                
            # Index the file with existing content
            self.vector_search.index_file(file_path, content)
            
            QMessageBox.information(
                self,
                "Success",
                f"The file {os.path.basename(file_path)} has been successfully reindexed."
            )
            return True
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to reindex file: {str(e)}"
            )
            return False