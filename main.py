import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTreeView, QListWidget, QPushButton, 
                           QInputDialog, QColorDialog, QLabel, QFileSystemModel,
                           QMessageBox, QLineEdit, QRadioButton, QButtonGroup,
                           QComboBox, QHeaderView, QMenuBar, QMenu, QDialog, QFileDialog,
                           QTabWidget, QProgressDialog)
from PySide6.QtCore import Qt, QDir, QStorageInfo
from PySide6.QtGui import QColor
from sqlalchemy import and_, or_
from models import init_db, File, Tag
from config import Config
from vector_search import VectorSearch
from api_settings import APISettingsDialog
from password_management import PasswordManagementDialog
from tag_suggestion import TagSuggestionDialog

class FileTagManager(QMainWindow):
    def __init__(self):
        super().__init__()
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
        layout = QHBoxLayout(central_widget)
        
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
        
        # Search section
        search_section = QVBoxLayout()
        
        # Create tab widget for different search types
        search_tabs = QTabWidget()
        
        # Tag search tab
        tag_search_tab = QWidget()
        tag_search_layout = QVBoxLayout(tag_search_tab)
        tag_search_label = QLabel("Search by Tags")
        tag_search_layout.addWidget(tag_search_label)
        
        # Move existing search results widget to tag search tab
        self.search_results = QListWidget()
        self.search_results.itemDoubleClicked.connect(self.on_search_result_double_clicked)
        tag_search_layout.addWidget(self.search_results)
        
        # Move existing search controls to tag search tab
        search_controls = QHBoxLayout()
        self.and_radio = QRadioButton("AND")
        self.or_radio = QRadioButton("OR")
        self.and_radio.setChecked(True)
        search_controls.addWidget(self.and_radio)
        search_controls.addWidget(self.or_radio)
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_by_tags)
        search_controls.addWidget(search_btn)
        tag_search_layout.addLayout(search_controls)
        
        # RAG search tab
        rag_search_tab = QWidget()
        rag_search_layout = QVBoxLayout(rag_search_tab)
        
        # Query input
        query_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your search query...")
        query_layout.addWidget(self.query_input)
        
        rag_search_btn = QPushButton("Search")
        rag_search_btn.clicked.connect(self.search_by_content)
        query_layout.addWidget(rag_search_btn)
        
        rag_search_layout.addLayout(query_layout)
        
        # Tag filter section
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by tags:"))
        self.rag_and_radio = QRadioButton("AND")
        self.rag_or_radio = QRadioButton("OR")
        self.rag_and_radio.setChecked(True)
        filter_layout.addWidget(self.rag_and_radio)
        filter_layout.addWidget(self.rag_or_radio)
        rag_search_layout.addLayout(filter_layout)
        
        # Tag filter list
        self.tag_filter_list = QListWidget()
        self.tag_filter_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        rag_search_layout.addWidget(self.tag_filter_list)
        
        # RAG search results
        self.rag_search_results = QListWidget()
        self.rag_search_results.itemDoubleClicked.connect(self.on_search_result_double_clicked)
        rag_search_layout.addWidget(self.rag_search_results)
        
        # Index management button
        reindex_btn = QPushButton("Reindex Files")
        reindex_btn.clicked.connect(self.reindex_files)
        rag_search_layout.addWidget(reindex_btn)
        
        # Add tabs
        search_tabs.addTab(tag_search_tab, "Tag Search")
        search_tabs.addTab(rag_search_tab, "Content Search")
        search_section.addWidget(search_tabs)
        
        # Add all sections to main layout
        layout.addLayout(explorer_layout, stretch=2)
        layout.addLayout(tag_layout, stretch=1)
        layout.addLayout(file_tags_layout, stretch=1)
        layout.addLayout(search_section, stretch=1)

        self.refresh_tags()
        self.current_file_path = None

    def refresh_tags(self):
        # Update both tag lists
        self.tag_list.clear()
        self.tag_filter_list.clear()
        tags = self.db_session.query(Tag).all()
        for tag in tags:
            # Add to main tag list
            item = self.tag_list.addItem(tag.name)
            self.tag_list.item(self.tag_list.count() - 1).setBackground(QColor(tag.color))
            # Add to filter list
            item = self.tag_filter_list.addItem(tag.name)
            self.tag_filter_list.item(self.tag_filter_list.count() - 1).setBackground(QColor(tag.color))

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
        
        # Display results
        self.rag_search_results.clear()
        
        if not results:
            self.rag_search_results.addItem("No matching files found")
            return
            
        for result in results:
            # Create result text with score and tags
            score_percent = int(result['score'] * 100)
            tags_text = f" [Tags: {', '.join(result.get('tags', []))}]"
            display_text = (
                f"{os.path.basename(result['path'])} "
                f"(Match: {score_percent}%) "
                f"{tags_text}"
            )
            
            item = self.rag_search_results.addItem(display_text)
            # Store full path in tooltip
            self.rag_search_results.item(self.rag_search_results.count() - 1).setToolTip(result['path'])
            
            # Color code based on match score
            color = self._get_score_color(result['score'])
            self.rag_search_results.item(self.rag_search_results.count() - 1).setBackground(color)
    
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
                item = self.file_tags_list.addItem(tag.name)
                self.file_tags_list.item(self.file_tags_list.count() - 1).setBackground(QColor(tag.color))

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
        selected_items = self.tag_list.selectedItems()
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
            # Get the drive path and select it in the combo box
            drive = os.path.splitdrive(file_path)[0] + os.path.sep
            for i in range(self.drive_combo.count()):
                if self.drive_combo.itemData(i).startswith(drive):
                    self.drive_combo.setCurrentIndex(i)
                    break
                    
            # Navigate to the directory containing the file
            dir_path = os.path.dirname(file_path)
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

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = FileTagManager()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()