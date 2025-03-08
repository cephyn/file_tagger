# File Tagger

A Windows file management application that allows users to add custom tags to files and search using boolean logic.

## Features

- Browse files across all drives
- Add, edit, and delete custom tags with colors
- Apply multiple tags to files
- Search files using AND/OR boolean logic
- Sort files by name, size, type, and date modified
- Navigate through directory structure with up/home buttons
- Double-click search results to locate files

## Installation

1. Ensure you have Python 3.8 or higher installed
2. Clone this repository
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

```
python -m src.file_tagger.main
```

## Project Structure

```
file_tagger/
├── src/
│   └── file_tagger/
│       ├── __init__.py
│       ├── main.py
│       └── models.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Development

The application uses:
- PySide6 for the GUI
- SQLAlchemy for tag database management
- SQLite for data storage

## License

MIT