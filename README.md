# File Tagger

A Windows file management application that allows users to add custom tags to files and search using boolean logic. The application also supports integration with various AI providers and customizable workspace settings.

## Features

### File Management
- Browse files across all drives with proper sorting and column resizing
- Customizable home directory for quick access to your workspace
- Navigate through directory structure with up/home buttons
- Double-click search results to locate files
- Sort files by name, size, type, and date modified
- Quick navigation with drive selection dropdown
- Right-click context menu for file operations
- Open files directly from the application
- Open containing folder for quick access to file locations

### Tag Management
- Add, edit, and delete custom tags with colors
- Apply multiple tags to files
- Search files using AND/OR boolean logic
- Organize files with a flexible tagging system
- Batch tag multiple files at once
- Remove tags from files
- Auto-generate random colors for new tags
- Visual tag indicators with appropriate text contrast
- Create new tags directly from the tag suggestion interface

### Vector Search
- Semantic search through file contents
- Find files based on meaning, not just keywords
- Document chunking for improved search of large files
- Relevant text snippets in search results with highlighted matches
- Query expansion to find related terms
- Semantic ranking of search results with color-coded confidence scores
- Content-based file searching across multiple formats
- Tag filtering with semantic search (AND/OR logic)
- Interactive chat with search results
- Force reindex specific files when needed
- Remove files from search index
- Automatic metadata updates when tags change
- Document summaries for quick content overview
- Automatic indexing of newly tagged files
- Bulk reindexing operation for all files

### AI Integration
- AI-powered automatic tag suggestions:
  - Analyzes files using selected AI provider
  - Analyzes file content for text files (up to 10MB)
  - Full text extraction from PDF files
  - Shows confidence scores with visual indicators
  - Suggests matching existing tags
  - Proposes relevant new tags
  - Caches suggestions for 7 days
  - Interactive selection of suggested tags
  - Force refresh option to re-analyze files
- Chat with search results (RAG functionality)
- Batch processing of untagged files
- Automatic suggestions for newly discovered files

### Supported AI Providers
- OpenAI (GPT-3.5/4)
- Google Gemini
- Anthropic Claude (Claude-3 Sonnet)
- Local Models
  - LlamaCPP compatible models
  - CTransformers compatible models

### Security
- Password-protected configuration
- Secure encrypted storage of API keys
- Recovery key system for password reset
- Automatic file content cleanup after analysis

## Installation

1. Ensure you have Python 3.11 or higher installed
2. Clone this repository
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Optional: For local AI model support
   ```
   # For LlamaCPP models
   pip install llama-cpp-python
   
   # For CTransformers models
   pip install ctransformers
   ```

5. Optional: For vector search functionality
   ```
   # For semantic search capabilities
   pip install chromadb sentence-transformers
   ```

## Running the Application

```
python main.py
```

## Configuration

### Initial Setup
On first run, you'll be prompted to set a password for encrypting your settings. This password will be required each time you start the application.

### API Configuration
1. Click on "Settings" in the menu bar
2. Select "API Settings"
3. Choose your AI provider:
   - For cloud providers (OpenAI, Gemini, Claude): Enter your API key
   - For local models: Select model type (LlamaCPP/CTransformers) and choose model file
4. Optionally customize the AI system message
5. Click Save

### Home Directory Setup
1. Click on "Settings" in the menu bar
2. Select "Set Home Directory"
3. Choose your preferred starting directory
4. The selected directory will be your new starting location on next launch

### Using AI Tag Suggestions
1. Select a file in the file browser
2. Click "Suggest Tags (AI)" button
3. Wait for the AI to analyze the file
4. View suggested tags with confidence scores:
   - Progress bars show confidence level
   - Color gradient from red to green
   - Numeric score from 0 to 1
5. Select desired tags from both existing and new suggestions
6. Click "Apply Selected Tags"
7. View suggested explanations by clicking "Show Explanation"

Note: Tag suggestions are cached for 7 days to improve performance. Use the "Force Refresh" button to re-analyze a file.

### Using Vector Search
1. Navigate to the search tab
2. Enter your search query in the semantic search field
3. Optionally filter by tags using the tag selector
4. Click "Search" to find files based on their content
5. View results with relevant text snippets highlighting matches
6. Double-click results to open files

Note: Files are automatically indexed when first accessed. You can reindex all files from the settings menu.

### Batch Processing Untagged Files
1. Click "Scan Directory" in the menu bar
2. Wait for the scan to complete
3. View untagged files and their AI-suggested tags
4. Choose one of the options:
   - Select specific tags for specific files
   - Apply high-confidence suggestions to selected files
5. Click "Apply Selected Tags to Selected Files" or "Apply All Suggestions to Selected Files"

### Chat with Search Results
1. Perform a vector search
2. Select up to 3 files from the results (checkboxes)
3. Click "Chat with Results"
4. Ask questions about the selected documents
5. The AI will respond based on the content of the selected files

### Password Management
- Change password: Settings > Password Management > Change Password
- Reset password: Use recovery key if you forget your password
- View/Reset recovery key: Settings > Password Management > Recovery

## Project Structure

```
file_tagger/
├── main.py              # Main application entry point
├── api_settings.py      # API configuration dialog
├── config.py            # Configuration and encryption utilities
├── models.py            # Database models
├── ai_service.py        # AI provider integration
├── login.py             # Password verification interface
├── file_tag_manager.py  # Main application interface
├── tag_suggestion.py    # AI tag suggestion dialog
├── search.py            # RAG/Chat interface for search results
├── password_management.py # Security settings interface
├── vector_search/       # Semantic search functionality
│   ├── __init__.py
│   ├── content_extractor.py
│   ├── document_chunker.py
│   ├── search_utils.py
│   └── vector_search.py
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Dependencies

Main Dependencies:
- PySide6 - GUI framework
- SQLAlchemy - Database ORM
- cryptography - Secure storage of API keys
- openai - OpenAI API client
- google-generativeai - Google Gemini API client
- anthropic - Claude API client
- pypdf - PDF text extraction
- chromadb - Vector database for semantic search
- sentence-transformers - Text embedding models for semantic search

Optional Dependencies:
- llama-cpp-python - Local LlamaCPP model support
- ctransformers - Local CTransformers model support

## Security Features

- API keys are encrypted using Fernet (symmetric encryption)
- Keys are encrypted using a password-derived key with PBKDF2
- Configuration files are automatically excluded from version control
- Password is required on application startup
- File content is only analyzed temporarily, never stored
- Tag suggestions are cached securely in the local database
- Recovery key system for password reset

## Customization

### AI System Message
You can customize the system message used for AI interactions:
1. Go to Settings > API Settings
2. Edit the system message in the provided text area
3. Click Save to apply changes
4. Use "Reset to Default" to restore the original message

### Right-Click Actions
Right-clicking on search results provides useful options:
- Open File - Opens the selected file in its default application
- Open Containing Folder - Shows the file in File Explorer
- Force Reindex - Updates the vector search index for this file
- Remove from Search Index - Removes the file from vector search

## Version History

- 1.0.0 (April 2025) - Initial release with core functionality
- 0.0.1 (Development) - Current development version

## About

Developed by: Busy Wyvern
Website: [Busy Wyvern](https://www.busywyvern.com)
Built with Python, PySide6, SQLAlchemy, and Vector Search technologies
© 2025 All Rights Reserved

## License

MIT