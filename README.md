# File Tagger

A Windows file management application that allows users to add custom tags to files and search using boolean logic. The application also supports integration with various AI providers.

## Features

- Browse files across all drives with proper sorting and column resizing
- Add, edit, and delete custom tags with colors
- Apply multiple tags to files
- Search files using AND/OR boolean logic
- Sort files by name, size, type, and date modified
- Navigate through directory structure with up/home buttons
- Double-click search results to locate files
- AI-powered automatic tag suggestions:
  - Analyzes files using selected AI provider
  - Analyzes file content for text files (up to 1MB)
  - Shows confidence scores with visual indicators
  - Suggests matching existing tags
  - Proposes relevant new tags
  - Caches suggestions for 7 days
  - Interactive selection of suggested tags
  - Force refresh option to re-analyze files
- AI provider integration:
  - OpenAI (GPT-3.5)
  - Google Gemini
  - Anthropic Claude
- Secure encrypted storage of API keys
- Password-protected configuration

## Installation

1. Ensure you have Python 3.8 or higher installed
2. Clone this repository
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

```
python main.py
```

## Configuration

On first run, you'll be prompted to set a password for encrypting your API settings. This password will be required each time you start the application.

### API Setup
1. Click on "Settings" in the menu bar
2. Select "API Settings"
3. Choose your AI provider
4. Enter your API key (can toggle visibility)
5. Click Save

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

Note: Tag suggestions are cached for 7 days to improve performance. Use the "Force Refresh" button to re-analyze a file.

Your API keys are stored in an encrypted file and never committed to version control.

## Project Structure

```
file_tagger/
├── main.py              # Main application entry point
├── api_settings.py      # API configuration dialog
├── config.py           # Configuration and encryption utilities
├── models.py           # Database models
├── ai_service.py       # AI provider integration
├── tag_suggestion.py   # AI tag suggestion dialog
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Dependencies

- PySide6 - GUI framework
- SQLAlchemy - Database ORM
- cryptography - Secure storage of API keys
- openai - OpenAI API client
- google-generativeai - Google Gemini API client
- anthropic - Claude API client

## Security

- API keys are encrypted using Fernet (symmetric encryption)
- Keys are encrypted using a password-derived key with PBKDF2
- Configuration files are automatically excluded from version control
- Password is required on application startup
- File content is only analyzed temporarily, never stored
- Tag suggestions are cached securely in the local database

## License

MIT