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
- AI integration with support for:
  - OpenAI
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

On first run, you'll be prompted to set a password for encrypting your API settings. This password will be required each time you start the application. API keys can be configured through:

1. Click on "Settings" in the menu bar
2. Select "API Settings"
3. Choose your AI provider
4. Enter your API key (can toggle visibility)
5. Click Save

Your API keys are stored in an encrypted file and never committed to version control.

## Project Structure

```
file_tagger/
├── main.py              # Main application entry point
├── api_settings.py      # API configuration dialog
├── config.py           # Configuration and encryption utilities
├── models.py           # Database models
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Dependencies

- PySide6 - GUI framework
- SQLAlchemy - Database ORM
- cryptography - Secure storage of API keys

## Security

- API keys are encrypted using Fernet (symmetric encryption)
- Keys are encrypted using a password-derived key with PBKDF2
- Configuration files are automatically excluded from version control
- Password is required on application startup

## License

MIT