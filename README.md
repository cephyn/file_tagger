# File Tagger

A Windows file management application that allows users to add custom tags to files and search using boolean logic. The application also supports integration with various AI providers and customizable workspace settings.

## Features

### File Management
- Browse files across all drives with proper sorting and column resizing
- Customizable home directory for quick access to your workspace
- Navigate through directory structure with up/home buttons
- Double-click search results to locate files
- Sort files by name, size, type, and date modified

### Tag Management
- Add, edit, and delete custom tags with colors
- Apply multiple tags to files
- Search files using AND/OR boolean logic
- Organize files with a flexible tagging system

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

### Supported AI Providers
- OpenAI (GPT-3.5)
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

Note: Tag suggestions are cached for 7 days to improve performance. Use the "Force Refresh" button to re-analyze a file.

### Password Management
- Change password: Settings > Password Management > Change Password
- Reset password: Use recovery key if you forget your password
- View/Reset recovery key: Settings > Password Management > Recovery

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

Main Dependencies:
- PySide6 - GUI framework
- SQLAlchemy - Database ORM
- cryptography - Secure storage of API keys
- openai - OpenAI API client
- google-generativeai - Google Gemini API client
- anthropic - Claude API client
- pypdf - PDF text extraction

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

## License

MIT