import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTextEdit, QPushButton, QMessageBox, QApplication)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextCursor  # Added QTextCursor
from vector_search.content_extractor import ContentExtractor

class ChatWithResultsDialog(QDialog):
    """Dialog for chatting with top search results using AI."""
    def __init__(self, parent, ai_service, top_results, query):
        super().__init__(parent)
        if ai_service is None:
            raise ValueError("AI service not initialized. Please check your API settings and try again.")
            
        self.ai_service = ai_service
        self.top_results = top_results[:3] if len(top_results) >= 3 else top_results
        self.initial_query = query
        self.chat_history = []
        
        # Verify AI service is properly configured
        try:
            # Test if provider is set
            _ = self.ai_service.provider
            
            # For cloud providers, verify API key
            if self.ai_service.provider in ['openai', 'anthropic', 'gemini']:
                if not self.ai_service.api_key:
                    raise ValueError(f"No API key configured for {self.ai_service.provider}")
                    
            # For local provider, verify model path
            elif self.ai_service.provider == 'local':
                if not hasattr(self.ai_service, 'model'):
                    raise ValueError("Local AI model not properly initialized")
            else:
                raise ValueError(f"Unsupported AI provider: {self.ai_service.provider}")
                
        except Exception as e:
            raise ValueError(f"AI service configuration error: {str(e)}")
        
        self.setWindowTitle("Chat with Search Results")
        self.resize(800, 600)
        self.setup_ui()
        
        # Initialize chat with the search query
        self.start_chat()
    
    def setup_ui(self):
        """Set up the chat interface."""
        layout = QVBoxLayout()
        
        # Result files section
        results_layout = QHBoxLayout()
        results_layout.addWidget(QLabel("Chatting with:"))
        
        # Files being used
        for result in self.top_results:
            file_label = QLabel(os.path.basename(result['path']))
            file_label.setToolTip(result['path'])
            file_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
            results_layout.addWidget(file_label)
        
        layout.addLayout(results_layout)
        
        # Chat history area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.setMaximumHeight(80)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        
        self.setLayout(layout)
    
    def start_chat(self):
        """Initialize the chat with the AI."""
        # Prepare context from the top results with full document content
        context = "I'm going to ask questions about the following documents:\n\n"
        
        for i, result in enumerate(self.top_results):
            file_path = result['path']
            context += f"Document {i+1}: {os.path.basename(file_path)}\n"
            
            # Get full document content instead of just snippets
            try:
                full_content = ContentExtractor.extract_file_content(file_path)
                if full_content:
                    # Add the full content but include structural markers
                    context += f"Content:\n```\n{full_content}\n```\n\n"
                else:
                    # Fallback to snippets if full extraction fails
                    context += f"Content snippets (full extraction failed):\n"
                    for snippet in result.get('snippets', []):
                        context += f"- {snippet}\n"
                    context += "\n"
            except Exception as e:
                # Fallback to snippets if error
                context += f"Content snippets (extraction error: {str(e)}):\n"
                for snippet in result.get('snippets', []):
                    context += f"- {snippet}\n"
                context += "\n"
        
        # Add the initial query as system message
        system_message = (
            f"You are an assistant helping with questions about documents. "
            f"Base your answers only on the document content provided. "
            f"If the answer is not in the documents, say that you don't know based on the available information."
        )
        
        # Display a welcome message
        self.chat_display.append("<b>AI Assistant:</b> Hello! I can answer questions about the top search results. What would you like to know?")
        
        # Store context and system message in chat history
        self.chat_history = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": context}
        ]
        
        # Process the initial query if there is one
        if self.initial_query:
            self.chat_input.setPlainText(self.initial_query)
            self.send_message()
    
    def send_message(self):
        """Send the user message and get AI response."""
        user_message = self.chat_input.toPlainText().strip()
        if not user_message:
            return
        
        # Display user message
        self.chat_display.append(f"<b>You:</b> {user_message}")
        
        # Add to chat history
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Clear input
        self.chat_input.clear()
        
        # Show typing indicator
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)  # Using QTextCursor enum
        self.chat_display.setTextCursor(cursor)
        self.chat_display.append("<b>AI Assistant:</b> <i>Thinking...</i>")
        QApplication.processEvents()
        
        # Get AI response
        try:
            # Debug info
            print(f"Using AI provider: {self.ai_service.provider}")
            
            # Prepare prompt from chat history
            prompt = self.prepare_prompt()
            
            # Call the AI service
            response = self.get_ai_response(prompt)
            
            # Remove the "Thinking..." message and add the actual response
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)  # Using QTextCursor enum
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)  # Using QTextCursor enums
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # Remove the newline
            self.chat_display.append(f"<b>AI Assistant:</b> {response}")
            
            # Add to chat history
            self.chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            # Remove the "Thinking..." message and show the error
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)  # Using QTextCursor enum
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)  # Using QTextCursor enums
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # Remove the newline
            error_message = f"Error: {str(e)}\nType: {type(e).__name__}"
            print(f"AI Response Error: {error_message}")  # Debug print
            self.chat_display.append(f"<b>AI Assistant:</b> Sorry, I encountered an error: {error_message}")
    
    def prepare_prompt(self) -> str:
        """Prepare the prompt from chat history."""
        prompt = ""
        
        # Add system and context messages first
        for message in self.chat_history:
            if message["role"] == "system":
                prompt += f"{message['content']}\n\n"
        
        # Add conversation history (excluding system messages)
        for message in self.chat_history:
            if message["role"] != "system":
                role_name = "User" if message["role"] == "user" else "Assistant"
                prompt += f"{role_name}: {message['content']}\n\n"
        
        # Add final prompt for the assistant to respond
        prompt += "Assistant: "
        
        return prompt
    
    def get_ai_response(self, prompt: str) -> str:
        """Get a response from the AI service."""
        try:
            # Get the AI provider name from the AI service
            provider = self.ai_service.provider
            print(f"Provider: {provider}")  # Debug print
            
            if provider == 'openai':
                return self.get_openai_response(prompt)
            elif provider == 'anthropic':
                return self.get_claude_response(prompt)
            elif provider == 'gemini':
                return self.get_gemini_response(prompt)
            elif provider == 'local':
                if not hasattr(self.ai_service, 'model'):
                    raise ValueError("Local model not properly initialized")
                return self.get_local_response(prompt)
            else:
                return f"Error: Unsupported AI provider '{provider}'"
        except AttributeError as e:
            print(f"AI Service Error: Missing attribute - {str(e)}")
            raise ValueError(f"AI service not properly configured: {str(e)}")
        except Exception as e:
            print(f"AI Response Error: {type(e).__name__} - {str(e)}")
            raise
    
    def get_openai_response(self, prompt: str) -> str:
        """Get a response from OpenAI."""
        try:
            print("Getting OpenAI response...")  # Debug print
            if 'openai' not in self.ai_service.modules:
                raise ValueError("OpenAI module not properly initialized")
                
            response = self.ai_service.modules['openai'].ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.chat_history[0]["content"]},
                    {"role": "system", "content": self.chat_history[1]["content"]},
                    *[{"role": msg["role"], "content": msg["content"]} 
                      for msg in self.chat_history[2:] if msg["role"] != "system"]
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {type(e).__name__} - {str(e)}")
            raise
    
    def get_claude_response(self, prompt: str) -> str:
        """Get a response from Anthropic Claude."""
        try:
            print("Getting Claude response...")  # Debug print
            if not hasattr(self.ai_service, 'client'):
                raise ValueError("Claude client not properly initialized")
                
            messages = []
            for msg in self.chat_history:
                if msg["role"] != "system":
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            response = self.ai_service.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=self.chat_history[0]["content"] + "\n\n" + self.chat_history[1]["content"],
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude error: {type(e).__name__} - {str(e)}")
            raise
    
    def get_gemini_response(self, prompt: str) -> str:
        """Get a response from Google Gemini."""
        try:
            print("Getting Gemini response...")  # Debug print
            if not hasattr(self.ai_service, 'model'):
                raise ValueError("Gemini model not properly initialized")
                
            # Convert chat history to Gemini format
            messages = []
            
            # Start with system messages
            system_content = self.chat_history[0]["content"] + "\n\n" + self.chat_history[1]["content"]
            messages.append({"role": "user", "parts": [{"text": f"System instructions: {system_content}"}]})
            messages.append({"role": "model", "parts": [{"text": "I understand and will follow these instructions."}]})
            
            # Add conversation history
            for msg in self.chat_history[2:]:  # Skip system messages
                role = "user" if msg["role"] == "user" else "model"
                messages.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            # Create a new chat
            chat = self.ai_service.model.start_chat(history=messages)
            
            # Send the last user message
            response = chat.send_message(messages[-1]["parts"][0]["text"] if messages[-1]["role"] == "user" else "")
            return response.text
        except Exception as e:
            print(f"Gemini error: {type(e).__name__} - {str(e)}")
            raise
    
    def get_local_response(self, prompt: str) -> str:
        """Get a response from the local model."""
        try:
            print("Getting local model response...")  # Debug print
            print(f"Local model type: {self.ai_service.local_model_type}")  # Debug print
            
            if not hasattr(self.ai_service, 'model'):
                raise ValueError("Local model not properly initialized")

            if self.ai_service.local_model_type == 'llama':
                # For LlamaCpp
                full_prompt = f"System: {self.chat_history[0]['content']}\n\n{self.chat_history[1]['content']}\n\n" + prompt
                output = self.ai_service.model(
                    full_prompt,
                    max_tokens=1000,
                    temperature=0.7,
                    stop=["</s>", "User:", "user:"]
                )
                return output['choices'][0]['text']
            else:
                # For CTransformers
                full_prompt = f"System: {self.chat_history[0]['content']}\n\n{self.chat_history[1]['content']}\n\n" + prompt
                response = self.ai_service.model(
                    full_prompt,
                    max_new_tokens=1000,
                    temperature=0.7,
                    stop_sequences=["</s>", "User:", "user:"]
                )
                return response
        except Exception as e:
            print(f"Local model error: {type(e).__name__} - {str(e)}")
            raise