import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTextEdit, QPushButton, QMessageBox, QApplication)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

class ChatWithResultsDialog(QDialog):
    """Dialog for chatting with top search results using AI."""
    def __init__(self, parent, ai_service, top_results, query):
        super().__init__(parent)
        self.ai_service = ai_service
        self.top_results = top_results[:3] if len(top_results) >= 3 else top_results
        self.initial_query = query
        self.chat_history = []
        
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
        # Prepare context from the top results
        context = "I'm going to ask questions about the following documents:\n\n"
        
        for i, result in enumerate(self.top_results):
            context += f"Document {i+1}: {os.path.basename(result['path'])}\n"
            context += f"Content snippets:\n"
            for snippet in result.get('snippets', []):
                context += f"- {snippet}\n"
            context += "\n"
        
        # Add the initial query as system message
        system_message = (
            f"You are an assistant helping with questions about documents. "
            f"Base your answers only on the document content snippets provided. "
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
        self.chat_display.append("<b>AI Assistant:</b> <i>Thinking...</i>")
        QApplication.processEvents()
        
        # Get AI response
        try:
            # Prepare prompt from chat history
            prompt = self.prepare_prompt()
            
            # Call the AI service
            response = self.get_ai_response(prompt)
            
            # Update the last "typing" message with the actual response
            current_html = self.chat_display.toHtml()
            current_html = current_html.replace("<b>AI Assistant:</b> <i>Thinking...</i>", f"<b>AI Assistant:</b> {response}")
            self.chat_display.setHtml(current_html)
            
            # Add to chat history
            self.chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            # Update the "typing" message with error
            current_html = self.chat_display.toHtml()
            current_html = current_html.replace("<b>AI Assistant:</b> <i>Thinking...</i>", 
                                             f"<b>AI Assistant:</b> Sorry, I encountered an error: {str(e)}")
            self.chat_display.setHtml(current_html)
    
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
        # Get the AI provider name from the AI service
        provider = self.ai_service.provider
        
        if provider == 'openai':
            return self.get_openai_response(prompt)
        elif provider == 'anthropic':
            return self.get_claude_response(prompt)
        elif provider == 'gemini':
            return self.get_gemini_response(prompt)
        elif provider == 'local':
            return self.get_local_response(prompt)
        else:
            return f"Error: Unsupported AI provider '{provider}'"
    
    def get_openai_response(self, prompt: str) -> str:
        """Get a response from OpenAI."""
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
    
    def get_claude_response(self, prompt: str) -> str:
        """Get a response from Anthropic Claude."""
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
    
    def get_gemini_response(self, prompt: str) -> str:
        """Get a response from Google Gemini."""
        messages = []
        for msg in self.chat_history:
            if msg["role"] != "system":
                role = "user" if msg["role"] == "user" else "model"
                messages.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        chat = self.ai_service.model.start_chat(
            history=[],
            system_instruction=self.chat_history[0]["content"] + "\n\n" + self.chat_history[1]["content"],
        )
        
        for msg in messages:
            if msg["role"] == "user":
                chat.send_message(msg["parts"][0]["text"])
        
        response = chat.send_message(messages[-1]["parts"][0]["text"] if messages[-1]["role"] == "user" else "")
        return response.text
    
    def get_local_response(self, prompt: str) -> str:
        """Get a response from the local model."""
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