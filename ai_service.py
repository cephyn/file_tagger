import os
from typing import List, Tuple, Dict, Optional, Callable
import mimetypes
import hashlib
import json
from datetime import datetime, timedelta
import importlib
from pypdf import PdfReader
from models import TagSuggestionCache

MAX_AI_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB max for AI analysis
CACHE_DURATION = timedelta(days=7)  # Cache suggestions for 7 days

class AIService:
    def __init__(self, provider: str, api_key: str, db_session, progress_callback: Callable[[str, int], None] = None,
                 local_model_path: str = None, local_model_type: str = None, system_message: str = None):
        """Initialize AI service with provider and API key."""
        self.provider = provider
        self.api_key = api_key
        self.db_session = db_session
        self.modules = {}  # Store imported modules
        self.progress_callback = progress_callback
        self.local_model_path = local_model_path
        self.local_model_type = local_model_type
        self.system_message = system_message or "You are a helpful AI assistant for tagging and organizing files."
        self._setup_client()
        
    def _setup_client(self):
        """Set up the appropriate AI client based on provider."""
        try:
            if self.provider == 'openai':
                self.modules['openai'] = importlib.import_module('openai')
                self.modules['openai'].api_key = self.api_key
            elif self.provider == 'anthropic':
                self.modules['anthropic'] = importlib.import_module('anthropic')
                self.client = self.modules['anthropic'].Anthropic(api_key=self.api_key)
            elif self.provider == 'gemini':
                self.modules['google'] = importlib.import_module('google.generativeai')
                self.modules['google'].configure(api_key=self.api_key)
                # Include system message in the model initialization
                self.model = self.modules['google'].GenerativeModel(
'gemini-2.0-flash-lite',
                    system_instruction=self.system_message
)
            elif self.provider == 'local':
                # Check which local model type to use
                if self.local_model_type == 'llama':
                    self._setup_llama_cpp()
                elif self.local_model_type == 'ctransformers':
                    self._setup_ctransformers()
                else:
                    raise ValueError(f"Unsupported local model type: {self.local_model_type}")
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except ImportError as e:
            module = str(e).split("'")[1]
            provider_map = {
                'openai': 'openai',
                'anthropic': 'anthropic',
                'google.generativeai': 'google-generativeai',
                'llama_cpp': 'llama-cpp-python',
                'ctransformers': 'ctransformers'
            }
            package = provider_map.get(module, module)
            raise ImportError(
                f"The {package} package is required for {self.provider}. "
                f"Please install it using: pip install {package}"
            )
    
    def _setup_llama_cpp(self):
        """Set up the LlamaCpp runtime."""
        try:
            if not self.local_model_path:
                raise ValueError("No local model path specified")
                
            if self.progress_callback:
                self.progress_callback("Loading local model (LlamaCpp)...", 10)
                
            self.modules['llama_cpp'] = importlib.import_module('llama_cpp')
            
            # Configure model parameters
            self.model = self.modules['llama_cpp'].Llama(
                model_path=self.local_model_path,
                n_ctx=2048,  # Context size
                n_threads=os.cpu_count() or 4  # Use all available CPU cores
            )
            
            if self.progress_callback:
                self.progress_callback("Local model loaded successfully", 20)
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"Error loading local model: {str(e)}", 0)
            raise
    
    def _setup_ctransformers(self):
        """Set up the CTransformers runtime."""
        try:
            if not self.local_model_path:
                raise ValueError("No local model path specified")
                
            if self.progress_callback:
                self.progress_callback("Loading local model (CTransformers)...", 10)
                
            self.modules['ctransformers'] = importlib.import_module('ctransformers')
            
            # Detect model type from filename for common models
            model_filename = os.path.basename(self.local_model_path).lower()
            
            # Default to llama if we can't determine the model type
            model_type = "llama"
            
            # Try to detect model type from filename
            if "gemma" in model_filename:
                model_type = "gemma"
            elif "gpt2" in model_filename:
                model_type = "gpt2"
            elif "gpt-j" in model_filename or "gptj" in model_filename:
                model_type = "gptj"
            elif "llama" in model_filename:
                model_type = "llama"
                
            self.model = self.modules['ctransformers'].AutoModelForCausalLM.from_pretrained(
                self.local_model_path,
                model_type=model_type,
                threads=os.cpu_count() or 4
            )
            
            if self.progress_callback:
                self.progress_callback("Local model loaded successfully", 20)
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"Error loading local model: {str(e)}", 0)
            raise
            
    def _get_file_info(self, file_path: str) -> Tuple[str, str]:
        """Get file information and sample content for analysis."""
        if self.progress_callback:
            self.progress_callback("Starting file analysis...", 0)

        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        size = os.path.getsize(file_path)
        modified = os.path.getmtime(file_path)
        
        # Calculate file hash for cache validation
        hasher = hashlib.sha256()
        content_sample = ""
        
        try:
            with open(file_path, "rb") as f:
                # Read first chunk for hashing
                chunk = f.read(8192)
                hasher.update(chunk)
                
                # For text files, try to get content sample
                if mime_type and mime_type.startswith("text/"):
                    if self.progress_callback:
                        self.progress_callback("Extracting text content...", 30)
                    try:
                        f.seek(0)
                        content = f.read(MAX_AI_CONTENT_SIZE).decode("utf-8")
                        content_sample = f"\\nContent Sample:\\n{content[:1000]}..."
                    except:
                        pass  # Ignore decoding errors
                # For PDF files, extract text without size limit
                elif mime_type == "application/pdf":
                    try:
                        f.seek(0)
                        reader = PdfReader(f)
                        text = ""
                        total_pages = len(reader.pages)
                        
                        # Extract text from all pages with progress updates
                        for i, page in enumerate(reader.pages):
                            if self.progress_callback:
                                progress = 30 + int((i / total_pages) * 40)  # 30-70% progress
                                self.progress_callback(f"Extracting text from page {i+1}/{total_pages}...", progress)
                            text += page.extract_text()
                            
                        # Trim content for AI analysis if needed
                        if len(text.encode("utf-8")) > MAX_AI_CONTENT_SIZE:
                            if self.progress_callback:
                                self.progress_callback("Trimming content for AI analysis...", 75)
                            text = text.encode("utf-8")[:MAX_AI_CONTENT_SIZE].decode("utf-8", errors="ignore")
                            text += "...(truncated for AI analysis)"
                            
                        if text:
                            content_sample = f"\\nContent Sample:\\n{text}"
                    except Exception as e:
                        print(f"Warning: Could not extract PDF text: {e}")
                
                # Continue hashing rest of file
                while chunk := f.read(8192):
                    hasher.update(chunk)
                    
        except Exception as e:
            print(f"Warning: Could not read file content: {e}")
        
        file_info = (
            f"Filename: {filename}\\n"
            f"Type: {mime_type}\\n"
            f"Size: {size} bytes\\n"
            f"Last modified: {modified}"
            f"{content_sample}"
        )

        if self.progress_callback:
            self.progress_callback("File analysis complete", 80)
        
        return file_info, hasher.hexdigest()
        
    def analyze_file(self, file_path: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """
        Analyze a file and suggest tags with confidence scores.
        Returns ([(existing_tag, confidence)], [(new_tag, confidence)], explanation_text)
        """
        # Check cache first
        file_info, file_hash = self._get_file_info(file_path)
        cached = self._check_cache(file_path, file_hash)
        if cached:
            if self.progress_callback:
                self.progress_callback("Using cached results", 100)
            return self._parse_cached_suggestions(cached)
            
        if self.progress_callback:
            self.progress_callback("Preparing AI analysis...", 85)

        prompt = (
            f"Analyze this file information and suggest appropriate tags:\\n\\n{file_info}\\n\\n"
            f"Existing tags in the system: {', '.join(existing_tags)}\\n\\n"
            "Provide your response in this format:\\n"
            "EXISTING_TAGS: tag1 (confidence), tag2 (confidence), ...\\n"
            "NEW_TAGS: tag1 (confidence), tag2 (confidence), ...\\n"
            "EXPLANATION: brief explanation of why these tags were chosen\\n\\n"
            "Notes:\\n"
            "- Confidence should be a number between 0 and 1\\n"
            "- Only suggest tags with confidence > 0.3\\n"
            "- Consider both the filename and content (if available)"
        )
        
        if self.progress_callback:
            self.progress_callback("Sending to AI service...", 90)

        if self.provider == 'openai':
            result = self._analyze_openai(prompt, existing_tags)
        elif self.provider == 'anthropic':
            result = self._analyze_claude(prompt, existing_tags)
        elif self.provider == 'gemini':
            result = self._analyze_gemini(prompt, existing_tags)
        elif self.provider == 'local':
            result = self._analyze_local(prompt, existing_tags)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
        # Cache the results
        if self.progress_callback:
            self.progress_callback("Caching results...", 95)
        self._cache_suggestions(file_path, file_hash, result)

        if self.progress_callback:
            self.progress_callback("Analysis complete", 100)
        return result
            
    def _analyze_openai(self, prompt: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """Analyze using OpenAI."""
        response = self.modules['openai'].ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return self._parse_response(response.choices[0].message.content, existing_tags)
        
    def _analyze_claude(self, prompt: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """Analyze using Anthropic Claude."""
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system=self.system_message,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_response(response.content[0].text, existing_tags)
        
    def _analyze_gemini(self, prompt: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """Analyze using Google Gemini."""
        # System message is now passed during model initialization, so we only need to send the user prompt
        response = self.model.generate_content(prompt)
        return self._parse_response(response.text, existing_tags)
    
    def _analyze_local(self, prompt: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """Analyze using local model."""
        if self.local_model_type == 'llama':
            response = self._analyze_with_llama_cpp(prompt)
        elif self.local_model_type == 'ctransformers':
            response = self._analyze_with_ctransformers(prompt)
        else:
            raise ValueError(f"Unsupported local model type: {self.local_model_type}")
        
        return self._parse_response(response, existing_tags)
    
    def _analyze_with_llama_cpp(self, prompt: str) -> str:
        """Generate response using LlamaCPP."""
        if self.progress_callback:
            self.progress_callback("Generating with local LLM...", 85)
        
        # For local models, incorporate the system message into the prompt
        full_prompt = f"System: {self.system_message}\n\nUser: {prompt}"
            
        output = self.model(
            full_prompt,
            max_tokens=1000,
            temperature=0.7,
            stop=["</s>", "User:", "user:"]
        )
        return output['choices'][0]['text']
    
    def _analyze_with_ctransformers(self, prompt: str) -> str:
        """Generate response using CTransformers."""
        if self.progress_callback:
            self.progress_callback("Generating with local LLM...", 85)
        
        # For local models, incorporate the system message into the prompt
        full_prompt = f"System: {self.system_message}\n\nUser: {prompt}"
        
        response = self.model(
            full_prompt,
            max_new_tokens=1000,
            temperature=0.7,
            stop_sequences=["</s>", "User:", "user:"]
        )
        return response
        
    def _parse_response(self, response: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """Parse AI response into existing and new tags with confidence scores and explanation."""
        existing_matches = []
        new_suggestions = []
        explanation = ""
        
        lines = response.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('EXISTING_TAGS:'):
                current_section = 'existing'
                tags = line.replace('EXISTING_TAGS:', '').strip()
                existing_matches = self._parse_tags_with_confidence(tags)
            elif line.startswith('NEW_TAGS:'):
                current_section = 'new'
                tags = line.replace('NEW_TAGS:', '').strip()
                new_suggestions = self._parse_tags_with_confidence(tags)
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
                explanation = line.replace('EXPLANATION:', '').strip()
                
                # Collect multi-line explanation
                j = i + 1
                while j < len(lines) and not (lines[j].startswith('EXISTING_TAGS:') or 
                                            lines[j].startswith('NEW_TAGS:') or
                                            lines[j].strip() == ""):
                    explanation += " " + lines[j].strip()
                    j += 1
                
        # Validate existing tags actually exist and have minimum confidence
        existing_matches = [
            (t, c) for t, c in existing_matches 
            if t in existing_tags and c > 0.3
        ]
        
        # Filter new suggestions by minimum confidence
        new_suggestions = [(t, c) for t, c in new_suggestions if c > 0.3]
        
        return existing_matches, new_suggestions, explanation
        
    def _parse_tags_with_confidence(self, tags_str: str) -> List[Tuple[str, float]]:
        """Parse tags with confidence scores from string like 'tag1 (0.9), tag2 (0.8)'"""
        result = []
        if not tags_str:
            return result
            
        parts = [p.strip() for p in tags_str.split(',')]
        for part in parts:
            try:
                if '(' not in part or ')' not in part:
                    continue
                tag = part[:part.rfind('(')].strip()
                confidence = float(part[part.rfind('(')+1:part.rfind(')')])
                if tag and 0 <= confidence <= 1:
                    result.append((tag, confidence))
            except:
                continue
                
        return result
        
    def _check_cache(self, file_path: str, file_hash: str) -> Optional[Dict]:
        """Check if we have valid cached suggestions for this file."""
        cache_entry = self.db_session.query(TagSuggestionCache).filter_by(
            file_path=file_path,
            file_hash=file_hash,
            provider=self.provider
        ).first()
        
        if cache_entry and datetime.utcnow() - cache_entry.timestamp < CACHE_DURATION:
            return cache_entry.suggestions
            
        return None
        
    def _cache_suggestions(self, file_path: str, file_hash: str, 
                         suggestions: Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str]):
        """Cache tag suggestions for a file."""
        existing_tags, new_tags, explanation = suggestions
        cache_data = {
            'existing_tags': [(tag, float(conf)) for tag, conf in existing_tags],
            'new_tags': [(tag, float(conf)) for tag, conf in new_tags],
            'explanation': explanation
        }
        
        # Update or create cache entry
        cache_entry = self.db_session.query(TagSuggestionCache).filter_by(
            file_path=file_path
        ).first()
        
        if cache_entry:
            cache_entry.file_hash = file_hash
            cache_entry.suggestions = cache_data
            cache_entry.provider = self.provider
            cache_entry.timestamp = datetime.utcnow()
        else:
            cache_entry = TagSuggestionCache(
                file_path=file_path,
                file_hash=file_hash,
                suggestions=cache_data,
                provider=self.provider
            )
            self.db_session.add(cache_entry)
            
        self.db_session.commit()
        
    def _parse_cached_suggestions(self, cached_data: Dict) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """Convert cached suggestions back to the expected format."""
        existing_tags = [(t, float(c)) for t, c in cached_data['existing_tags']]
        new_tags = [(t, float(c)) for t, c in cached_data['new_tags']]
        explanation = cached_data.get('explanation', "No explanation available")
        return existing_tags, new_tags, explanation