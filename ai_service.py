import os
from typing import List, Tuple, Dict, Optional
import mimetypes
import hashlib
import json
from datetime import datetime, timedelta
import importlib
from models import TagSuggestionCache

MAX_CONTENT_SIZE = 1024 * 1024  # 1MB max for content analysis
CACHE_DURATION = timedelta(days=7)  # Cache suggestions for 7 days

class AIService:
    def __init__(self, provider: str, api_key: str, db_session):
        """Initialize AI service with provider and API key."""
        self.provider = provider
        self.api_key = api_key
        self.db_session = db_session
        self.modules = {}  # Store imported modules
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
                self.model = self.modules['google'].GenerativeModel('gemini-2.0-flash-lite')
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except ImportError as e:
            module = str(e).split("'")[1]
            provider_map = {
                'openai': 'openai',
                'anthropic': 'anthropic',
                'google.generativeai': 'google-generativeai'
            }
            package = provider_map.get(module, module)
            raise ImportError(
                f"The {package} package is required for {self.provider}. "
                f"Please install it using: pip install {package}"
            )
            
    def _get_file_info(self, file_path: str) -> Tuple[str, str]:
        """Get file information and sample content for analysis."""
        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        size = os.path.getsize(file_path)
        modified = os.path.getmtime(file_path)
        
        # Calculate file hash for cache validation
        hasher = hashlib.sha256()
        content_sample = ""
        
        try:
            with open(file_path, 'rb') as f:
                # Read first chunk for hashing
                chunk = f.read(8192)
                hasher.update(chunk)
                
                # For text files, try to get content sample
                if mime_type and mime_type.startswith('text/'):
                    try:
                        f.seek(0)
                        content = f.read(MAX_CONTENT_SIZE).decode('utf-8')
                        content_sample = f"\nContent Sample:\n{content[:1000]}..."
                    except:
                        pass  # Ignore decoding errors
                
                # Continue hashing rest of file
                while chunk := f.read(8192):
                    hasher.update(chunk)
                    
        except Exception as e:
            print(f"Warning: Could not read file content: {e}")
        
        file_info = (
            f"Filename: {filename}\n"
            f"Type: {mime_type}\n"
            f"Size: {size} bytes\n"
            f"Last modified: {modified}"
            f"{content_sample}"
        )
        
        return file_info, hasher.hexdigest()
        
    def analyze_file(self, file_path: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Analyze a file and suggest tags with confidence scores.
        Returns ([(existing_tag, confidence)], [(new_tag, confidence)])
        """
        # Check cache first
        file_info, file_hash = self._get_file_info(file_path)
        cached = self._check_cache(file_path, file_hash)
        if cached:
            return self._parse_cached_suggestions(cached)
            
        prompt = (
            f"Analyze this file information and suggest appropriate tags:\n\n{file_info}\n\n"
            f"Existing tags in the system: {', '.join(existing_tags)}\n\n"
            "Provide your response in this format:\n"
            "EXISTING_TAGS: tag1 (confidence), tag2 (confidence), ...\n"
            "NEW_TAGS: tag1 (confidence), tag2 (confidence), ...\n"
            "EXPLANATION: brief explanation of why these tags were chosen\n\n"
            "Notes:\n"
            "- Confidence should be a number between 0 and 1\n"
            "- Only suggest tags with confidence > 0.3\n"
            "- Consider both the filename and content (if available)"
        )
        
        if self.provider == 'openai':
            result = self._analyze_openai(prompt, existing_tags)
        elif self.provider == 'anthropic':
            result = self._analyze_claude(prompt, existing_tags)
        elif self.provider == 'gemini':
            result = self._analyze_gemini(prompt, existing_tags)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
        # Cache the results
        self._cache_suggestions(file_path, file_hash, result)
        return result
            
    def _analyze_openai(self, prompt: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Analyze using OpenAI."""
        response = self.modules['openai'].ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return self._parse_response(response.choices[0].message.content, existing_tags)
        
    def _analyze_claude(self, prompt: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Analyze using Anthropic Claude."""
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_response(response.content[0].text, existing_tags)
        
    def _analyze_gemini(self, prompt: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Analyze using Google Gemini."""
        response = self.model.generate_content(prompt)
        return self._parse_response(response.text, existing_tags)
        
    def _parse_response(self, response: str, existing_tags: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Parse AI response into existing and new tags with confidence scores."""
        existing_matches = []
        new_suggestions = []
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('EXISTING_TAGS:'):
                current_section = 'existing'
                tags = line.replace('EXISTING_TAGS:', '').strip()
                existing_matches = self._parse_tags_with_confidence(tags)
            elif line.startswith('NEW_TAGS:'):
                current_section = 'new'
                tags = line.replace('NEW_TAGS:', '').strip()
                new_suggestions = self._parse_tags_with_confidence(tags)
                
        # Validate existing tags actually exist and have minimum confidence
        existing_matches = [
            (t, c) for t, c in existing_matches 
            if t in existing_tags and c > 0.3
        ]
        
        # Filter new suggestions by minimum confidence
        new_suggestions = [(t, c) for t, c in new_suggestions if c > 0.3]
        
        return existing_matches, new_suggestions
        
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
                         suggestions: Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]):
        """Cache tag suggestions for a file."""
        existing_tags, new_tags = suggestions
        cache_data = {
            'existing_tags': [(tag, float(conf)) for tag, conf in existing_tags],
            'new_tags': [(tag, float(conf)) for tag, conf in new_tags]
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
        
    def _parse_cached_suggestions(self, cached_data: Dict) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Convert cached suggestions back to the expected format."""
        existing_tags = [(t, float(c)) for t, c in cached_data['existing_tags']]
        new_tags = [(t, float(c)) for t, c in cached_data['new_tags']]
        return existing_tags, new_tags