#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
LOG_FILE = "review_agent.log"
REVIEWED_FILE = "reviewed_translations.json"

# OpenAI configuration
OPENAI_MODEL = "gpt-5-nano"  # Using gpt-4o model
OPENAI_API_KEY="6kCxwQz8DyEwdVXDi4qL_5RRhNSI_JZG8VHhErGaiNBrelOC5ZksjBUcLAHT3BlbkFJ63Z3EaoiznwMxSQaLKzG2hST3oWSslRjt3pDYFA5UfG4ny2hWnn8km82LXefs9m6JBN4uof9IA"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Performance settings
MAX_WORKERS = 100  # Process up to 100 translations in parallel
BATCH_SIZE = 50   # Process 50 GPT evaluations concurrently

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Google Drive configuration
GOOGLE_DRIVE_CREDENTIALS_PATH = os.environ.get('GOOGLE_DRIVE_CREDENTIALS_PATH', 'credentials.json')
GOOGLE_DRIVE_ROOT_FOLDER_ID = os.environ.get('GOOGLE_DRIVE_ROOT_FOLDER_ID', None)
GOOGLE_DRIVE_SHARED_DRIVE_ID = os.environ.get('GOOGLE_DRIVE_SHARED_DRIVE_ID', None)
GOOGLE_DRIVE_TRANSLATIONS_FOLDER_ID = os.environ.get('GOOGLE_DRIVE_TRANSLATIONS_FOLDER_ID', None)

# Extract ID from URL if full URL is provided
if GOOGLE_DRIVE_SHARED_DRIVE_ID and 'drive.google.com' in str(GOOGLE_DRIVE_SHARED_DRIVE_ID):
    GOOGLE_DRIVE_SHARED_DRIVE_ID = GOOGLE_DRIVE_SHARED_DRIVE_ID.split('/')[-1]
    logger.info(f"Extracted shared drive ID: {GOOGLE_DRIVE_SHARED_DRIVE_ID}")

class GoogleDriveUploader:
    """Helper class to handle Google Drive uploads."""
    
    def __init__(self, credentials_path: str):
        self.service = None
        self.folder_cache = {}  # Cache folder IDs for this session only
        self.cache_lock = threading.Lock()  # Thread safety for cache
        self.shared_drive_id = GOOGLE_DRIVE_SHARED_DRIVE_ID
        self.translations_folder_id = GOOGLE_DRIVE_TRANSLATIONS_FOLDER_ID
        
        if os.path.exists(credentials_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                self.service = build('drive', 'v3', credentials=credentials)
                logger.info("Google Drive service initialized successfully")
                
                # Setup folder structure
                self._setup_folders()
            except Exception as e:
                logger.error(f"Failed to initialize Google Drive service: {e}")
        else:
            logger.warning(f"Google Drive credentials file not found: {credentials_path}")
    
    def _setup_folders(self):
        """Setup folder structure using fixed folder ID."""
        try:
            if self.translations_folder_id:
                # Use the fixed translations folder ID
                with self.cache_lock:
                    self.folder_cache["translations"] = self.translations_folder_id
                logger.info(f"Using fixed translations folder ID: {self.translations_folder_id}")
            else:
                # Create translations folder if no fixed ID provided
                translations_id = self.create_folder_if_not_exists("translations")
                if translations_id:
                    with self.cache_lock:
                        self.folder_cache["translations"] = translations_id
                    logger.info(f"Created new translations folder: {translations_id}")
        except Exception as e:
            logger.error(f"Error setting up folders: {e}")
    
    def create_folder_if_not_exists(self, folder_name: str, parent_id: str = None) -> Optional[str]:
        """Create a folder in Google Drive if it doesn't exist."""
        if not self.service:
            return None
        
        try:
            # First check if folder already exists
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            else:
                query += f" and '{self.shared_drive_id}' in parents"
            
            # Support for shared drives
            list_params = {"q": query, "fields": "files(id, name)"}
            if self.shared_drive_id:
                list_params.update({
                    "supportsAllDrives": True,
                    "includeItemsFromAllDrives": True,
                    "driveId": self.shared_drive_id,
                    "corpora": "drive"
                })
            
            results = self.service.files().list(**list_params).execute()
            items = results.get('files', [])
            
            if items:
                folder_id = items[0]['id']
                logger.info(f"Found existing folder: {folder_name} (ID: {folder_id})")
                return folder_id
            
            # Create new folder if it doesn't exist
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                file_metadata['parents'] = [parent_id]
            elif self.shared_drive_id:
                # If creating in shared drive root
                file_metadata['parents'] = [self.shared_drive_id]
            
            create_params = {"body": file_metadata, "fields": "id"}
            if self.shared_drive_id:
                create_params["supportsAllDrives"] = True
            
            folder = self.service.files().create(**create_params).execute()
            folder_id = folder.get('id')
            
            logger.info(f"Created new Google Drive folder: {folder_name} (ID: {folder_id})")
            return folder_id
            
        except HttpError as e:
            logger.error(f"Error creating folder {folder_name}: {e}")
            return None
    
    def find_or_create_folder(self, folder_name: str, parent_id: str) -> Optional[str]:
        """Find an existing folder or create it if it doesn't exist."""
        if not self.service:
            return None
        
        try:
            # First check if folder already exists
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents"
            
            # Support for shared drives
            list_params = {"q": query, "fields": "files(id, name)"}
            if self.shared_drive_id:
                list_params.update({
                    "supportsAllDrives": True,
                    "includeItemsFromAllDrives": True
                })
            
            results = self.service.files().list(**list_params).execute()
            items = results.get('files', [])
            
            if items:
                folder_id = items[0]['id']
                logger.info(f"Found existing folder: {folder_name} (ID: {folder_id})")
                return folder_id
            
            # Create new folder if it doesn't exist
            logger.info(f"Folder {folder_name} not found, creating new one")
            return self.create_folder_if_not_exists(folder_name, parent_id)
            
        except HttpError as e:
            logger.error(f"Error finding/creating folder {folder_name}: {e}")
            return None
    
    def upload_file(self, local_path: str, drive_path: str, parent_folder_id: str = None) -> Optional[bool]:
        """Upload a file to Google Drive.

        Returns True if a new upload occurs, None if the file already exists, and False on error.
        """
        if not self.service or not os.path.exists(local_path):
            return False
        
        # Set timeout for the upload
        import socket
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)  # 30 second timeout
            
        try:
            # Parse the drive path to create folder structure
            path_parts = drive_path.split('/')
            filename = path_parts[-1]
            folder_parts = path_parts[:-1]
            
            # Create folder hierarchy
            current_parent_id = parent_folder_id or GOOGLE_DRIVE_ROOT_FOLDER_ID or self.shared_drive_id
            
            for i, folder_name in enumerate(folder_parts):
                if folder_name:
                    # Use fixed translations folder ID if this is the translations folder
                    if i == 0 and folder_name == "translations" and self.translations_folder_id:
                        current_parent_id = self.translations_folder_id
                        logger.debug(f"Using fixed translations folder ID: {self.translations_folder_id}")
                    else:
                        # Check cache first
                        cache_key = f"{current_parent_id}/{folder_name}"
                        with self.cache_lock:
                            if cache_key in self.folder_cache:
                                current_parent_id = self.folder_cache[cache_key]
                                logger.debug(f"Using cached folder ID for {folder_name}")
                                continue
                        
                        # For date folders (like 24_Oct_2025), check if it already exists
                        folder_id = self.find_or_create_folder(folder_name, current_parent_id)
                        if folder_id:
                            current_parent_id = folder_id
                            # Cache the folder ID
                            with self.cache_lock:
                                self.folder_cache[cache_key] = folder_id
                        else:
                            return False
            
            # Check if file already exists
            query = f"name='{filename}'"
            if current_parent_id:
                query += f" and '{current_parent_id}' in parents"
            
            list_params = {"q": query, "fields": "files(id)"}
            if GOOGLE_DRIVE_SHARED_DRIVE_ID:
                list_params.update({
                    "supportsAllDrives": True,
                    "includeItemsFromAllDrives": True
                })
            
            results = self.service.files().list(**list_params).execute()
            existing_files = results.get('files', [])
            
            if existing_files:
                logger.info(f"Skipping upload; file already exists on Google Drive: {drive_path}")
                return None
            else:
                # Upload new file
                media = MediaFileUpload(local_path, resumable=True)
                file_metadata = {'name': filename}
                
                # Create new file
                if current_parent_id:
                    file_metadata['parents'] = [current_parent_id]
                
                create_params = {
                    "body": file_metadata,
                    "media_body": media,
                    "fields": "id"
                }
                if GOOGLE_DRIVE_SHARED_DRIVE_ID:
                    create_params["supportsAllDrives"] = True
                
                self.service.files().create(**create_params).execute()
                logger.info(f"Uploaded file to Google Drive: {drive_path}")
            
            return True
            
        except HttpError as e:
            logger.error(f"Error uploading file {local_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading file {local_path}: {e}")
            return False
        finally:
            # Restore original timeout
            socket.setdefaulttimeout(original_timeout)

class TranslationReviewAgent:
    def __init__(self):
        self.reviewed = self.load_reviewed()
        self.reviewed_lock = threading.Lock()
        self.llm = ChatOpenAI(
            base_url = "https://api.openai.com/v1",
            api_key = "sk-",
            model=OPENAI_MODEL,
            max_retries=MAX_RETRIES
        )
        # Initialize Google Drive uploader (but won't use it during processing)
        self.gdrive_uploader = None
        
        # Track files to upload later
        self.files_to_upload = []
        self.upload_lock = threading.Lock()
        
    def load_reviewed(self) -> set:
        """Load set of already reviewed translations."""
        if os.path.exists(REVIEWED_FILE):
            with open(REVIEWED_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()
    
    def save_reviewed(self):
        """Save set of reviewed translations."""
        with self.reviewed_lock:
            with open(REVIEWED_FILE, "w", encoding="utf-8") as f:
                json.dump(list(self.reviewed), f, ensure_ascii=False, indent=2)
    
    def extract_language_info(self, folder_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract language code and name from folder name."""
        language_map = {
            "Hindi": ("hi", "Hindi"),
            "Bengali": ("bn", "Bengali"),
            "Bangla": ("bn", "Bangla"),
            "Telugu": ("te", "Telugu"),
            "Marathi": ("mr", "Marathi"),
            "Tamil": ("ta", "Tamil"),
            "Gujarati": ("gu", "Gujarati"),
            "Urdu": ("ur", "Urdu"),
            "Kannada": ("kn", "Kannada"),
            "Odia": ("or", "Odia"),
            "Malayalam": ("ml", "Malayalam"),
            "Punjabi": ("pa", "Punjabi"),
            "Assamese": ("as", "Assamese"),
            "Maithili": ("mai", "Maithili"),
            "Sanskrit": ("sa", "Sanskrit"),
            "Konkani": ("gom", "Konkani"),
            "Sindhi": ("sd", "Sindhi"),
            "Kashmiri": ("ks", "Kashmiri"),
            "Nepali": ("ne", "Nepali"),
            "Manipuri": ("mni", "Manipuri"),
            "Bodo": ("brx", "Bodo"),
            "Dogri": ("doi", "Dogri"),
            "Santali": ("sat", "Santali")
        }
        
        parts = folder_name.split('_')
        if parts:
            lang_name = parts[-1]
            if lang_name in language_map:
                return language_map[lang_name]
        return None, None
    
    def evaluate_translations_with_gpt(self, translations: Dict[str, str], 
                                     target_language: str, 
                                     original_text: str = None) -> Tuple[str, str, Dict]:
        """Use GPT to evaluate and select the best translation."""
        
        prompt = f"""You are an expert translator and linguist specializing in Indian languages. 
        Please evaluate the following {len(translations)} translations of the same text into {target_language} and select the best one.

        Consider these criteria:
        1. Linguistic accuracy and naturalness in {target_language}
        2. MAKE SURE all the acronyms of english in the PARTICULAR {target_language}
        3. Cultural appropriateness
        4. Grammar and syntax correctness
        5. Fluency and readability
        6. Proper use of {target_language} script and characters
        7. Compare each sentence of one translation to the another
        8. Preservation of meaning from the original 

        """
        
        if original_text:
            truncated_original = original_text[:1000] + "..." if len(original_text) > 1000 else original_text
            prompt += f"Original English text (truncated):\n{truncated_original}\n\n"
        
        prompt += "Translations to evaluate:\n\n"
        
        for i, (service, text) in enumerate(translations.items(), 1):
            truncated = text[:500] + "..." if len(text) > 500 else text
            prompt += f"Translation {i} (from {service}):\n{truncated}\n\n"
        
        prompt += """Please analyze each translation and respond with a JSON object containing:
            {
                "best_service": "name of the service that provided the best translation",
                "reasoning": "detailed explanation of why this translation is the best",
                "scores": {
                    "service_name": {
                        "score": 0-100,
                        "strengths": ["list of strengths"],
                        "weaknesses": ["list of weaknesses"]
                    }
                },
                "confidence": 0-100
            }"""
        
        # Make API call with retries
        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    SystemMessage(content="You are an expert translator evaluating translation quality. Always respond with valid JSON."),
                    HumanMessage(content=prompt)
                ]
                
                response = self.llm.invoke(messages)
                response_content = response.content
                
                # Clean up response content to handle common JSON issues
                # Remove any potential markdown code blocks
                if "```json" in response_content:
                    response_content = response_content.split("```json")[1].split("```")[0]
                elif "```" in response_content:
                    response_content = response_content.split("```")[1].split("```")[0]
                
                # Parse JSON with better error handling
                try:
                    result = json.loads(response_content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    logger.error(f"Response content: {response_content[:500]}")
                    raise
                
                # Validate response structure
                if "best_service" in result:
                    # Handle variations in service names from GPT
                    best_service = result["best_service"]
                    
                    # Try to find the actual service name in translations
                    actual_service = None
                    for service in translations:
                        if service.lower() in best_service.lower():
                            actual_service = service
                            break
                    
                    # If no match found, check for specific patterns
                    if not actual_service:
                        if "translation 1" in best_service.lower() or "bhashini" in best_service.lower():
                            if "bhashini" in translations:
                                actual_service = "bhashini"
                        elif "translation 2" in best_service.lower() or "google" in best_service.lower():
                            if "google_translate" in translations:
                                actual_service = "google_translate"
                        elif "translation 3" in best_service.lower() or "azure" in best_service.lower():
                            if "azure_translator" in translations:
                                actual_service = "azure_translator"
                    
                    if actual_service and actual_service in translations:
                        result["best_service"] = actual_service
                        return (
                            actual_service,
                            translations[actual_service],
                            result
                        )
                    else:
                        logger.error(f"Could not map GPT response '{best_service}' to available services: {list(translations.keys())}")
                else:
                    logger.error(f"Invalid GPT response structure: {result}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                        
            except Exception as e:
                logger.error(f"GPT API error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
        
        # Fallback to first available translation if GPT fails
        logger.warning("GPT evaluation failed, using fallback selection")
        first_service = list(translations.keys())[0]
        return first_service, translations[first_service], {"fallback": True}
    
    def refine_translation_with_gpt(self, text: str, target_language: str) -> str:
        
        """Ask GPT to refine the selected translation to PIB/GoI official style."""
        prompt = f"""
        You are an expert linguist and professional translator for the Government of India.
        Your task is to refine the following {target_language} translation so that it strictly follows the official Press Information Bureau (PIB) house style.

        Guidelines:
        - Tone: Formal, factual, neutral, and diplomatic.
        - Grammar: Perfect grammar, punctuation, and syntax.
        - Vocabulary: Prefer standard official terms, avoid English words unless commonly used.
        - Transliteration: Avoid transliteration where a proper native term exists.
        - Script: Use the correct {target_language} script, without non-standard marks or mixed-language text.
        - Style: Should read like an official government press release -- concise, professional, clear.
        - Accuracy: Do not add, remove, or change meaning -- only improve expression and presentation.
        - Output must reflect formal tone training from PIB reference material.
        - Simulate post-processing grammar and syntax validation (Indic NLP Library standards).
        - Uphold a glossary of official terms and deterministic transliteration rules.
        - Replace literal English idioms with established government terminology.
        - Return only the refined translation text without any explanations or formatting.

        Mandatory corrections (apply wherever relevant):
        - PIB house style.)
        - Compound form per official Hindi grammar.)
        - Prefer simpler Devanagari without nukta.)
        - Maintain formal PIB tone.
        - Use official translated name.

        Here is the translation to refine:
        {text}
        """
        
        try:
            messages = [
                SystemMessage(content="You are a professional government translator."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error refining translation: {e}")
            return text  # fallback to original if GPT fails

    
    def process_translation_folder(self, folder_path: str, original_text_path: str = None) -> bool:
        """Process a single translation folder and save the best translation."""
        folder_name = os.path.basename(folder_path)
        folder_id = folder_path.replace(os.sep, '/')
        
        # Check if already reviewed by looking for metadata file
        metadata_path = os.path.join(folder_path, 'translation_metadata.json')
        if os.path.exists(metadata_path):
            logger.info(f"Already reviewed: {folder_name} - adding to upload queue")
            
            # Look for the final translation file (should be named after the folder)
            output_filename = f"{folder_name}.txt"
            output_path = os.path.join(folder_path, output_filename)
            
            if os.path.exists(output_path):
                # Add to upload queue
                path_parts = folder_path.split(os.sep)
                if len(path_parts) >= 3:  # translations/date/article_language
                    gdrive_path = os.path.join(path_parts[0], path_parts[1], output_filename)
                else:
                    gdrive_path = os.path.relpath(output_path)
                
                with self.upload_lock:
                    self.files_to_upload.append((output_path, gdrive_path))
                logger.debug(f"Added to upload queue: {gdrive_path}")
            
            with self.reviewed_lock:
                self.reviewed.add(folder_id)
            return False  # Return False since we didn't do new processing
        
        # Skip if already reviewed in current session
        with self.reviewed_lock:
            if folder_id in self.reviewed:
                logger.debug(f"Already reviewed in session: {folder_name}")
                return False
        
        logger.info(f"Processing: {folder_name}")
        
        # Extract language information
        language_code, language_name = self.extract_language_info(folder_name)
        if not language_name:
            logger.warning(f"Could not extract language from: {folder_name}")
            language_name = "Unknown"
        
        # Get original text if available
        original_text = None
        if original_text_path and os.path.exists(original_text_path):
            try:
                with open(original_text_path, 'r', encoding='utf-8') as f:
                    original_text = f.read()
            except Exception as e:
                logger.error(f"Error reading original text: {e}")
        
        # Read all translations
        translations = {}
        folder_name = os.path.basename(folder_path)
        output_filename = f"{folder_name}.txt"
        
        for file in os.listdir(folder_path):
            if file.endswith('.txt') and file != 'translation.txt' and file != output_filename:
                service_name = file.replace('.txt', '')
                file_path = os.path.join(folder_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            translations[service_name] = content
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
        
        if not translations:
            logger.warning(f"No translations found in {folder_name}")
            return False
        
        # Use GPT to select best translation
        best_service, best_text, evaluation_result = self.evaluate_translations_with_gpt(
            translations, language_name, original_text
        )
        
        # Step 2: Refine the chosen translation to match PIB/GoI standards
        logger.info(f"Refining best translation ({best_service}) for {language_name}")
        refined_text = self.refine_translation_with_gpt(best_text, language_name)
        if refined_text and len(refined_text) > 10:
            best_text = refined_text
            logger.info(f"Refinement completed for {folder_name}")
        else:
            logger.warning(f"Refinement failed or returned empty, keeping original selection")

        
        if best_text:
            # Save the best translation with folder name as filename
            folder_name = os.path.basename(folder_path)
            output_filename = f"{folder_name}.txt"
            output_path = os.path.join(folder_path, output_filename)
            try:
                output_changed = True
                if os.path.exists(output_path):
                    try:
                        with open(output_path, 'r', encoding='utf-8') as existing_file:
                            existing_content = existing_file.read()
                        output_changed = existing_content != best_text
                    except Exception as read_err:
                        logger.warning(f"Could not read existing output for change detection ({output_path}): {read_err}")
                        output_changed = True
                
                if output_changed:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(best_text)
                    logger.info(f"Saved updated translation file: {output_path}")
                else:
                    logger.info(f"No content change detected for {output_path}; existing file retained")
                
                # Save metadata
                metadata = {
                    'selected_service': best_service,
                    'language_code': language_code,
                    'language_name': language_name,
                    'evaluation_model': OPENAI_MODEL,
                    'evaluation_result': evaluation_result,
                    'review_timestamp': datetime.now().isoformat(),
                    'available_services': list(translations.keys()),
                    'original_text_available': original_text is not None,
                    'refined_to_pib_standard': True,

                }
                
                metadata_path = os.path.join(folder_path, 'translation_metadata.json')
                metadata_changed = True
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as existing_meta:
                            existing_metadata = json.load(existing_meta)
                        metadata_changed = existing_metadata != metadata
                    except Exception as read_meta_err:
                        logger.warning(f"Could not read existing metadata for change detection ({metadata_path}): {read_meta_err}")
                        metadata_changed = True
                
                if metadata_changed:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved updated metadata: {metadata_path}")
                else:
                    logger.info(f"No metadata change detected for {metadata_path}; existing file retained")
                
                # Handle different types of evaluation_result
                if isinstance(evaluation_result, dict):
                    confidence = evaluation_result.get('confidence', 'N/A')
                    reasoning = evaluation_result.get('reasoning', '')
                else:
                    confidence = 'N/A'
                    reasoning = ''
                
                logger.info(f"✓ Selected {best_service} for {language_name} (confidence: {confidence})")
                if reasoning:
                    logger.info(f"  Reasoning: {reasoning[:200]}...")
                
                # Add to upload queue (instead of uploading now)
                path_parts = folder_path.split(os.sep)
                if len(path_parts) >= 3:  # translations/date/article_language
                    gdrive_path = os.path.join(path_parts[0], path_parts[1], output_filename)
                else:
                    gdrive_path = os.path.relpath(output_path)
                
                with self.upload_lock:
                    self.files_to_upload.append((output_path, gdrive_path))
                
                # Mark as reviewed
                with self.reviewed_lock:
                    self.reviewed.add(folder_id)
                
                # Save reviewed set periodically
                if len(self.reviewed) % 10 == 0:
                    self.save_reviewed()
                
                return True
                
            except Exception as e:
                logger.error(f"Error saving translation: {e}")
                return False
        else:
            logger.error(f"No valid translation found for {folder_name}")
            return False
    
    def process_translations_parallel(self, translation_folders: List[Tuple[str, str]]):
        """Process multiple translations in parallel."""
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_folder = {}
            
            for folder_path, original_path in translation_folders:
                future = executor.submit(self.process_translation_folder, folder_path, original_path)
                future_to_folder[future] = folder_path
            
            completed = 0
            total = len(translation_folders)
            
            for future in as_completed(future_to_folder):
                folder_path = future_to_folder[future]
                try:
                    result = future.result()
                    completed += 1
                    if result:
                        logger.info(f"Progress: {completed}/{total} completed")
                except Exception as e:
                    logger.error(f"Error processing {folder_path}: {e}")
                    completed += 1
    
    def upload_all_to_drive(self):
        """Upload all processed files to Google Drive in batch."""
        if not self.files_to_upload:
            logger.info("No files to upload to Google Drive")
            return
        
        logger.info("="*60)
        total_requests = len(self.files_to_upload)
        unique_targets = {path for _, path in self.files_to_upload}
        logger.info(f"Starting Google Drive uploads for {total_requests} files ({len(unique_targets)} unique paths)")
        logger.info("="*60)
        
        # Initialize Google Drive uploader
        self.gdrive_uploader = GoogleDriveUploader(GOOGLE_DRIVE_CREDENTIALS_PATH)
        
        if not self.gdrive_uploader or not self.gdrive_uploader.service:
            logger.error("Google Drive service not available. Skipping uploads.")
            return
        
        uploaded = 0
        skipped_duplicates = 0
        already_exists = 0
        failed = 0
        seen_paths = set()
        
        # Upload files one by one to avoid SSL errors
        for local_path, gdrive_path in self.files_to_upload:
            if gdrive_path in seen_paths:
                skipped_duplicates += 1
                logger.info(f"Skipping duplicate upload request: {gdrive_path}")
                continue
            seen_paths.add(gdrive_path)
            try:
                logger.info(f"Uploading: {gdrive_path}")
                result = self.gdrive_uploader.upload_file(local_path, gdrive_path)
                if result is True:
                    uploaded += 1
                    logger.info(f"✓ Uploaded ({uploaded}/{len(unique_targets)})")
                elif result is None:
                    already_exists += 1
                    logger.info(f"• Already on Drive, skipped ({already_exists} total already present)")
                else:
                    failed += 1
                    logger.warning(f"✗ Failed to upload: {gdrive_path}")
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                failed += 1
                logger.error(f"Error uploading {gdrive_path}: {e}")
        
        logger.info("="*60)
        logger.info(f"Google Drive upload completed")
        logger.info(f"Successfully uploaded: {uploaded} files")
        logger.info(f"Already existed (skipped): {already_exists} files")
        logger.info(f"Duplicate requests skipped: {skipped_duplicates} items")
        logger.info(f"Failed uploads: {failed} files")
        logger.info("="*60)
    
    def review_all_translations(self):
        """Review all translations in the translations directory."""
        translations_dir = "translations"
        if not os.path.exists(translations_dir):
            logger.error("Translations directory not found")
            return
        
        # Collect all translation folders to process
        all_translations = []
        skipped_count = 0
        
        # Process each date folder
        for date_folder in sorted(os.listdir(translations_dir)):
            date_path = os.path.join(translations_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
            
            logger.info(f"\nCollecting translations from: {date_folder}")
            
            # Find original articles for this date
            articles_date_path = os.path.join("articles", date_folder)
            
            # Process each article's language folders
            article_groups = {}
            for item in os.listdir(date_path):
                item_path = os.path.join(date_path, item)
                if os.path.isdir(item_path):
                    # Extract article name (everything before the last underscore)
                    parts = item.rsplit('_', 1)
                    if len(parts) == 2:
                        article_base = parts[0]
                        language = parts[1]
                        if article_base not in article_groups:
                            article_groups[article_base] = []
                        article_groups[article_base].append((language, item_path))
            
            # Collect translations for parallel processing
            for article_base, language_folders in article_groups.items():
                # Find original article
                original_path = None
                if os.path.exists(articles_date_path):
                    for article_file in os.listdir(articles_date_path):
                        if article_file.startswith(article_base) and article_file.endswith('.txt'):
                            original_path = os.path.join(articles_date_path, article_file)
                            break
                
                # Add each language folder to processing queue
                for language, folder_path in language_folders:
                    all_translations.append((folder_path, original_path))
                    # Count if already reviewed
                    metadata_path = os.path.join(folder_path, 'translation_metadata.json')
                    if os.path.exists(metadata_path):
                        skipped_count += 1
        
        # Process all translations in parallel
        logger.info("="*60)
        logger.info(f"Total translation folders found: {len(all_translations)}")
        logger.info(f"Already reviewed (will upload only): {skipped_count}")
        logger.info(f"Need review + upload: {len(all_translations) - skipped_count}")
        logger.info(f"Using {MAX_WORKERS} workers")
        logger.info("="*60)
        
        if len(all_translations) == 0:
            logger.info("No translations found!")
            return
        
        start_time = time.time()
        self.process_translations_parallel(all_translations)
        end_time = time.time()
        
        # Save final reviewed set
        self.save_reviewed()
        
        # Calculate statistics
        elapsed_time = end_time - start_time
        translations_per_minute = len(all_translations) / (elapsed_time / 60) if elapsed_time > 0 else 0
        
        logger.info("="*60)
        logger.info(f"Processing completed in {elapsed_time:.1f} seconds")
        logger.info(f"Total folders processed: {len(all_translations)}")
        logger.info(f"Already reviewed (upload only): {skipped_count}")
        logger.info(f"Newly reviewed: {len(all_translations) - skipped_count}")
        logger.info(f"Processing rate: {translations_per_minute:.1f} folders/minute")
        logger.info("="*60)
        
        # Now upload everything to Google Drive
        self.upload_all_to_drive()


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Batch Translation Review Agent (GPT-powered) started")
    logger.info(f"Using model: {OPENAI_MODEL}")
    logger.info(f"Max workers: {MAX_WORKERS}")
    logger.info("Mode: Process all first, then upload to Drive")
    
    # Check for API key
    if not os.environ.get('OPENAI_API_KEY') and not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set!")
        return
    
    agent = TranslationReviewAgent()
    agent.review_all_translations()
    
    logger.info("Batch Translation Review Agent completed")
    

if __name__ == "__main__":
    main()
