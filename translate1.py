import requests
import json
import os
import logging
import time
from datetime import datetime
# No need to import scraping functions since scrap.py runs as cron job

# Bhashini translation service endpoint
BHASHINI_ENDPOINT = "http://192.168.50.71:8009/translate"

# File to track translated articles
TRANSLATED_FILE = "translated_articles.json"
LOG_FILE = "scrap.log"

# Translation services to use
TRANSLATION_SERVICES = ["google_translate", "azure_translator", "bhashini"]

# ---------------- LOGGING SETUP ----------------
# Use the same logger configuration as scrap.py
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# Bhashini supported Indian languages (all 22 official languages)
# indian_languages = [                     
#                       "as",    # Assamese                                          
#                       "bn",    # Bangla (Bengali)                         
#                       "brx",   # Bodo                         
#                       "doi",   # Dogri                                   
#                       "gu",    # Gujarati                                             
#                       "hi",    # Hindi                                   
#                       "kn",    # Kannada                                             
#                       "ks",    # Kashmiri                                                       
#                       "gom",   # Konkani                                                             
#                       "mai",   # Maithili                                                             
#                       "ml",    # Malayalam                                                             
#                       "mni",   # Manipuri                                                             
#                       "mr",    # Marathi                                                             
#                       "ne",    # Nepali                                                             
#                       "or",    # Odia                                                             
#                       "pa",    # Punjabi                                                             
#                       "sa",    # Sanskrit                                                             
#                       "sat",   # Santali                                                   
#                       "sd",    # Sindhi                                                   
#                       "ta",    # Tamil                                                   
#                       "te",    # Telugu                                         
#                       "ur"     # Urdu                               
#                     ]                     

indian_languages = [                     
                       "as",    # Assamese                                          
                       "bn",    # Bangla (Bengali)                         
                     #   "brx",   # Bodo                         
                     #   "doi",   # Dogri                                   
                       "gu",    # Gujarati                                             
                       "hi",    # Hindi                                   
                       "kn",    # Kannada                                             
                     #   "ks",    # Kashmiri                                                       
                     #   "gom",   # Konkani                                                             
                     #   "mai",   # Maithili                                                             
                       "ml",    # Malayalam                                                             
                     #   "mni",   # Manipuri                                                             
                       "mr",    # Marathi                                                             
                     #   "ne",    # Nepali                                                             
                       "or",    # Odia                                                             
                       "pa",    # Punjabi                                                             
                     #   "sa",    # Sanskrit                                                             
                     #   "sat",   # Santali                                                   
                     #   "sd",    # Sindhi                                                   
                       "ta",    # Tamil                                                   
                       "te",    # Telugu                                         
                       "ur"     # Urdu                               
                     ]                     

# Language code to name mapping (all Bhashini supported languages)
language_names = {
    "as": "Assamese",
    "bn": "Bangla",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "gom": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu"
}

def load_translated():
    """Load the set of already translated articles."""
    if os.path.exists(TRANSLATED_FILE):
        with open(TRANSLATED_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_translated(translated):
    """Save the set of translated articles."""
    with open(TRANSLATED_FILE, "w", encoding="utf-8") as f:
        json.dump(list(translated), f)

def translate_and_save(text, title="article", output_dir="translations"):
    """
    Translate text to all configured Indian languages using all 3 services and save to files.
    Translates one language at a time to avoid rate limiting.
    
    Args:
        text: The text to translate
        title: Title/prefix for the output files
        output_dir: Directory to save translations
    """
    logger.info(f"Starting translation for: {title}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Text length: {len(text)} characters")
    logger.info(f"Total languages to translate: {len(indian_languages)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    translations_saved = []
    
    
    # Translate one language at a time with retry logic
    for i, language_code in enumerate(indian_languages):
        language_name = language_names.get(language_code, language_code)
        retry_count = 0
        max_retries = 5  # Maximum retry attempts per language
        success = False
        
        while not success and retry_count < max_retries:
            try:
                if retry_count == 0:
                    logger.info(f"Translating to {language_name} ({i+1}/{len(indian_languages)})")
                else:
                    logger.info(f"Retrying {language_name} (attempt {retry_count + 1}/{max_retries})")
                
                # API request payload with all 3 services
                payload = {
                    "services": TRANSLATION_SERVICES,
                    "source_language": "en",
                    "target_language": language_code,
                    "text": text
                }

                headers = {
                    'accept': 'application/json',
                    'Content-Type': 'application/json'
                }

                response = requests.post(BHASHINI_ENDPOINT, headers=headers, json=payload)
                
                # Check for HTTP errors
                if response.status_code != 200:
                    logger.error(f"HTTP Error {response.status_code} for {language_name}: {response.text}")
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = min(10, 2 * retry_count)  # Progressive backoff: 2s, 4s, 6s, 8s, 10s
                        logger.info(f"Retrying {language_name} in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for {language_name}")
                        break
                
                result = response.json()
                
                # Check for API errors in response
                if 'error' in result:
                    logger.error(f"API Error for {language_name}: {result['error']}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2)
                        continue
                    break

                # Extract translated text from all services
                if 'results' in result and isinstance(result['results'], list) and len(result['results']) > 0:
                    # Create language directory
                    language_dir = os.path.join(output_dir, f"{title}_{language_name}")
                    os.makedirs(language_dir, exist_ok=True)
                    
                    translations_for_language = 0
                    
                    # Process each service result
                    for service_result in result['results']:
                        service_name = service_result.get('service_name', 'unknown')
                        
                        # Check if translation was successful
                        if service_result.get('success', False):
                            translated_text = service_result.get('translated_text')
                            if translated_text:
                                # Clean the text
                                clean_text = translated_text.replace('\\n', '\n').replace('\\r', '\r').replace('\\"', '"')
                                
                                # Basic validation for Telugu/Urdu
                                if language_code in ['te', 'ur'] and any('\u0900' <= char <= '\u097F' for char in clean_text[:100]):
                                    logger.warning(f"Detected Devanagari script in {language_name} translation from {service_name} - possible API error")
                                
                                # Save to file with service name
                                filename = f"{service_name}.txt"
                                filepath = os.path.join(language_dir, filename)
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    f.write(clean_text)
                                
                                translations_saved.append(filepath)
                                translations_for_language += 1
                                logger.info(f"Translation saved: {filepath}")
                            else:
                                logger.error(f"No translated text from {service_name} for {language_name}")
                        else:
                            error_msg = service_result.get('error_message', 'Unknown error')
                            logger.error(f"{service_name} translation failed for {language_name}: {error_msg}")
                    
                    # Consider it successful if at least one service provided translation
                    if translations_for_language > 0:
                        success = True
                        logger.info(f"Successfully translated {language_name} with {translations_for_language}/{len(TRANSLATION_SERVICES)} services")
                    else:
                        logger.error(f"All services failed for {language_name}")
                        retry_count += 1
                        if retry_count < max_retries:
                            time.sleep(2)
                            continue
                else:
                    logger.warning(f"Unexpected response format for {language_name}: {result}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2)
                        continue
                    break
                
                # Add delay between translations to be respectful to the API
                if success and i < len(indian_languages) - 1:  # Don't delay after the last translation
                    time.sleep(0.5)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error translating to {language_name}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(5)
                    continue
                break
            except Exception as e:
                logger.error(f"Unexpected error translating to {language_name}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)
                    continue
                break
        
        if not success:
            logger.error(f"Failed to translate to {language_name} after {retry_count} attempts")
    
    logger.info(f"Translation completed: {len(translations_saved)} total files saved")
    return translations_saved, None


# Function to translate scraped articles
def translate_scraped_articles():
    """
    Translate articles that have been scraped but not yet translated.
    This function is designed to work with scrap.py running as a cron job.
    """
    logger.info("=" * 60)
    logger.info("Starting translation check for scraped articles")
    
    # Load the set of already translated articles
    translated = load_translated()
    logger.info(f"Already translated articles: {len(translated)}")
    new_translations = 0
    
    # Read the latest articles from the articles directory
    articles_dir = "articles"
    if os.path.exists(articles_dir):
        # Get all subdirectories (date folders)
        for date_folder in sorted(os.listdir(articles_dir), reverse=True):
            date_path = os.path.join(articles_dir, date_folder)
            if os.path.isdir(date_path):
                # Process each article in the date folder
                for article_file in os.listdir(date_path):
                    if article_file.endswith('.txt'):
                        article_path = os.path.join(date_path, article_file)
                        
                        # Create unique identifier for this article
                        article_id = f"{date_folder}/{article_file}"
                        
                        # Skip if already translated
                        if article_id in translated:
                            continue
                        
                        # Check if article already has some translations
                        article_name = os.path.splitext(article_file)[0]
                        translation_dir = os.path.join("translations", date_folder)
                        
                        # Check if any language directories exist for this article
                        existing_translations = False
                        if os.path.exists(translation_dir):
                            for item in os.listdir(translation_dir):
                                if item.startswith(f"{article_name}_") and os.path.isdir(os.path.join(translation_dir, item)):
                                    # Check if all services have translations
                                    lang_dir = os.path.join(translation_dir, item)
                                    existing_services = [f for f in os.listdir(lang_dir) if f.endswith('.txt')]
                                    if len(existing_services) == len(TRANSLATION_SERVICES):
                                        existing_translations = True
                                        break
                        
                        if existing_translations:
                            logger.info(f"Article {article_file} already has complete translations, marking as done")
                            translated.add(article_id)
                            continue
                        
                        # Read article content
                        logger.info(f"Processing article: {article_file} from {date_folder}")
                        try:
                            with open(article_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Translate and save
                            translations_saved, _ = translate_and_save(content, article_name, translation_dir)
                            
                            if translations_saved:
                                # Mark as translated
                                translated.add(article_id)
                                new_translations += 1
                                logger.info(f"Successfully translated: {article_file}")
                            else:
                                logger.warning(f"No translations saved for: {article_file}")
                            
                        except Exception as e:
                            logger.error(f"Error processing {article_file}: {e}")
                
                # Continue processing all date folders
    
    # Save updated translated list
    if new_translations > 0:
        save_translated(translated)
        logger.info(f"Translation session completed: {new_translations} new articles translated")
    else:
        logger.info("No new articles to translate")
    
    logger.info("=" * 60)


# Main execution
if __name__ == "__main__":
    logger.info("Translation script started")
    
    # Translate scraped articles
    translate_scraped_articles()
    
    logger.info("Translation script finished")
