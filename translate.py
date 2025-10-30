import requests
import time
import json
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------- CONFIG ----------------
API_URL = "http://192.168.50.71:8009/translate"
SOURCE_LANG = "en"
TARGET_LANG = "hi"
SERVICES = ["google_translate", "azure_translator", "bhashini"]

ARTICLE_FILE = "sample.txt"
MAX_RETRIES = 3
RETRY_DELAY = 2  # exponential backoff
CHUNK_SIZE = 1000  # approx chars per chunk for GPT evaluation

# ---------------- LOGGING ----------------
log_file = "translation.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)


# ---------------- UTILS ----------------
def load_article(file_path: str) -> str:
    logging.info(f"Loading article from {file_path}")
    path = Path(file_path)
    if not path.is_file():
        logging.error(f"Article file not found: {file_path}")
        raise FileNotFoundError(f"{file_path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, size: int = CHUNK_SIZE):
    """Split text into chunks for GPT evaluation."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    temp = ""
    for p in paragraphs:
        if len(temp) + len(p) + 2 <= size:
            temp += ("\n\n" if temp else "") + p
        else:
            if temp:
                chunks.append(temp)
            temp = p
    if temp:
        chunks.append(temp)
    return chunks


def translate_text(text: str) -> (dict, float):
    """Send text to translation API with retries."""
    payload = {
        "services": SERVICES,
        "source_language": SOURCE_LANG,
        "target_language": TARGET_LANG,
        "text": text,
    }
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"Translation request attempt {attempt}...")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            break
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                logging.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Translation failed.")
                raise

    elapsed = time.time() - start_time
    logging.info(f"Translation completed in {elapsed:.2f}s")

    translations = {}
    for item in data.get("results", []):
        service = item.get("service_name")
        translated = item.get("translated_text", "")
        translations[service] = translated

    return translations, elapsed


def evaluate_chunk(chunk: str, translations: dict) -> dict:
    """Evaluate a single chunk with GPT-5-nano."""
    llm = ChatOpenAI(model="gpt-5-nano", base_url="http://192.168.50.71:4000/v1")
    system_prompt = (
        "You are a bilingual translation evaluator. Compare multiple Hindi translations of the same English text. "
        "Evaluate them word-by-word for accuracy, fluency, naturalness, and preservation of meaning. "
        "Return the best translation and explain why."
    )
    user_prompt = f"""
English Text:
{chunk}

Translations:
{json.dumps(translations, ensure_ascii=False, indent=2)}

Evaluate and select the best translation. Return JSON:
{{
  "best_service": "<service_name>",
  "reasoning": "<detailed reasoning>"
}}
"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return json.loads(response.content)
        except Exception as e:
            logging.warning(f"GPT evaluation attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                logging.info(f"Retrying GPT evaluation in {delay}s...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached for chunk evaluation.")
                return {"best_service": "unknown", "reasoning": str(e)}


def save_translations(folder_path: Path, translations: dict):
    """Save all service translations individually."""
    for service, text in translations.items():
        file_name = f"{service}_translation.txt"
        with open(folder_path / file_name, "w", encoding="utf-8") as f:
            f.write(text)
    logging.info("All service translations saved.")


def save_results(folder_path: Path, best_service: str, translations: dict, evaluation: dict, times: dict):
    """Save best translation and metadata."""
    best_translation = translations.get(best_service, "")
    if best_translation:
        with open(folder_path / "best_translation.txt", "w", encoding="utf-8") as f:
            f.write(best_translation)
    metadata = {
        "best_service": best_service,
        "reasoning": evaluation.get("reasoning", ""),
        "timestamps": times,
        "services_compared": list(translations.keys()),
    }
    with open(folder_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logging.info("Best translation and metadata saved.")
    return 0.0  # <-- always return numeric value


# ---------------- MAIN ----------------
def main():
    total_start = time.time()
    try:
        article = load_article(ARTICLE_FILE)
        base_name = Path(ARTICLE_FILE).stem
        folder_path = Path(base_name)
        folder_path.mkdir(exist_ok=True)

        # 1️⃣ Translate full article
        translations, t_translate = translate_text(article)
        save_translations(folder_path, translations)

        # 2️⃣ Split article into chunks for evaluation
        chunks = split_into_chunks(article)
        logging.info(f"Article split into {len(chunks)} chunks for GPT evaluation.")

        # 3️⃣ Evaluate each chunk
        chunk_best_services = []
        t_eval_total = 0
        for i, chunk in enumerate(chunks, 1):
            logging.info(f"Evaluating chunk {i}/{len(chunks)}")
            start_chunk = time.time()
            eval_data = evaluate_chunk(chunk, translations)
            t_eval_total += time.time() - start_chunk
            chunk_best_services.append(eval_data.get("best_service", "unknown"))

        # 4️⃣ Aggregate best service (majority vote)
        best_service = max(set(chunk_best_services), key=chunk_best_services.count)
        evaluation = {
            "best_service": best_service,
            "reasoning": f"Selected by majority vote over {len(chunks)} chunks: {chunk_best_services}"
        }

        # 5️⃣ Save results
        t_save = save_results(folder_path, best_service, translations, evaluation,
                      {"translation_time": t_translate, "evaluation_time": t_eval_total}) or 0.0

        total_elapsed = time.time() - total_start
        logging.info("---- PROCESS SUMMARY ----")
        logging.info(f"Translation Time: {t_translate:.2f}s")
        logging.info(f"Evaluation Time:  {t_eval_total:.2f}s")
        logging.info(f"Saving Time:      {t_save:.2f}s")
        logging.info(f"Total Time:       {total_elapsed:.2f}s")
        logging.info(f"Best Service:     {best_service}")
        logging.info(f"Reasoning:        {evaluation['reasoning']}")

    except Exception as e:
        logging.error(f"Process terminated with error: {e}")


if __name__ == "__main__":
    main()

    logging.info("✅ Process completed successfully.")