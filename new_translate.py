import requests
import time
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# --------------------------------------------
# CONFIG
# --------------------------------------------
API_URL = "http://192.168.50.71:8009/translate"
SERVICES = ["google_translate", "azure_translator", "bhashini"]
SOURCE_LANG = "en"
TARGET_LANG = "hi"

# 0️⃣ Load article
with open("article.txt", "r", encoding="utf-8") as f:
    article = f.read()


def translate_text(text):
    payload = {
        "services": SERVICES,
        "source_language": SOURCE_LANG,
        "target_language": TARGET_LANG,
        "text": text,
    }

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=payload)
    elapsed = time.time() - start_time

    response.raise_for_status()
    data = response.json()

    translations = {}
    for item in data.get("results", []):
        service = item.get("service_name")
        translated = item.get("translated_text", "")
        translations[service] = translated

    return translations, elapsed


def evaluate_translations(article, translations):
    start_time = time.time()

    llm = ChatOpenAI(model="gpt-5-nano", base_url="http://192.168.50.71:4000/v1")

    system_prompt = (
        "You are a linguistic evaluator. Compare multiple Hindi translations of the same English article. "
        "Evaluate them word-by-word for accuracy, fluency, naturalness, and preservation of meaning. "
        "Return the best translation and explain why."
    )

    user_prompt = f"""
Original English Article:
{article}

Translations to compare:
{json.dumps(translations, ensure_ascii=False, indent=2)}

Evaluate each translation word-by-word and select the best one.
Return JSON strictly in this format:
{{
  "best_service": "<service_name>",
  "reasoning": "<detailed reasoning>"
}}
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    elapsed = time.time() - start_time

    try:
        parsed = json.loads(response.content)
    except Exception:
        parsed = {"best_service": "unknown", "reasoning": response.content}

    return parsed, elapsed


def save_results(best_service, reasoning, translations, times):
    start_time = time.time()

    best_translation = translations.get(best_service, "")

    # Save best translation
    with open("best_translation.txt", "w", encoding="utf-8") as f:
        f.write(best_translation)

    # Save metadata
    metadata = {
        "best_service": best_service,
        "reasoning": reasoning,
        "timestamps": times,
        "services_compared": list(translations.keys()),
    }
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    return elapsed


if __name__ == "__main__":
    t0 = time.time()
    translations, t_translate = translate_text(article)
    print(f"✅ Translations received in {t_translate:.2f}s")

    evaluation, t_eval = evaluate_translations(article, translations)
    print(f"🤖 Evaluation completed in {t_eval:.2f}s")

    if evaluation["best_service"] in translations:
        t_save = save_results(
            evaluation["best_service"],
            evaluation["reasoning"],
            translations,
            {
                "translation_time": t_translate,
                "evaluation_time": t_eval,
            },
        )
        print(f"💾 Results saved in {t_save:.2f}s")

        print("\n🧾 TIME REPORT")
        print(f"Translation Time: {t_translate:.2f}s")
        print(f"Evaluation Time:  {t_eval:.2f}s")
        print(f"Saving Time:      {t_save:.2f}s")
        print(f"Total Time:       {time.time() - t0:.2f}s\n")

        print(f"✅ Best Service: {evaluation['best_service']}")
        print(f"📄 Reasoning: {evaluation['reasoning']}")
    else:
        print("⚠️ No valid best translation found to save.")
