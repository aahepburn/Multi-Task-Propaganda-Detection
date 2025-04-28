import pandas as pd
import openai
import time
import json

train_raw_text_df = pd.read_csv("train_raw_text.csv")
train_raw_text_df = train_raw_text_df[train_raw_text_df["Language"] != "EN"]

annotations_files = ["subtask1_RU.csv", "subtask1_PT.csv", "subtask1_HI.csv", "subtask1_BG.csv"]
annotations_df = pd.concat([pd.read_csv(f) for f in annotations_files], ignore_index=True)
annotations_df.rename(columns={"File": "Filename"}, inplace=True)

openai_client = openai.OpenAI(api_key="api-key")


### TRANSLATION UTILS


# call OpenAI API
def call_openai_api(prompt, role_desc, retries=3):
    for attempt in range(retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": role_desc},
                          {"role": "user", "content": prompt}],

                temperature=0.2
            )
            result = response.choices[0].message.content.strip()
            if result:
                return result
        except Exception as e:
            print(f"OpenAI API call failed (Attempt {attempt+1}): {e}")
            time.sleep(2)
    return None

# translate article
def translate_text(article):
    prompt = f"""
    Translate the following news article into English:

    ---
    ARTICLE:
    {article}
    ---

    Return only the translated text.
    """
    return call_openai_api(prompt, "You are a professional translator.")

# translate entities with position matching
def translate_entities(article_non_en, article_en, entities):
    prompt = f"""
    Given the original non-English article and its English translation, translate the following entities while maintaining their exact occurrence in the English text:

    ---
    Original Non-English Article:
    {article_non_en}

    Translated English Article:
    {article_en}

    Entities:
    {', '.join(entities)}
    ---

    Return a JSON object where each entity maps to its exact English translation found in the text.
    """
    response = call_openai_api(prompt, "You are a professional translator ensuring entity alignment.")

# debugging
    if response:
        try:
            print(f"Raw GPT-4o Response: {response}")
            cleaned_response = response.strip('```json').strip('```').strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            print(f"Error parsing JSON response. Raw response:\n{response}")
            return {}
    return {}



# TRANSLATION LOOP

translated_articles = []
translated_entities = []
start_time = time.time()

for _, row in train_raw_text_df.iterrows():
    filename = row["Filename"]
    article_non_en = row["Content"]

    entity_group = annotations_df[annotations_df["Filename"] == filename]
    if entity_group.empty:
        continue  # skip if no matching entities found

    entities_to_translate = entity_group["Entity"].tolist()
    entity_labels = entity_group[['Label1', 'Label2', 'Label3', 'Label4']].astype(str).agg(lambda x: ', '.join(x.dropna().unique()), axis=1).tolist()  # extract labels from annotation dataset

    try:
        article_en = translate_text(article_non_en)
        if not article_en:
            print(f"Skipping {filename} due to empty translation response.")
            continue

        entity_translations = translate_entities(article_non_en, article_en, entities_to_translate)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

    translated_articles.append({
        "Filename": filename,
        "Translated_Text": article_en
    })

    for (original_entity, translated_entity), label in zip(entity_translations.items(), entity_labels):
        match_start = article_en.find(translated_entity)
        match_end = match_start + len(translated_entity) if match_start != -1 else -1

        translated_entities.append({
            "Filename": filename,
            "Original_Entity": original_entity,
            "Translated_Entity": translated_entity,
            "Start": match_start,
            "End": match_end,
            "Label": label
        })

    print(f"Processed {filename} in {round(time.time() - start_time, 2)}s")



translated_articles_df = pd.DataFrame(translated_articles)
translated_articles_df.to_csv("translated_articles.csv", index=False)


translated_entities_df = pd.DataFrame(translated_entities)
translated_entities_df.to_csv("translated_entities.csv", index=False)

print(f"Processing completed in {round(time.time() - start_time, 2)}s. Output saved.")
