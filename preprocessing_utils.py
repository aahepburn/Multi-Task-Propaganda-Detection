import pandas as pd
import re

# ============
# DOMAIN INFERENCE
# ============

def infer_from_filename(fname):
    if re.search(r"UA|RU|URW", str(fname)):
        return "UA"
    elif re.search(r"CC", str(fname)):
        return "CC"
    return "-"

def infer_from_narrative(narrative):
    if pd.isna(narrative):
        return "-"
    if re.search(r"\bURW\b|\bURW:", narrative, re.IGNORECASE):
        return "UA"
    elif re.search(r"\bCC\b|\bCC:", narrative, re.IGNORECASE):
        return "CC"
    return "-"

def infer_from_text(text):
    if pd.isna(text):
        return "-"
    if re.search(r"\b(russia|ukraine|nato|donbas|invasion|putin|zelensky|military|conflict|war|soldier|kyiv|kremlin|zakharova)\b", text, re.IGNORECASE):
        return "UA"
    elif re.search(r"\b(climate|carbon|emission|warming|renewable|greenhouse|sustainability|pollution|energy|temperature|co2|weather|fossil|deforestation|ganga)\b", text, re.IGNORECASE):
        return "CC"
    return "-"

def infer_domain(row):
    domain = infer_from_filename(row["Filename"])
    if domain != "-":
        return domain
    domain = infer_from_narrative(row.get("Narrative", ""))
    if domain != "-":
        return domain
    return infer_from_text(row.get("Translated_Text", ""))

# ============
# LABEL CLEANING
# ============

def prefix_other(label_str, domain):
    if pd.isna(label_str):
        return label_str
    domain = str(domain).strip().upper()
    prefix = "URW" if domain == "UA" else "CC" if domain == "CC" else "-"
    parts = [s.strip() for s in str(label_str).split(";")]
    return ";".join([
        f"{prefix}: Other" if part == "Other" else part
        for part in parts
    ])

def deduplicate_label_string(label_str):
    if pd.isna(label_str):
        return label_str
    seen = set()
    result = []
    for part in [s.strip() for s in label_str.split(";")]:
        if part not in seen:
            seen.add(part)
            result.append(part)
    return ";".join(result)

def fix_labels_preserve_structure(df, domain_map):
    df["Domain"] = df["Filename"].map(domain_map).fillna("-")

    df["Narrative"] = [
        deduplicate_label_string(prefix_other(narr, dom))
        for narr, dom in zip(df["Narrative"], df["Domain"])
    ]

    if "Subnarrative" in df.columns:
        df["Subnarrative"] = [
            deduplicate_label_string(prefix_other(subnarr, dom))
            for subnarr, dom in zip(df["Subnarrative"], df["Domain"])
        ]

    return df.drop(columns=["Domain"])

# ============
# PROCESSING
# ============

# ARTICLE FILES – fix domain
article_paths = [
    "train-all-articles.csv",
    "test-S2-articles.csv",
    "train-S1-articles.csv",
    "test-S1-articles.csv"
]

for path in article_paths:
    print(f"Processing articles: {path}")
    df = pd.read_csv(path)
    labels_path = path.replace("articles", "labels") if "S2" in path else None

    # Merge label narrative only if needed for domain inference
    if labels_path and "S2" in path:
        try:
            df_labels = pd.read_csv(labels_path, usecols=["Filename", "Narrative"])
            if "Narrative" in df.columns:
                df = df.drop(columns=["Narrative"])
            df = pd.merge(df, df_labels, on="Filename", how="left")
        except FileNotFoundError:
            print(f" Labels file not found for {path}, skipping merge.")

    # Apply domain inference
    df["Domain"] = df.apply(infer_domain, axis=1)

    # Drop narrative after inference so it's not saved in article files
    if "Narrative" in df.columns:
        df = df.drop(columns=["Narrative"])

    df.to_csv(path, index=False)
    print(f" Domain updated in: {path}")

# LABEL FILES – fix prefixes and deduplicate
label_article_map = {
    "train-S2-labels.csv": "train-all-articles.csv",
    "test-S2-labels.csv": "test-S2-articles.csv"
}

for label_path, article_path in label_article_map.items():
    print(f" Processing labels: {label_path}")

    df_labels = pd.read_csv(label_path)
    original_cols = df_labels.columns.tolist()

    try:
        df_articles = pd.read_csv(article_path, usecols=["Filename", "Domain"])
    except FileNotFoundError:
        print(f" Article file not found: {article_path}")
        continue

    domain_map = dict(zip(df_articles["Filename"], df_articles["Domain"]))

    df_labels = fix_labels_preserve_structure(df_labels, domain_map)
    df_labels.to_csv(label_path, index=False, columns=original_cols)
    print(f" Fixed label prefixes and deduplicated: {label_path}")


import pandas as pd
import re

def clean_translated_articles(df, text_col="Translated_Text"):
    def clean_text(text):
        if pd.isna(text):
            return text
        text = text.strip()
        text = re.sub(r"\\n|\\t|\\+", " ", text)             # remove escaped characters
        text = re.sub(r"\bARTICLE\b|\bARTICLE TEXT\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text)                  # collapse multiple spaces
        return text.strip()

    df[text_col] = df[text_col].apply(clean_text)
    return df

# List of article files to clean
article_files = [
    "test-S1-articles.csv",
    "test-S2-articles.csv",
    "train-all-articles.csv",
    "train-S1-articles.csv"
]

# Apply cleaning and save
for file in article_files:
    print(f"Cleaning {file}...")
    df = pd.read_csv(file)
    if "Translated_Text" in df.columns:
        df = clean_translated_articles(df)
        df.to_csv(file, index=False)
        print(f" Cleaned and saved: {file}")
    else:
        print(f"Skipped {file} (missing 'Translated_Text' column)")

