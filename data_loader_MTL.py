import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from skmultilearn.model_selection import iterative_train_test_split


# ===================================
# MultiTask Dataset + DataLoader
# ===================================

class MultiTaskDataset(Dataset):
    def __init__(self, texts, task_labels_dict, tokenizer, max_len):
        self.texts = texts
        self.task_labels_dict = task_labels_dict
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        for task_name, label_matrix in self.task_labels_dict.items():
            item[f"{task_name}_labels"] = torch.tensor(label_matrix[idx], dtype=torch.float)
        return item

def make_loader(texts, labels, task_name, tokenizer, max_len, batch_size, shuffle=True):
    dataset = MultiTaskDataset(texts, {task_name: labels}, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ===================================
# Fine-Grained Flat
# ===================================


def prepare_data_MTL_fine_flat(
    TASK,
    model_name,
    max_len,
    batch_size,
    train_domains,
    test_domains,
    train_languages,
    debug=False
    ):
   
    # Load files
    articles = pd.read_csv("Data/train-all-articles.csv")
    s1 = pd.read_csv("Data/train-S1-labels.csv")
    s2 = pd.read_csv("Data/train-S2-labels.csv")
    test_s1_articles = pd.read_csv("Data/test-S1-articles.csv")
    test_s1_labels = pd.read_csv("Data/test-S1-labels.csv")
    test_s2_articles = pd.read_csv("Data/test-S2-articles.csv")
    test_s2_labels = pd.read_csv("Data/test-S2-labels.csv")

    # Filter training articles
    filtered_articles = articles[articles["Domain"].isin(train_domains)]
    if "ALL" not in train_languages:
        filtered_articles = filtered_articles[filtered_articles["Language"].isin(train_languages)]

    # ========== Stratified Split for S1 ==========
    s1_labeled = pd.merge(filtered_articles, s1, on="Filename")
    s1_labeled["Entity_Labels"] = s1_labeled["Label"].apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip()])
    s1_labeled.dropna(subset=["Entity_Labels"], inplace=True)

    mlb_s1 = MultiLabelBinarizer()
    y_full_s1 = mlb_s1.fit_transform(s1_labeled["Entity_Labels"])
    X_s1 = s1_labeled[["Filename"]].values
    X_train_s1, y_train_dummy, X_val_s1, y_val_dummy = iterative_train_test_split(X_s1, y_full_s1, test_size=0.2)

    train_filenames_s1 = set(X_train_s1.flatten())
    val_filenames_s1 = set(X_val_s1.flatten())
    train_articles_s1 = filtered_articles[filtered_articles["Filename"].isin(train_filenames_s1)]
    val_articles_s1 = filtered_articles[filtered_articles["Filename"].isin(val_filenames_s1)]

    # ========== Stratified Split for S2 ==========
    s2_labeled = pd.merge(filtered_articles, s2, on="Filename")
    s2_labeled["Narrative_Labels"] = s2_labeled["Label"].apply(lambda x: [s.strip() for s in str(x).split(";") if s.strip()])
    s2_labeled.dropna(subset=["Narrative_Labels"], inplace=True)

    mlb_s2 = MultiLabelBinarizer()
    y_full_s2 = mlb_s2.fit_transform(s2_labeled["Narrative_Labels"])
    X_s2 = s2_labeled[["Filename"]].values
    X_train_s2, y_train_dummy, X_val_s2, y_val_dummy = iterative_train_test_split(X_s2, y_full_s2, test_size=0.2)

    train_filenames_s2 = set(X_train_s2.flatten())
    val_filenames_s2 = set(X_val_s2.flatten())
    train_articles_s2 = filtered_articles[filtered_articles["Filename"].isin(train_filenames_s2)]
    val_articles_s2 = filtered_articles[filtered_articles["Filename"].isin(val_filenames_s2)]

    if debug:
        train_articles_s1 = train_articles_s1.sample(100)
        val_articles_s1 = val_articles_s1.sample(100)
        train_articles_s2 = train_articles_s2.sample(100)
        val_articles_s2 = val_articles_s2.sample(100)
        test_s1_articles = test_s1_articles.sample(100)
        test_s2_articles = test_s2_articles.sample(100)

    def insert_entity_marker(text, start, end):
        try:
            return text[:int(start)] + "[ENTITY]" + text[int(start):int(end)] + "[/ENTITY]" + text[int(end):]
        except:
            return text

    # ENTITY FRAMING
    df_train_s1 = pd.merge(s1, train_articles_s1, on="Filename")
    df_val_s1 = pd.merge(s1, val_articles_s1, on="Filename")
    df_test_s1 = pd.merge(test_s1_labels, test_s1_articles, on="Filename")
    for df in [df_train_s1, df_val_s1, df_test_s1]:
        df.dropna(subset=["Translated_Text", "Entity", "Label", "Start", "End"], inplace=True)
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int)
        df["Input_Text"] = df.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
        df["Entity_Labels"] = df["Label"].apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip()])

    y_train_s1 = mlb_s1.transform(df_train_s1["Entity_Labels"])
    y_val_s1 = mlb_s1.transform(df_val_s1["Entity_Labels"])
    y_test_s1 = mlb_s1.transform(df_test_s1["Label"].apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip()]))

    # NARRATIVE CLASSIFICATION
    df_train_s2 = pd.merge(train_articles_s2, s2, on="Filename")
    df_val_s2 = pd.merge(val_articles_s2, s2, on="Filename")
    df_test_s2 = pd.merge(test_s2_articles, test_s2_labels, on="Filename")

    for df in [df_train_s2, df_val_s2, df_test_s2]:
        df.dropna(subset=["Translated_Text", "Label"], inplace=True)
        df["Narrative_Labels"] = df["Label"].apply(lambda x: [s.strip() for s in str(x).split(";") if s.strip()])

    y_train_s2 = mlb_s2.transform(df_train_s2["Narrative_Labels"])
    y_val_s2 = mlb_s2.transform(df_val_s2["Narrative_Labels"])
    known_narratives = set(mlb_s2.classes_)
    df_test_s2["Narrative_Labels"] = df_test_s2["Narrative_Labels"].apply(
    lambda labels: [l for l in labels if l in known_narratives] if isinstance(labels, list) else []
    )
    y_test_s2 = mlb_s2.transform(df_test_s2["Narrative_Labels"])


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_train_s1 = MultiTaskDataset(df_train_s1["Input_Text"].tolist(), {"entity_framing": y_train_s1}, tokenizer, max_len)
    dataset_val_s1   = MultiTaskDataset(df_val_s1["Input_Text"].tolist(), {"entity_framing": y_val_s1}, tokenizer, max_len)
    dataset_test_s1  = MultiTaskDataset(df_test_s1["Input_Text"].tolist(), {"entity_framing": y_test_s1}, tokenizer, max_len)

    dataset_train_s2 = MultiTaskDataset(df_train_s2["Translated_Text"].tolist(), {"narrative_classification": y_train_s2}, tokenizer, max_len)
    dataset_val_s2   = MultiTaskDataset(df_val_s2["Translated_Text"].tolist(), {"narrative_classification": y_val_s2}, tokenizer, max_len)
    dataset_test_s2  = MultiTaskDataset(df_test_s2["Translated_Text"].tolist(), {"narrative_classification": y_test_s2}, tokenizer, max_len)

    return (
        df_train_s1, df_val_s1, df_test_s1, y_train_s1, y_val_s1, y_test_s1, mlb_s1,
        df_train_s2, df_val_s2, df_test_s2, y_train_s2, y_val_s2, y_test_s2, mlb_s2,
        DataLoader(dataset_train_s1, batch_size=batch_size, shuffle=True),
        DataLoader(dataset_val_s1, batch_size=batch_size),
        DataLoader(dataset_test_s1, batch_size=batch_size),
        DataLoader(dataset_train_s2, batch_size=batch_size, shuffle=True),
        DataLoader(dataset_val_s2, batch_size=batch_size),
        DataLoader(dataset_test_s2, batch_size=batch_size),
        {"entity_framing": y_train_s1.shape[1], "narrative_classification": y_train_s2.shape[1]}
    )




# Coarse MTL Preprocessing Function

def prepare_data_MTL_coarse(TASK, model_name, max_len, batch_size, train_domains, test_domains, train_languages, debug=False):
    # Load data
    articles = pd.read_csv("Data/train-all-articles.csv")
    s1 = pd.read_csv("Data/train-S1-labels.csv")
    s2 = pd.read_csv("Data/train-S2-labels.csv")
    test_s1_articles = pd.read_csv("Data/test-S1-articles.csv")
    test_s1_labels = pd.read_csv("Data/test-S1-labels.csv")
    test_s2_articles = pd.read_csv("Data/test-S2-articles.csv")
    test_s2_labels = pd.read_csv("Data/test-S2-labels.csv")

    # Filter
    filtered_articles = articles[articles["Domain"].isin(train_domains)]
    if "ALL" not in train_languages:
        filtered_articles = filtered_articles[filtered_articles["Language"].isin(train_languages)]

    # Define coarse label map
    coarse_map = {
        "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
        "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor", "Spy",
                       "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
        "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"]
    }
    inverse_map = {v: k for k, vals in coarse_map.items() for v in vals}

    def map_coarse_labels(x):
        return list({inverse_map.get(label.strip(), label.strip()) for label in str(x).split(",") if label.strip()})

    def parse_narrative_labels(x):
        return [s.strip() for s in str(x).split(";") if s.strip().lower() != "nan"]

    # --- Stratified split for S1 ---
    s1_labeled = pd.merge(filtered_articles, s1, on="Filename")
    s1_labeled["Coarse_Labels"] = s1_labeled["Label"].apply(map_coarse_labels)
    s1_labeled.dropna(subset=["Coarse_Labels"], inplace=True)

    mlb_s1 = MultiLabelBinarizer()
    y_s1_full = mlb_s1.fit_transform(s1_labeled["Coarse_Labels"])
    X_s1 = s1_labeled[["Filename"]].values
    X_train_s1, _, X_val_s1, _ = iterative_train_test_split(X_s1, y_s1_full, test_size=0.2)

    train_articles_s1 = filtered_articles[filtered_articles["Filename"].isin(X_train_s1.flatten())]
    val_articles_s1 = filtered_articles[filtered_articles["Filename"].isin(X_val_s1.flatten())]

    # --- Stratified split for S2 ---
    s2_labeled = pd.merge(filtered_articles, s2, on="Filename")
    s2_labeled["Narrative_Labels"] = s2_labeled["Narrative"].apply(parse_narrative_labels)
    s2_labeled.dropna(subset=["Narrative_Labels"], inplace=True)

    mlb_s2 = MultiLabelBinarizer()
    y_s2_full = mlb_s2.fit_transform(s2_labeled["Narrative_Labels"])
    X_s2 = s2_labeled[["Filename"]].values
    X_train_s2, _, X_val_s2, _ = iterative_train_test_split(X_s2, y_s2_full, test_size=0.2)

    train_articles_s2 = filtered_articles[filtered_articles["Filename"].isin(X_train_s2.flatten())]
    val_articles_s2 = filtered_articles[filtered_articles["Filename"].isin(X_val_s2.flatten())]

    if debug:
        train_articles_s1 = train_articles_s1.sample(100)
        val_articles_s1 = val_articles_s1.sample(100)
        train_articles_s2 = train_articles_s2.sample(100)
        val_articles_s2 = val_articles_s2.sample(100)
        test_s1_articles = test_s1_articles.sample(100)
        test_s2_articles = test_s2_articles.sample(100)

    # Process S1
    df_train_s1 = pd.merge(s1, train_articles_s1, on="Filename")
    df_val_s1 = pd.merge(s1, val_articles_s1, on="Filename")
    df_test_s1 = pd.merge(test_s1_labels, test_s1_articles, on="Filename")

    for df in [df_train_s1, df_val_s1, df_test_s1]:
        df["Label"] = df["Label"].apply(map_coarse_labels)

    y_train_s1 = mlb_s1.transform(df_train_s1["Label"])
    y_val_s1 = mlb_s1.transform(df_val_s1["Label"])
    y_test_s1 = mlb_s1.transform(df_test_s1["Label"])

    # Process S2
    df_train_s2 = pd.merge(train_articles_s2, s2, on="Filename")
    df_val_s2 = pd.merge(val_articles_s2, s2, on="Filename")
    df_test_s2 = pd.merge(test_s2_articles, test_s2_labels, on="Filename")

    for df in [df_train_s2, df_val_s2, df_test_s2]:
        df["Narrative"] = df["Narrative"].apply(parse_narrative_labels)

    y_train_s2 = mlb_s2.transform(df_train_s2["Narrative"])
    y_val_s2 = mlb_s2.transform(df_val_s2["Narrative"])
    y_test_s2 = mlb_s2.transform(df_test_s2["Narrative"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return (
        df_train_s1, df_val_s1, df_test_s1, y_train_s1, y_val_s1, y_test_s1, mlb_s1,
        df_train_s2, df_val_s2, df_test_s2, y_train_s2, y_val_s2, y_test_s2, mlb_s2,
        make_loader(df_train_s1["Translated_Text"].tolist(), y_train_s1, "entity_framing", tokenizer, max_len, batch_size),
        make_loader(df_val_s1["Translated_Text"].tolist(), y_val_s1, "entity_framing", tokenizer, max_len, batch_size, shuffle=False),
        make_loader(df_test_s1["Translated_Text"].tolist(), y_test_s1, "entity_framing", tokenizer, max_len, batch_size, shuffle=False),
        make_loader(df_train_s2["Translated_Text"].tolist(), y_train_s2, "narrative_classification", tokenizer, max_len, batch_size),
        make_loader(df_val_s2["Translated_Text"].tolist(), y_val_s2, "narrative_classification", tokenizer, max_len, batch_size, shuffle=False),
        make_loader(df_test_s2["Translated_Text"].tolist(), y_test_s2, "narrative_classification", tokenizer, max_len, batch_size, shuffle=False),
        {"entity_framing": y_train_s1.shape[1], "narrative_classification": y_train_s2.shape[1]}
    )


def prepare_data_MTL_mixed(
    task,
    train_domains,
    test_domains,
    train_languages,
    model_name,
    max_len,
    batch_size,
    granularity_s1="fine",
    granularity_s2="fine"
):
    # Load task 1
    if granularity_s1 == "fine":
        (
            df_train_s1, df_val_s1, df_test_s1, y_train_s1, y_val_s1, y_test_s1, mlb_s1,
            _, _, _, _, _, _, _,  # ignore task2 outputs
            train_loader_s1, val_loader_s1, test_loader_s1,
            _, _, _
        ) = prepare_data_MTL_fine_flat(
            task,
            train_domains=train_domains,
            test_domains=test_domains,
            train_languages=train_languages,
            model_name=model_name,
            max_len=max_len,
            batch_size=batch_size
        )
    elif granularity_s1 == "coarse":
        (
            df_train_s1, df_val_s1, df_test_s1, y_train_s1, y_val_s1, y_test_s1, mlb_s1,
            _, _, _, _, _, _, _,  # ignore task2 outputs
            train_loader_s1, val_loader_s1, test_loader_s1,
            _, _, _
        ) = prepare_data_MTL_coarse(
            task,
            train_domains=train_domains,
            test_domains=test_domains,
            train_languages=train_languages,
            model_name=model_name,
            max_len=max_len,
            batch_size=batch_size
        )

    # Load task 2
    if granularity_s2 == "fine":
        (
            _, _, _, _, _, _, _,
            df_train_s2, df_val_s2, df_test_s2, y_train_s2, y_val_s2, y_test_s2, mlb_s2,
            _, _, _,
            train_loader_s2, val_loader_s2, test_loader_s2,
            _
        ) = prepare_data_MTL_fine_flat(
            task,
            train_domains=train_domains,
            test_domains=test_domains,
            train_languages=train_languages,
            model_name=model_name,
            max_len=max_len,
            batch_size=batch_size
        )
    elif granularity_s2 == "coarse":
        (
            _, _, _, _, _, _, _,
            df_train_s2, df_val_s2, df_test_s2, y_train_s2, y_val_s2, y_test_s2, mlb_s2,
            _, _, _,
            train_loader_s2, val_loader_s2, test_loader_s2,
            _
        ) = prepare_data_MTL_coarse(
            task,
            train_domains=train_domains,
            test_domains=test_domains,
            train_languages=train_languages,
            model_name=model_name,
            max_len=max_len,
            batch_size=batch_size
        )

    # Class count dictionary (can be reassembled from mlb)
    num_classes_dict = {
        "task1": len(mlb_s1.classes_),
        "task2": len(mlb_s2.classes_)
    }

    return (
        df_train_s1, df_val_s1, df_test_s1, y_train_s1, y_val_s1, y_test_s1, mlb_s1,
        df_train_s2, df_val_s2, df_test_s2, y_train_s2, y_val_s2, y_test_s2, mlb_s2,
        train_loader_s1, val_loader_s1, test_loader_s1,
        train_loader_s2, val_loader_s2, test_loader_s2,
        num_classes_dict
    )

