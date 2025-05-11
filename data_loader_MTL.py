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



# ===================================
# Hierarchical
# ===================================


from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd

def prepare_data_MTL_hierarchical(
    TASK, model_name, max_len, batch_size, train_domains, test_domains, train_languages, debug=False
):
    # Load data
    articles = pd.read_csv("Data/train-all-articles.csv")
    s1 = pd.read_csv("Data/train-S1-labels.csv")
    s2 = pd.read_csv("Data/train-S2-labels.csv")
    test_s1_articles = pd.read_csv("Data/test-S1-articles.csv")
    test_s1_labels = pd.read_csv("Data/test-S1-labels.csv")
    test_s2_articles = pd.read_csv("Data/test-S2-articles.csv")
    test_s2_labels = pd.read_csv("Data/test-S2-labels.csv")

    filtered_articles = articles[articles["Domain"].isin(train_domains)]
    if "ALL" not in train_languages:
        filtered_articles = filtered_articles[filtered_articles["Language"].isin(train_languages)]

    # === STRATIFIED SPLIT FOR ENTITY FRAMING (S1) ===
    s1_labeled = pd.merge(filtered_articles, s1, on="Filename")
    s1_labeled["Entity_Labels"] = s1_labeled["Label"].apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip()])
    s1_labeled.dropna(subset=["Entity_Labels"], inplace=True)

    mlb_s1 = MultiLabelBinarizer()
    y_full_s1 = mlb_s1.fit_transform(s1_labeled["Entity_Labels"])
    X_s1 = s1_labeled[["Filename"]].values
    X_train_s1, _, X_val_s1, _ = iterative_train_test_split(X_s1, y_full_s1, test_size=0.2)

    train_articles_s1 = filtered_articles[filtered_articles["Filename"].isin(X_train_s1.flatten())]
    val_articles_s1 = filtered_articles[filtered_articles["Filename"].isin(X_val_s1.flatten())]

    # === STRATIFIED SPLIT FOR NARRATIVE CLASSIFICATION (S2) ===
    s2_labeled = pd.merge(filtered_articles, s2, on="Filename")
    s2_labeled["Narrative_Labels"] = s2_labeled["Label"].apply(lambda x: [s.strip() for s in str(x).split(";") if s.strip()])
    s2_labeled.dropna(subset=["Narrative_Labels"], inplace=True)

    mlb_s2 = MultiLabelBinarizer()
    y_full_s2 = mlb_s2.fit_transform(s2_labeled["Narrative_Labels"])
    X_s2 = s2_labeled[["Filename"]].values
    X_train_s2, _, X_val_s2, _ = iterative_train_test_split(X_s2, y_full_s2, test_size=0.2)

    train_articles_s2 = filtered_articles[filtered_articles["Filename"].isin(X_train_s2.flatten())]
    val_articles_s2 = filtered_articles[filtered_articles["Filename"].isin(X_val_s2.flatten())]

    if debug:
        train_articles_s1 = train_articles_s1.sample(100)
        val_articles_s1 = val_articles_s1.sample(100)
        train_articles_s2 = train_articles_s2.sample(100)
        val_articles_s2 = val_articles_s2.sample(100)
        test_s1_articles = test_s1_articles.sample(100)
        test_s2_articles = test_s2_articles.sample(100)

    # --- ENTITY FRAMING (with hierarchy) ---
    hierarchy_map_ef = {
        "Guardian": "Protagonist", "Martyr": "Protagonist", "Peacemaker": "Protagonist", "Rebel": "Protagonist",
        "Underdog": "Protagonist", "Virtuous": "Protagonist",
        "Instigator": "Antagonist", "Conspirator": "Antagonist", "Tyrant": "Antagonist", "Foreign Adversary": "Antagonist",
        "Traitor": "Antagonist", "Spy": "Antagonist", "Saboteur": "Antagonist", "Corrupt": "Antagonist",
        "Incompetent": "Antagonist", "Terrorist": "Antagonist", "Deceiver": "Antagonist", "Bigot": "Antagonist",
        "Forgotten": "Innocent", "Exploited": "Innocent", "Victim": "Innocent", "Scapegoat": "Innocent"
    }

    def clean_and_expand_roles(x):
        labels = [s.strip() for s in str(x).split(",") if s.strip()]
        expanded = set(labels)
        for label in labels:
            parent = hierarchy_map_ef.get(label)
            if parent:
                expanded.add(parent)
        return list(expanded)

    df_train_s1 = pd.merge(s1, train_articles_s1, on="Filename")
    df_val_s1 = pd.merge(s1, val_articles_s1, on="Filename")
    df_test_s1 = pd.merge(test_s1_labels, test_s1_articles, on="Filename")

    df_train_s1["Label"] = df_train_s1["Label"].apply(clean_and_expand_roles)
    df_val_s1["Label"] = df_val_s1["Label"].apply(clean_and_expand_roles)
    df_test_s1["Label"] = df_test_s1["Label"].apply(clean_and_expand_roles)

    y_train_s1 = mlb_s1.transform(df_train_s1["Label"])
    y_val_s1 = mlb_s1.transform(df_val_s1["Label"])
    y_test_s1 = mlb_s1.transform(df_test_s1["Label"])

    # --- NARRATIVE CLASSIFICATION (with hierarchy) ---
    def clean_labels_semicolon(x):
        return [s.strip() for s in str(x).split(";") if s.strip() and s.strip().lower() != "nan"]

    df_train_s2 = pd.merge(train_articles_s2, s2, on="Filename")
    df_val_s2 = pd.merge(val_articles_s2, s2, on="Filename")
    df_test_s2 = pd.merge(test_s2_articles, test_s2_labels, on="Filename")

    df_train_s2["Narrative"] = df_train_s2["Label"].apply(clean_labels_semicolon)
    df_val_s2["Narrative"] = df_val_s2["Label"].apply(clean_labels_semicolon)
    df_test_s2["Narrative"] = df_test_s2["Label"].apply(clean_labels_semicolon)

    y_train_s2 = mlb_s2.transform(df_train_s2["Narrative"])
    y_val_s2 = mlb_s2.transform(df_val_s2["Narrative"])
    y_test_s2 = mlb_s2.transform(df_test_s2["Narrative"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ENTITY hierarchy maps
    child_to_parent_ef = {child: parent for child, parent in hierarchy_map_ef.items()}
    label_to_index_ef = {label: i for i, label in enumerate(mlb_s1.classes_)}

    # NARRATIVE hierarchy maps
    parent_to_children_nc = {
            # ---------- URW ----------
            "URW: Blaming the war on others rather than the invader": [
                "URW: Ukraine is the aggressor",
                "URW: The West are the aggressors"
            ],
            "URW: Discrediting Ukraine": [
                "URW: Rewriting Ukraine’s history",
                "URW: Discrediting Ukrainian nation and society",
                "URW: Discrediting Ukrainian military",
                "URW: Discrediting Ukrainian government and officials and policies",
                "URW: Ukraine is a puppet of the West",
                "URW: Ukraine is a hub for criminal activities",
                "URW: Ukraine is associated with nazism",
                "URW: Situation in Ukraine is hopeless"
            ],
            "URW: Russia is the Victim": [
                "URW: The West is russophobic",
                "URW: Russia actions in Ukraine are only self-defence",
                "URW: UA is anti-RU extremists"
            ],
            "URW: Praise of Russia": [
                "URW: Praise of Russian military might",
                "URW: Praise of Russian President Vladimir Putin",
                "URW: Russia is a guarantor of peace and prosperity",
                "URW: Russia has international support from a number of countries and people",
                "URW: Russian invasion has strong national support"
            ],
            "URW: Overpraising the West": [
                "URW: NATO will destroy Russia",
                "URW: The West belongs in the right side of history",
                "URW: The West has the strongest international support"
            ],
            "URW: Speculating war outcomes": [
                "URW: Russian army is collapsing",
                "URW: Russian army will lose all the occupied territories",
                "URW: Ukrainian army is collapsing"
            ],
            "URW: Discrediting the West, Diplomacy": [
                "URW: The EU is divided",
                "URW: The West is weak",
                "URW: The West is overreacting",
                "URW: The West does not care about Ukraine, only about its interests",
                "URW: Diplomacy does/will not work",
                "URW: West is tired of Ukraine"
            ],
            "URW: Negative Consequences for the West": [
                "URW: Sanctions imposed by Western countries will backfire",
                "URW: The conflict will increase the Ukrainian refugee flows to Europe"
            ],
            "URW: Distrust towards Media": [
                "URW: Western media is an instrument of propaganda",
                "URW: Ukrainian media cannot be trusted"
            ],
            "URW: Amplifying war-related fears": [
                "URW: By continuing the war we risk WWIII",
                "URW: Russia will also attack other countries",
                "URW: There is a real possibility that nuclear weapons will be employed",
                "URW: NATO should/will directly intervene"
            ],
            "URW: Hidden plots by secret schemes of powerful groups": [
                "URW: Hidden plots by secret schemes of powerful groups"
            ],

            # ---------- CC ----------
            "CC: Criticism of climate policies": [
                "CC: Climate policies are ineffective",
                "CC: Climate policies have negative impact on the economy",
                "CC: Climate policies are only for profit"
            ],
            "CC: Criticism of institutions and authorities": [
                "CC: Criticism of the EU",
                "CC: Criticism of international entities",
                "CC: Criticism of national governments",
                "CC: Criticism of political organizations and figures"
            ],
            "CC: Climate change is beneficial": [
                "CC: CO2 is beneficial",
                "CC: Temperature increase is beneficial"
            ],
            "CC: Downplaying climate change": [
                "CC: Climate cycles are natural",
                "CC: Weather suggests the trend is global cooling",
                "CC: Temperature increase does not have significant impact",
                "CC: CO2 concentrations are too small to have an impact",
                "CC: Human activities do not impact climate change",
                "CC: Ice is not melting",
                "CC: Sea levels are not rising",
                "CC: Humans and nature will adapt to the changes"
            ],
            "CC: Questioning the measurements and science": [
                "CC: Methodologies/metrics used are unreliable/faulty",
                "CC: Data shows no temperature increase",
                "CC: Greenhouse effect/carbon dioxide do not drive climate change",
                "CC: Scientific community is unreliable"
            ],
            "CC: Criticism of climate movement": [
                "CC: Climate movement is alarmist",
                "CC: Climate movement is corrupt",
                "CC: Ad hominem attacks on key activists"
            ],
            "CC: Controversy about green technologies": [
                "CC: Renewable energy is dangerous",
                "CC: Renewable energy is unreliable",
                "CC: Renewable energy is costly",
                "CC: Nuclear energy is not climate friendly"
            ],
            "CC: Hidden plots by secret schemes of powerful groups": [
                "CC: Blaming global elites",
                "CC: Climate agenda has hidden motives"
            ],
            "CC: Amplifying Climate Fears": [
                "CC: Earth will be uninhabitable soon",
                "CC: Amplifying existing fears of global warming",
                "CC: Doomsday scenarios for humans",
                "CC: Whatever we do it is already too late"
            ],
            "CC: Green policies are geopolitical instruments": [
                "CC: Climate-related international relations are abusive/exploitative",
                "CC: Green activities are a form of neo-colonialism"
            ]
    }
    child_to_parent_nc = {child: parent for parent, children in parent_to_children_nc.items() for child in children}
    label_to_index_nc = {label: i for i, label in enumerate(mlb_s2.classes_)}

    return (
        df_train_s1, df_val_s1, df_test_s1, y_train_s1, y_val_s1, y_test_s1, mlb_s1,
        df_train_s2, df_val_s2, df_test_s2, y_train_s2, y_val_s2, y_test_s2, mlb_s2,
        make_loader(df_train_s1["Translated_Text"].tolist(), y_train_s1, "entity_framing", tokenizer, max_len, batch_size),
        make_loader(df_val_s1["Translated_Text"].tolist(), y_val_s1, "entity_framing", tokenizer, max_len, batch_size, shuffle=False),
        make_loader(df_test_s1["Translated_Text"].tolist(), y_test_s1, "entity_framing", tokenizer, max_len, batch_size, shuffle=False),
        make_loader(df_train_s2["Translated_Text"].tolist(), y_train_s2, "narrative_classification", tokenizer, max_len, batch_size),
        make_loader(df_val_s2["Translated_Text"].tolist(), y_val_s2, "narrative_classification", tokenizer, max_len, batch_size, shuffle=False),
        make_loader(df_test_s2["Translated_Text"].tolist(), y_test_s2, "narrative_classification", tokenizer, max_len, batch_size, shuffle=False),
        {"entity_framing": y_train_s1.shape[1], "narrative_classification": y_train_s2.shape[1]},
        {"entity_framing": child_to_parent_ef, "narrative_classification": child_to_parent_nc},
        {"entity_framing": label_to_index_ef, "narrative_classification": label_to_index_nc}
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
