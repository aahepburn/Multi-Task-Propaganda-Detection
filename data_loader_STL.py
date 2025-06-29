import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split


def prepare_data_STL_fine(
    TASK,
    train_domains=["UA", "CC"],
    test_domains=["UA", "CC"],
    train_languages=["ALL"],
    debug=False
):
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

    if TASK == "narrative_classification":
        TEXT_COL = "Translated_Text"
        LABEL_COL = "Label"

        df_all = pd.merge(filtered_articles, s2, on="Filename")

        def clean_labels(label_str):
            return [s.strip() for s in str(label_str).split(";") if s.strip().lower() != "nan"]

        df_all.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
        df_all[LABEL_COL] = df_all[LABEL_COL].apply(clean_labels)

    elif TASK == "entity_framing":
        TEXT_COL = "Input_Text"
        LABEL_COL = "Label"
        test_s1_labels.rename(columns={"Translated_Entity": "Entity"}, inplace=True)

        df_all = pd.merge(s1, filtered_articles, on="Filename")

        def insert_entity_marker(text, start, end):
            try:
                start, end = int(start), int(end)
                return text[:start] + "[ENTITY]" + text[start:end] + "[/ENTITY]" + text[end:]
            except:
                return text

        df_all.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
        df_all["Start"] = df_all["Start"].astype(int)
        df_all["End"] = df_all["End"].astype(int)
        df_all["Input_Text"] = df_all.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
        df_all[LABEL_COL] = df_all[LABEL_COL].apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip().lower() != "nan"])

    else:
        raise ValueError("Unknown TASK specified.")

    # Fit MultiLabelBinarizer and perform stratified split
    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(df_all[LABEL_COL])
    X = df_all["Filename"].values.reshape(-1, 1)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X, y_all, test_size=0.2)

    train_idx_map = {fname: i for i, fname in enumerate(X_train.flatten())}
    val_idx_map = {fname: i for i, fname in enumerate(X_val.flatten())}

    df_train = df_all[df_all["Filename"].isin(X_train.flatten())].copy()
    df_val = df_all[df_all["Filename"].isin(X_val.flatten())].copy()

    df_train["sort_idx"] = df_train["Filename"].map(train_idx_map)
    df_val["sort_idx"] = df_val["Filename"].map(val_idx_map)

    df_train = df_train.sort_values("sort_idx").drop(columns="sort_idx").reset_index(drop=True)
    df_val = df_val.sort_values("sort_idx").drop(columns="sort_idx").reset_index(drop=True)

    # Prepare test set
    if TASK == "narrative_classification":
        df_test = pd.merge(test_s2_articles, test_s2_labels, on="Filename")
        df_test.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
        df_test[LABEL_COL] = df_test[LABEL_COL].apply(clean_labels)
    else:  # entity_framing
        df_test = pd.merge(test_s1_labels, test_s1_articles, on="Filename")
        df_test.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
        df_test["Start"] = df_test["Start"].astype(int)
        df_test["End"] = df_test["End"].astype(int)
        df_test["Input_Text"] = df_test.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
        df_test[LABEL_COL] = df_test[LABEL_COL].apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip().lower() != "nan"])

    if debug:
        df_train = df_train.sample(100, random_state=42).reset_index(drop=True)
        df_val = df_val.sample(100, random_state=42).reset_index(drop=True)
        df_test = df_test.sample(100, random_state=42).reset_index(drop=True)

    # Final safety check: Recompute y_* from final DataFrames
    #y_train = mlb.transform(df_train[LABEL_COL])
   # y_val = mlb.transform(df_val[LABEL_COL])
   # y_test = mlb.transform(df_test[LABEL_COL])


    # Final safety check: Recompute y_* from final DataFrames
    y_train = mlb.transform(df_train[LABEL_COL])
    y_val = mlb.transform(df_val[LABEL_COL])

    known_labels = set(mlb.classes_)
    df_test[LABEL_COL] = df_test[LABEL_COL].apply(
        lambda labels: [l for l in labels if l in known_labels] if isinstance(labels, list) else []
    )
    y_test = mlb.transform(df_test[LABEL_COL])



    if debug:
        df_train = df_train.sample(100, random_state=42).reset_index(drop=True)
        df_val = df_val.sample(100, random_state=42).reset_index(drop=True)
        df_test = df_test.sample(100, random_state=42).reset_index(drop=True)
        y_train = mlb.transform(df_train[LABEL_COL])
        y_val = mlb.transform(df_val[LABEL_COL])
        y_test = mlb.transform(df_test[LABEL_COL])

    return df_train, df_val, df_test, y_train, y_val, y_test, mlb, TEXT_COL, LABEL_COL



def prepare_data_STL_coarse(TASK, train_domains=["UA", "CC"], test_domains=["UA", "CC"], train_languages=["ALL"], debug=False):
    import pandas as pd
    from sklearn.preprocessing import MultiLabelBinarizer
    from skmultilearn.model_selection import iterative_train_test_split

    articles = pd.read_csv("Data/train-all-articles.csv")
    s1 = pd.read_csv("Data/train-S1-labels.csv")
    s2 = pd.read_csv("Data/train-S2-labels.csv")
    test_s1_articles = pd.read_csv("Data/test-S1-articles.csv")
    test_s1_labels = pd.read_csv("Data/test-S1-labels.csv")
    test_s2_articles = pd.read_csv("Data/test-S2-articles.csv")
    test_s2_labels = pd.read_csv("Data/test-S2-labels.csv")

    test_s1_labels.rename(columns={"Translated_Entity": "Entity"}, inplace=True)

    filtered_articles = articles[articles["Domain"].isin(train_domains)]
    if "ALL" not in train_languages:
        filtered_articles = filtered_articles[filtered_articles["Language"].isin(train_languages)]

    if TASK == "narrative_classification":
        TEXT_COL = "Translated_Text"
        LABEL_COL = "Narrative"

        df_all = pd.merge(filtered_articles, s2, on="Filename")

        def get_main_narratives(x):
            return [s.split(":")[0].strip() for s in str(x).split(";") if s.strip().lower() != "nan"]

        df_all.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
        df_all[LABEL_COL] = df_all[LABEL_COL].apply(get_main_narratives)

        mlb = MultiLabelBinarizer()
        y_all = mlb.fit_transform(df_all[LABEL_COL])
        X = df_all["Filename"].values.reshape(-1, 1)
        X_train, y_train, X_val, y_val = iterative_train_test_split(X, y_all, test_size=0.2)

        train_idx_map = {fname: i for i, fname in enumerate(X_train.flatten())}
        val_idx_map = {fname: i for i, fname in enumerate(X_val.flatten())}

        df_train = df_all[df_all["Filename"].isin(X_train.flatten())].copy()
        df_val = df_all[df_all["Filename"].isin(X_val.flatten())].copy()

        df_train["sort_idx"] = df_train["Filename"].map(train_idx_map)
        df_val["sort_idx"] = df_val["Filename"].map(val_idx_map)

        df_train = df_train.sort_values("sort_idx").drop(columns="sort_idx").reset_index(drop=True)
        df_val = df_val.sort_values("sort_idx").drop(columns="sort_idx").reset_index(drop=True)

        df_test = pd.merge(test_s2_articles, test_s2_labels, on="Filename")
        df_test.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
        df_test[LABEL_COL] = df_test[LABEL_COL].apply(get_main_narratives)

    elif TASK == "entity_framing":
        TEXT_COL = "Input_Text"
        LABEL_COL = "Label"

        ENTITY_PARENT_MAP = {
            "Guardian": "Protagonist", "Martyr": "Protagonist", "Peacemaker": "Protagonist",
            "Rebel": "Protagonist", "Underdog": "Protagonist", "Virtuous": "Protagonist",
            "Instigator": "Antagonist", "Conspirator": "Antagonist", "Tyrant": "Antagonist",
            "Foreign Adversary": "Antagonist", "Traitor": "Antagonist", "Spy": "Antagonist",
            "Saboteur": "Antagonist", "Corrupt": "Antagonist", "Incompetent": "Antagonist",
            "Terrorist": "Antagonist", "Deceiver": "Antagonist", "Bigot": "Antagonist",
            "Forgotten": "Innocent", "Exploited": "Innocent", "Victim": "Innocent", "Scapegoat": "Innocent"
        }

        def insert_entity_marker(text, start, end):
            try:
                start, end = int(start), int(end)
                return text[:start] + "[ENTITY]" + text[start:end] + "[/ENTITY]" + text[end:]
            except:
                return text

        df_all = pd.merge(s1, filtered_articles, on="Filename")
        df_all.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
        df_all["Start"] = df_all["Start"].astype(int)
        df_all["End"] = df_all["End"].astype(int)
        df_all["Input_Text"] = df_all.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
        df_all[LABEL_COL] = df_all[LABEL_COL].apply(
            lambda x: [ENTITY_PARENT_MAP.get(label.strip(), label.strip()) for label in str(x).split(",") if label.strip().lower() != "nan"]
        )
        df_all[LABEL_COL] = df_all[LABEL_COL].apply(lambda x: list(set(x)))

        mlb = MultiLabelBinarizer()
        y_all = mlb.fit_transform(df_all[LABEL_COL])
        X = df_all["Filename"].values.reshape(-1, 1)
        X_train, y_train, X_val, y_val = iterative_train_test_split(X, y_all, test_size=0.2)

        train_idx_map = {fname: i for i, fname in enumerate(X_train.flatten())}
        val_idx_map = {fname: i for i, fname in enumerate(X_val.flatten())}

        df_train = df_all[df_all["Filename"].isin(X_train.flatten())].copy()
        df_val = df_all[df_all["Filename"].isin(X_val.flatten())].copy()

        df_train["sort_idx"] = df_train["Filename"].map(train_idx_map)
        df_val["sort_idx"] = df_val["Filename"].map(val_idx_map)

        df_train = df_train.sort_values("sort_idx").drop(columns="sort_idx").reset_index(drop=True)
        df_val = df_val.sort_values("sort_idx").drop(columns="sort_idx").reset_index(drop=True)

        df_test = pd.merge(test_s1_articles, test_s1_labels, on="Filename")
        df_test.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
        df_test["Start"] = df_test["Start"].astype(int)
        df_test["End"] = df_test["End"].astype(int)
        df_test["Input_Text"] = df_test.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
        df_test[LABEL_COL] = df_test[LABEL_COL].apply(
            lambda x: [ENTITY_PARENT_MAP.get(label.strip(), label.strip()) for label in str(x).split(",") if label.strip().lower() != "nan"]
        )
        df_test[LABEL_COL] = df_test[LABEL_COL].apply(lambda x: list(set(x)))

    else:
        raise ValueError("Unknown TASK specified.")

    # === Final safety recomputation ===
    y_train = mlb.transform(df_train[LABEL_COL])
    y_val   = mlb.transform(df_val[LABEL_COL])
    y_test  = mlb.transform(df_test[LABEL_COL])

    if debug:
        df_train = df_train.sample(100, random_state=42).reset_index(drop=True)
        df_val = df_val.sample(100, random_state=42).reset_index(drop=True)
        df_test = df_test.sample(100, random_state=42).reset_index(drop=True)
        y_train = mlb.transform(df_train[LABEL_COL])
        y_val = mlb.transform(df_val[LABEL_COL])
        y_test = mlb.transform(df_test[LABEL_COL])

    return df_train, df_val, df_test, y_train, y_val, y_test, mlb, TEXT_COL, LABEL_COL

