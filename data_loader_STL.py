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
    y_train = mlb.transform(df_train[LABEL_COL])
    y_val = mlb.transform(df_val[LABEL_COL])
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
    test_s2_labels = test_s2_labels[["Filename", "Narrative"]]  # FIXED: no .columns overwrite

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


def prepare_data_STL_hierarchical(TASK,
                                   train_domains=["UA", "CC"],
                                   test_domains=["UA", "CC"],
                                   train_languages=["ALL"],
                                   debug=False):

    articles = pd.read_csv("Data/train-all-articles.csv")
    s1 = pd.read_csv("Data/train-S1-labels.csv")
    s2 = pd.read_csv("Data/train-S2-labels.csv")
    test_s1_articles = pd.read_csv("Data/test-S1-articles.csv")
    test_s1_labels = pd.read_csv("Data/test-S1-labels.csv")
    test_s2_articles = pd.read_csv("Data/test-S2-articles.csv")
    test_s2_labels = pd.read_csv("Data/test-S2-labels.csv")

    test_s1_labels.rename(columns={"Translated_Entity": "Entity"}, inplace=True)
    test_s2_labels.columns = ["Filename", "Narrative", "Subnarrative"]

    filtered_articles = articles[articles["Domain"].isin(train_domains)]
    if "ALL" not in train_languages:
        filtered_articles = filtered_articles[filtered_articles["Language"].isin(train_languages)]

    if TASK == "narrative_classification":
        TEXT_COL = "Translated_Text"
        LABEL_COL = "Label"

        df_all = pd.merge(filtered_articles, s2, on="Filename")

        def parse_labels(x):
            return [s.strip() for s in str(x).split(";") if s.strip().lower() != "nan"]

        df_all.dropna(subset=[TEXT_COL, "Label"], inplace=True)
        df_all[LABEL_COL] = df_all["Label"].apply(parse_labels)

        mlb = MultiLabelBinarizer()
        y_all = mlb.fit_transform(df_all[LABEL_COL])

        X = df_all[["Filename"]].values
        X_train, y_train, X_val, y_val = iterative_train_test_split(X, y_all, test_size=0.2)

        df_train = df_all[df_all["Filename"].isin(X_train.flatten())].copy()
        df_val = df_all[df_all["Filename"].isin(X_val.flatten())].copy()

        df_test = pd.merge(test_s2_articles, test_s2_labels, on="Filename")
        df_test.dropna(subset=[TEXT_COL, "Label"], inplace=True)
        df_test[LABEL_COL] = df_test["Label"].apply(parse_labels)
        y_test = mlb.transform(df_test[LABEL_COL])

        label_to_index = {label: i for i, label in enumerate(mlb.classes_)}

        # Hierarchy map
        parent_to_children = {
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
        child_to_parent = {child: parent for parent, children in parent_to_children.items() for child in children}

        if debug:
            df_train = df_train.sample(100)
            df_val = df_val.sample(100)
            df_test = df_test.sample(100)
            y_train = mlb.transform(df_train[LABEL_COL])
            y_val = mlb.transform(df_val[LABEL_COL])
            y_test = mlb.transform(df_test[LABEL_COL])

        return df_train, df_val, df_test, y_train, y_val, y_test, mlb, TEXT_COL, LABEL_COL, child_to_parent, label_to_index

    elif TASK == "entity_framing":
        TEXT_COL = "Translated_Text"
        LABEL_COL = "Label"

        df_all = pd.merge(s1, filtered_articles, on="Filename")

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

        df_all.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
        df_all["Start"] = df_all["Start"].astype(int)
        df_all["End"] = df_all["End"].astype(int)
        df_all["Input_Text"] = df_all.apply(lambda row: insert_entity_marker(row[TEXT_COL], row["Start"], row["End"]), axis=1)
        df_all[LABEL_COL] = df_all[LABEL_COL].apply(
            lambda x: list({ENTITY_PARENT_MAP.get(lbl.strip(), lbl.strip()) for lbl in str(x).split(",") if lbl.strip().lower() != "nan"})
        )

        mlb = MultiLabelBinarizer()
        y_all = mlb.fit_transform(df_all[LABEL_COL])
        X = df_all[["Filename"]].values
        X_train, y_train, X_val, y_val = iterative_train_test_split(X, y_all, test_size=0.2)

        df_train = df_all[df_all["Filename"].isin(X_train.flatten())].copy()
        df_val = df_all[df_all["Filename"].isin(X_val.flatten())].copy()

        df_test = pd.merge(test_s1_labels, test_s1_articles, on="Filename")
        df_test.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
        df_test["Start"] = df_test["Start"].astype(int)
        df_test["End"] = df_test["End"].astype(int)
        df_test["Input_Text"] = df_test.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
        df_test[LABEL_COL] = df_test[LABEL_COL].apply(
            lambda x: list({ENTITY_PARENT_MAP.get(lbl.strip(), lbl.strip()) for lbl in str(x).split(",") if lbl.strip().lower() != "nan"})
        )
        y_test = mlb.transform(df_test[LABEL_COL])

        label_to_index = {label: i for i, label in enumerate(mlb.classes_)}

        parent_to_children = {}
        for child, parent in ENTITY_PARENT_MAP.items():
            parent_to_children.setdefault(parent, []).append(child)
        child_to_parent = {child: parent for parent, children in parent_to_children.items() for child in children}

        if debug:
            df_train = df_train.sample(100)
            df_val = df_val.sample(100)
            df_test = df_test.sample(100)
            y_train = mlb.transform(df_train[LABEL_COL])
            y_val = mlb.transform(df_val[LABEL_COL])
            y_test = mlb.transform(df_test[LABEL_COL])

        return df_train, df_val, df_test, y_train, y_val, y_test, mlb, "Input_Text", LABEL_COL, child_to_parent, label_to_index

    else:
        raise NotImplementedError("Unknown or unsupported TASK.")

def prepare_data_STL_hierarchical(TASK,
                                   train_domains=["UA", "CC"],
                                   test_domains=["UA", "CC"],
                                   train_languages=["ALL"],
                                   debug=False):

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
    test_s2_labels.columns = ["Filename", "Narrative", "Subnarrative"]

    filtered_articles = articles[articles["Domain"].isin(train_domains)]
    if "ALL" not in train_languages:
        filtered_articles = filtered_articles[filtered_articles["Language"].isin(train_languages)]

    if TASK == "narrative_classification":
        TEXT_COL = "Translated_Text"
        LABEL_COL = "Label"

        df_all = pd.merge(filtered_articles, s2, on="Filename")

        def parse_labels(x):
            return [s.strip() for s in str(x).split(";") if s.strip().lower() != "nan"]

        df_all.dropna(subset=[TEXT_COL, "Subnarrative"], inplace=True)
        df_all[LABEL_COL] = df_all["Subnarrative"].apply(parse_labels)

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
        df_test.dropna(subset=[TEXT_COL, "Subnarrative"], inplace=True)
        df_test[LABEL_COL] = df_test["Subnarrative"].apply(parse_labels)

        label_to_index = {label: i for i, label in enumerate(mlb.classes_)}

        parent_to_children = {
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
        child_to_parent = {child: parent for parent, children in parent_to_children.items() for child in children}

        # Final label recomputation
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

        return df_train, df_val, df_test, y_train, y_val, y_test, mlb, TEXT_COL, LABEL_COL, child_to_parent, label_to_index

    elif TASK == "entity_framing":
        TEXT_COL = "Input_Text"
        LABEL_COL = "Label"

        df_all = pd.merge(s1, filtered_articles, on="Filename")

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

        df_all.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
        df_all["Start"] = df_all["Start"].astype(int)
        df_all["End"] = df_all["End"].astype(int)
        df_all["Input_Text"] = df_all.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
        df_all[LABEL_COL] = df_all[LABEL_COL].apply(
            lambda x: list({ENTITY_PARENT_MAP.get(lbl.strip(), lbl.strip()) for lbl in str(x).split(",") if lbl.strip().lower() != "nan"})
        )

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

        df_test = pd.merge(test_s1_labels, test_s1_articles, on="Filename")
        df_test.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
        df_test["Start"] = df_test["Start"].astype(int)
        df_test["End"] = df_test["End"].astype(int)
        df_test["Input_Text"] = df_test.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
        df_test[LABEL_COL] = df_test[LABEL_COL].apply(
            lambda x: list({ENTITY_PARENT_MAP.get(lbl.strip(), lbl.strip()) for lbl in str(x).split(",") if lbl.strip().lower() != "nan"})
        )

        # Final label recomputation
        y_train = mlb.transform(df_train[LABEL_COL])
        y_val   = mlb.transform(df_val[LABEL_COL])
        y_test  = mlb.transform(df_test[LABEL_COL])

        label_to_index = {label: i for i, label in enumerate(mlb.classes_)}
        parent_to_children = {}
        for child, parent in ENTITY_PARENT_MAP.items():
            parent_to_children.setdefault(parent, []).append(child)
        child_to_parent = {child: parent for parent, children in parent_to_children.items() for child in children}

        if debug:
            df_train = df_train.sample(100, random_state=42).reset_index(drop=True)
            df_val = df_val.sample(100, random_state=42).reset_index(drop=True)
            df_test = df_test.sample(100, random_state=42).reset_index(drop=True)
            y_train = mlb.transform(df_train[LABEL_COL])
            y_val = mlb.transform(df_val[LABEL_COL])
            y_test = mlb.transform(df_test[LABEL_COL])

        return df_train, df_val, df_test, y_train, y_val, y_test, mlb, TEXT_COL, LABEL_COL, child_to_parent, label_to_index

    else:
        raise NotImplementedError("Unknown or unsupported TASK.")
