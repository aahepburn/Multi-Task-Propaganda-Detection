import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def prepare_data_STL_fine(TASK,
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

    filtered_articles = articles[articles["Domain"].isin(train_domains)]
    if "ALL" not in train_languages:
        filtered_articles = filtered_articles[filtered_articles["Language"].isin(train_languages)]

    filtered_articles = filtered_articles.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(filtered_articles))
    train_articles = filtered_articles.iloc[:split_idx].copy()
    val_articles = filtered_articles.iloc[split_idx:].copy()

    if debug:
        train_articles = train_articles.sample(100)
        val_articles = val_articles.sample(100)
        test_s1_articles = test_s1_articles.sample(100)
        test_s2_articles = test_s2_articles.sample(100)

    if TASK == "narrative_classification":
        df_train = pd.merge(train_articles, s2, on="Filename")
        df_val   = pd.merge(val_articles, s2, on="Filename")
        df_test  = pd.merge(test_s2_articles, test_s2_labels, on="Filename")

        TEXT_COL = "Translated_Text"
        LABEL_COL = "Narrative"

        for df in [df_train, df_val, df_test]:
            df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
            df[LABEL_COL] = df[LABEL_COL].apply(
                lambda x: [s.strip() for s in str(x).split(";") if s.strip().lower() != "nan"]
            )

        full_set = pd.concat([df_train, df_val, df_test])
        mlb = MultiLabelBinarizer()
        all_labels = (
            df_train[LABEL_COL].dropna().tolist() +
            df_val[LABEL_COL].dropna().tolist() +
            df_test[LABEL_COL].dropna().tolist()
        )
        mlb.fit(all_labels)

    elif TASK == "entity_framing":
        df_train = pd.merge(s1, train_articles, on="Filename")
        df_val   = pd.merge(s1, val_articles, on="Filename")
        df_test  = pd.merge(test_s1_labels, test_s1_articles, on="Filename")

        TEXT_COL = "Translated_Text"
        LABEL_COL = "Label"

        def insert_entity_marker(text, start, end):
            try:
                start, end = int(start), int(end)
                return text[:start] + "[ENTITY]" + text[start:end] + "[/ENTITY]" + text[end:]
            except:
                return text

        for df in [df_train, df_val, df_test]:
            df.dropna(subset=[TEXT_COL, "Entity", LABEL_COL, "Start", "End"], inplace=True)
            df["Start"] = df["Start"].astype(int)
            df["End"] = df["End"].astype(int)
            df["Input_Text"] = df.apply(lambda row: insert_entity_marker(row[TEXT_COL], row["Start"], row["End"]), axis=1)
            df[LABEL_COL] = df[LABEL_COL].apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip().lower() != "nan"])

        mlb = MultiLabelBinarizer()
        all_labels = (
            df_train[LABEL_COL].dropna().tolist() +
            df_val[LABEL_COL].dropna().tolist() +
            df_test[LABEL_COL].dropna().tolist()
        )
        mlb.fit(all_labels)

    else:
        raise ValueError("Unknown TASK specified.")

    y_train = mlb.transform(df_train[LABEL_COL])
    y_val   = mlb.transform(df_val[LABEL_COL])
    y_test  = mlb.transform(df_test[LABEL_COL])

    return df_train, df_val, df_test, y_train, y_val, y_test, mlb, TEXT_COL, LABEL_COL



def prepare_data_STL_coarse(TASK, train_domains=["UA", "CC"], test_domains=["UA", "CC"], train_languages=["ALL"], debug=False):
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

    filtered_articles = filtered_articles.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(filtered_articles))
    train_articles = filtered_articles.iloc[:split_idx].copy()
    val_articles = filtered_articles.iloc[split_idx:].copy()

    if debug:
        train_articles = train_articles.sample(100)
        val_articles = val_articles.sample(100)
        test_s1_articles = test_s1_articles.sample(100)
        test_s2_articles = test_s2_articles.sample(100)

    if TASK == "narrative_classification":
        df_train = pd.merge(train_articles, s2, on="Filename")
        df_val = pd.merge(val_articles, s2, on="Filename")
        df_test = pd.merge(test_s2_articles, test_s2_labels, on="Filename")

        TEXT_COL = "Translated_Text"
        LABEL_COL = "Narrative"

        def parse_main_labels(x):
            if isinstance(x, str):
                return [label.strip() for label in x.split(";") if label.strip().lower() != "nan"]
            return []

        for df in [df_train, df_val, df_test]:
            df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
            df[LABEL_COL] = df[LABEL_COL].apply(parse_main_labels)

    elif TASK == "entity_framing":
        ENTITY_PARENT_MAP = {
            "Guardian": "Protagonist", "Martyr": "Protagonist", "Peacemaker": "Protagonist",
            "Rebel": "Protagonist", "Underdog": "Protagonist", "Virtuous": "Protagonist",
            "Instigator": "Antagonist", "Conspirator": "Antagonist", "Tyrant": "Antagonist",
            "Foreign Adversary": "Antagonist", "Traitor": "Antagonist", "Spy": "Antagonist",
            "Saboteur": "Antagonist", "Corrupt": "Antagonist", "Incompetent": "Antagonist",
            "Terrorist": "Antagonist", "Deceiver": "Antagonist", "Bigot": "Antagonist",
            "Forgotten": "Innocent", "Exploited": "Innocent", "Victim": "Innocent", "Scapegoat": "Innocent"
        }

        df_train = pd.merge(s1, train_articles, on="Filename")
        df_val = pd.merge(s1, val_articles, on="Filename")
        df_test = pd.merge(test_s1_labels, test_s1_articles, on="Filename")

        TEXT_COL = "Input_Text"
        LABEL_COL = "Label"

        def insert_entity_marker(text, start, end):
            try:
                start, end = int(start), int(end)
                return text[:start] + "[ENTITY]" + text[start:end] + "[/ENTITY]" + text[end:]
            except:
                return text

        for df in [df_train, df_val, df_test]:
            df.dropna(subset=["Translated_Text", "Entity", LABEL_COL, "Start", "End"], inplace=True)
            df["Start"] = df["Start"].astype(int)
            df["End"] = df["End"].astype(int)
            df["Input_Text"] = df.apply(lambda row: insert_entity_marker(row["Translated_Text"], row["Start"], row["End"]), axis=1)
            df[LABEL_COL] = df[LABEL_COL].apply(
                lambda x: [ENTITY_PARENT_MAP.get(label.strip(), label.strip()) for label in str(x).split(",") if label.strip().lower() != "nan"]
            )
            df[LABEL_COL] = df[LABEL_COL].apply(lambda labels: list(set(labels)))

    else:
        raise ValueError("Unknown TASK specified.")

    mlb = MultiLabelBinarizer()
    all_labels = (
        df_train[LABEL_COL].dropna().tolist() +
        df_val[LABEL_COL].dropna().tolist() +
        df_test[LABEL_COL].dropna().tolist()
    )
    mlb.fit(all_labels)

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

    filtered_articles = filtered_articles.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(filtered_articles))
    train_articles = filtered_articles.iloc[:split_idx].copy()
    val_articles = filtered_articles.iloc[split_idx:].copy()

    if debug:
        train_articles = train_articles.sample(100)
        val_articles = val_articles.sample(100)
        test_s1_articles = test_s1_articles.sample(100)
        test_s2_articles = test_s2_articles.sample(100)

    if TASK == "narrative_classification":
        df_train = pd.merge(train_articles, s2, on="Filename")
        df_val   = pd.merge(val_articles, s2, on="Filename")
        df_test  = pd.merge(test_s2_articles, test_s2_labels, on="Filename")

        TEXT_COL = "Translated_Text"
        LABEL_COL = "Subnarrative"

        def parse_labels(x):
            if isinstance(x, str):
                return [label.strip() for label in x.split(";") if label.strip().lower() != "nan"]
            return []

        for df in [df_train, df_val, df_test]:
            df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
            df[LABEL_COL] = df[LABEL_COL].apply(parse_labels)

        mlb = MultiLabelBinarizer()
        all_labels = df_train[LABEL_COL].tolist() + df_val[LABEL_COL].tolist() + df_test[LABEL_COL].tolist()
        mlb.fit(all_labels)

        y_train = mlb.transform(df_train[LABEL_COL])
        y_val   = mlb.transform(df_val[LABEL_COL])
        y_test  = mlb.transform(df_test[LABEL_COL])

        label_to_index = {label: i for i, label in enumerate(mlb.classes_)}

        # Create child_to_parent map
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

        return df_train, df_val, df_test, y_train, y_val, y_test, mlb, TEXT_COL, LABEL_COL, child_to_parent, label_to_index

    elif TASK == "entity_framing":
        df_train = pd.merge(s1, train_articles, on="Filename")
        df_val   = pd.merge(s1, val_articles, on="Filename")
        df_test  = pd.merge(test_s1_labels, test_s1_articles, on="Filename")

        TEXT_COL = "Translated_Text"
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

        for df in [df_train, df_val, df_test]:
            df.dropna(subset=[TEXT_COL, "Entity", LABEL_COL, "Start", "End"], inplace=True)
            df["Start"] = df["Start"].astype(int)
            df["End"] = df["End"].astype(int)
            df["Input_Text"] = df.apply(lambda row: insert_entity_marker(row[TEXT_COL], row["Start"], row["End"]), axis=1)
            df[LABEL_COL] = df[LABEL_COL].apply(
                lambda x: [label.strip() for label in str(x).split(",") if label.strip().lower() != "nan"]
            )
            df[LABEL_COL] = df[LABEL_COL].apply(lambda x: [ENTITY_PARENT_MAP.get(label, label) for label in x])
            df[LABEL_COL] = df[LABEL_COL].apply(lambda x: list(set(x)))

        mlb = MultiLabelBinarizer()
        all_labels = df_train[LABEL_COL].tolist() + df_val[LABEL_COL].tolist() + df_test[LABEL_COL].tolist()
        mlb.fit(all_labels)

        y_train = mlb.transform(df_train[LABEL_COL])
        y_val   = mlb.transform(df_val[LABEL_COL])
        y_test  = mlb.transform(df_test[LABEL_COL])

        label_to_index = {label: i for i, label in enumerate(mlb.classes_)}

        # Reverse map from ENTITY_PARENT_MAP
        parent_to_children = {}
        for child, parent in ENTITY_PARENT_MAP.items():
            parent_to_children.setdefault(parent, []).append(child)

        child_to_parent = {child: parent for parent, children in parent_to_children.items() for child in children}

        return df_train, df_val, df_test, y_train, y_val, y_test, mlb, "Input_Text", LABEL_COL, child_to_parent, label_to_index

    else:
        raise NotImplementedError("Unknown or unsupported TASK.")