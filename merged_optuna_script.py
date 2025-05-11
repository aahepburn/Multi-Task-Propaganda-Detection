import optuna
import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from single_task import TransformerClassifier, train_single_task_model, MultiLabelDataset
from multi_task import (
    MultiTaskTransformer,
    AdapterMultiTaskTransformer,
    train_mtl_flat
)
from data_loader_STL import prepare_data_STL_fine
from data_loader_MTL import prepare_data_MTL_fine_flat
import evaluation_utils as eval_util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def suggest_hyperparameters(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "epochs": trial.suggest_int("epochs", 2, 4),
        "threshold": trial.suggest_float("threshold", 0.2, 0.5),
    }

def objective_stl(trial, task_type="narrative_classification", model_name="roberta-base"):
    params = suggest_hyperparameters(trial)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df_train, df_val, df_test, y_train, y_val, y_test, mlb, TEXT_COL, LABEL_COL = prepare_data_STL_fine(
        task_type, ["UA", "CC"], ["UA", "CC"]
    )

    train_dataset = MultiLabelDataset(df_train[TEXT_COL].tolist(), y_train, tokenizer, 512)
    val_dataset = MultiLabelDataset(df_val[TEXT_COL].tolist(), y_val, tokenizer, 512)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    model = TransformerClassifier(model_name, num_classes=len(mlb.classes_)).to(device)

    model = train_single_task_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_val=y_val,
        MODEL_PATH="optuna_tmp.pt",
        LEARNING_RATE=params["learning_rate"],
        EPOCHS=params["epochs"],
        device=device,
        predict_proba=eval_util.predict_proba,
        evaluate_threshold_sweep=eval_util.evaluate_threshold_sweep
    )

    results = eval_util.evaluate_per_domain_flat(
        model, val_loader, df_val, val_loader, df_val, mlb, device, threshold=params["threshold"]
    )

    return results["val"]["macro"]

def objective_mtl(trial, model_name="roberta-base"):
    params = suggest_hyperparameters(trial)

    (
        df_train_s1, df_val_s1, _, y_train_s1, y_val_s1, _, mlb_s1,
        df_train_s2, df_val_s2, _, y_train_s2, y_val_s2, _, mlb_s2,
        train_loader_s1, val_loader_s1, _,
        train_loader_s2, val_loader_s2, _,
        num_classes_dict
    ) = prepare_data_MTL_fine_flat(
        TASK="multi_task",
        model_name=model_name,
        max_len=512,
        batch_size=params["batch_size"],
        train_domains=["UA", "CC"],
        test_domains=["UA", "CC"],
        train_languages=["ALL"]
    )

    task_classes = {
        "narrative_classification": y_train_s2.shape[1],
        "entity_framing": y_train_s1.shape[1]
    }

    model = MultiTaskTransformer(model_name, task_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    train_mtl_flat(
        model=model,
        loaders={
            "narrative_classification": train_loader_s2,
            "entity_framing": train_loader_s1
        },
        val_data={
            "narrative_classification": (val_loader_s2, df_val_s2, y_val_s2, mlb_s2),
            "entity_framing": (val_loader_s1, df_val_s1, y_val_s1, mlb_s1)
        },
        mlbs={
            "narrative_classification": mlb_s2,
            "entity_framing": mlb_s1
        },
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=params["epochs"],
        train_domain=["UA", "CC"],
        test_domain=["UA", "CC"]
    )

    results = eval_util.evaluate_mtl_all_tasks(
        model=model,
        task_loaders={
            "narrative_classification": val_loader_s2,
            "entity_framing": val_loader_s1
        },
        task_dfs={
            "narrative_classification": df_val_s2,
            "entity_framing": df_val_s1
        },
        task_targets={
            "narrative_classification": y_val_s2,
            "entity_framing": y_val_s1
        },
        task_mlbs={
            "narrative_classification": mlb_s2,
            "entity_framing": mlb_s1
        },
        domain_list=["UA", "CC"],
        device=device,
        load_from_disk=False
    )

    f1s = [v["macro"] for v in results.values()]
    return sum(f1s) / len(f1s)

def objective_mtl_adapter(trial, model_name="roberta-base"):
    params = suggest_hyperparameters(trial)

    (
        df_train_s1, df_val_s1, _, y_train_s1, y_val_s1, _, mlb_s1,
        df_train_s2, df_val_s2, _, y_train_s2, y_val_s2, _, mlb_s2,
        train_loader_s1, val_loader_s1, _,
        train_loader_s2, val_loader_s2, _,
        num_classes_dict
    ) = prepare_data_MTL_fine_flat(
        TASK="multi_task_adapter",
        model_name=model_name,
        max_len=512,
        batch_size=params["batch_size"],
        train_domains=["UA", "CC"],
        test_domains=["UA", "CC"],
        train_languages=["ALL"]
    )

    model = AdapterMultiTaskTransformer(
        model_name=model_name,
        num_classes_dict=num_classes_dict,
        adapter_dim=128  # Fixed for basic finetuning
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    train_mtl_flat(
        model=model,
        loaders={
            "narrative_classification": train_loader_s2,
            "entity_framing": train_loader_s1
        },
        val_data={
            "narrative_classification": (val_loader_s2, df_val_s2, y_val_s2, mlb_s2),
            "entity_framing": (val_loader_s1, df_val_s1, y_val_s1, mlb_s1)
        },
        mlbs={
            "narrative_classification": mlb_s2,
            "entity_framing": mlb_s1
        },
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=params["epochs"],
        train_domain=["UA", "CC"],
        test_domain=["UA", "CC"]
    )

    results = eval_util.evaluate_mtl_all_tasks(
        model=model,
        task_loaders={
            "narrative_classification": val_loader_s2,
            "entity_framing": val_loader_s1
        },
        task_dfs={
            "narrative_classification": df_val_s2,
            "entity_framing": df_val_s1
        },
        task_targets={
            "narrative_classification": y_val_s2,
            "entity_framing": y_val_s1
        },
        task_mlbs={
            "narrative_classification": mlb_s2,
            "entity_framing": mlb_s1
        },
        domain_list=["UA", "CC"],
        device=device,
        load_from_disk=False
    )

    f1s = [v["macro"] for v in results.values()]
    return sum(f1s) / len(f1s)
