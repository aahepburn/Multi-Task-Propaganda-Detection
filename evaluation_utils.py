import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score,classification_report

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score,classification_report




# ==========================
# FIXED THRESHOLD EVALUATION
# ==========================

def evaluate_flat(model, loader, df_source, mlb, device, label="TEST", threshold=0.35):

    """
    Evaluates a multi-label classification model using a fixed probability threshold.

    Args:
        loader (DataLoader): A PyTorch DataLoader yielding batches of tokenised input data
        df_source (pd.DataFrame): Source dataframe containing metadata for each example, including domain info.
        mlb (MultiLabelBinarizer): The fitted multi-label binarizer used for encoding and decoding labels.
        label (str, optional): Label for the dataset (e.g., 'TEST', 'VALIDATION'). Used for logging. Defaults to "TEST".
        threshold (float, optional): Probability threshold to convert predicted probabilities into binary labels. Defaults to 0.25.

    Returns:
        dict: A dictionary containing overall macro F1, micro F1, exact match score,
              the threshold used, and the list of labels used after filtering.
              Also prints per-domain breakdowns of these metrics.

    Notes:
        - Filters out labels that are completely unseen in both predictions and ground truths
          to avoid skewed metric calculations.
        - Performs evaluation on the entire dataset as well as broken down by domain.
    """
    model.eval()
    y_true, y_pred, domains = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Evaluating {label}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()

            y_pred.extend(probs)
            y_true.extend(labels)

            start = i * loader.batch_size
            end = start + len(labels)
            domains.extend(df_source["Domain"].iloc[start:end].tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    domains = np.array(domains)

    y_pred_bin = (y_pred > threshold).astype(int)

    # filter columns where y_true or y_pred has no samples
    mask = (y_true.sum(axis=0) + y_pred_bin.sum(axis=0)) > 0
    y_true = y_true[:, mask]
    y_pred_bin = y_pred_bin[:, mask]
    filtered_labels = np.array(mlb.classes_)[mask]

    macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    exact = (y_pred_bin == y_true).all(axis=1).mean()

    print(f"\n {label} (Fixed Threshold={threshold:.2f}):")
    print(f"Macro F1: {macro:.3f}")
    print(f"Micro F1: {micro:.3f}")
    print(f"Exact Match: {exact:.3f}")

    print("\n----------------------------")
    print("Per-Domain Breakdown")
    print("----------------------------")
    for domain in np.unique(domains):
        idx = np.where(domains == domain)[0]
        y_true_d = y_true[idx]
        y_pred_d = y_pred_bin[idx]

        macro_d = f1_score(y_true_d, y_pred_d, average="macro", zero_division=0)
        micro_d = f1_score(y_true_d, y_pred_d, average="micro", zero_division=0)
        exact_d = (y_pred_d == y_true_d).all(axis=1).mean()

        print(f"\n Domain: {domain}")
        print(f"Macro F1: {macro_d:.3f}")
        print(f"Micro F1: {micro_d:.3f}")
        print(f"Exact Match: {exact_d:.3f}")

    return {
        "macro": macro,
        "micro": micro,
        "exact": exact,
        "threshold": threshold,
        "labels_used": filtered_labels.tolist()
    }

# ----------------------------
# EVALUATE PER DOMAIN
def evaluate_per_domain_flat(model, val_loader, df_val, test_loader, df_test, mlb, device, threshold=0.35):
    """
    Evaluates model performance separately on validation and test domains using a fixed threshold.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        val_loader (DataLoader): PyTorch DataLoader for the validation set.
        df_val (pd.DataFrame): Original validation DataFrame containing domain info and labels.
        test_loader (DataLoader): PyTorch DataLoader for the test set.
        df_test (pd.DataFrame): Original test DataFrame containing domain info and labels.
        mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer used for decoding/encoding label space.
        threshold (float, optional): Probability threshold for binarizing predictions. Defaults to 0.25.

    Returns:
        dict: A dictionary with:
            - 'val': Evaluation metrics on the validation set
            - 'test': Evaluation metrics on the test set
            - 'ood_gap_macro': Difference in macro F1 between validation and test sets
    """
    print("\n=========================")
    print("Validation (Fixed Threshold)")
    print("=========================")
    val_results = evaluate_flat(model, val_loader, df_val.reset_index(drop=True), mlb, device, label="VALIDATION", threshold=threshold)

    print("\n=========================")
    print("Test (Fixed Threshold)")
    print("=========================")
    test_results = evaluate_flat(model, test_loader, df_test.reset_index(drop=True), mlb, device, label="TEST", threshold=threshold)

    print("\n=========================")
    print("OOD Generalization (Fixed Threshold)")
    print("=========================")
    macro_drop = val_results["macro"] - test_results["macro"]
    print(f"Δ Macro F1 (val - test): {macro_drop:.3f}")

    return {
        "val": val_results,
        "test": test_results,
        "ood_gap_macro": macro_drop
    }


# --------------------
# EVALUATE PER CLASS

def evaluate_per_class_flat(model, loader, df_source, mlb, device, label="TEST", threshold=0.35):

    """
        Evaluates a multi-label classification model and produces per-class and per-domain metrics.

        Args:
            loader (DataLoader): PyTorch DataLoader yielding batches of tokenized examples.
            df_source (pd.DataFrame): Source dataframe with original metadata, including domain info.
            mlb (MultiLabelBinarizer): Fitted binarizer used to decode/encode label vectors.
            label (str, optional): Label for this evaluation run (e.g. 'TEST', 'VALIDATION'). Defaults to "TEST".
            threshold (float, optional): Fixed probability threshold for converting predicted probabilities
                                         to binary outputs. Defaults to 0.25.
            model (torch.nn.Module, optional): Trained model used for prediction. Defaults to global `model`.

        Returns:
            dict: A dictionary with the following keys:
                - 'macro': Overall macro-averaged F1 score
                - 'micro': Overall micro-averaged F1 score
                - 'exact': Exact match ratio across all examples
                - 'threshold': Threshold used for binarization
                - 'labels_used': List of class names included after masking out unused labels
                - 'overall_report': Per-class precision, recall, F1 (all domains), as a dict
                - 'domain_reports': Dict of per-domain classification reports (as strings)
    """

    model.eval()
    y_true, y_pred, domains = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Evaluating {label}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()

            y_pred.extend(probs)
            y_true.extend(labels)

            start = i * loader.batch_size
            end = start + len(labels)
            domains.extend(df_source["Domain"].iloc[start:end].tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    domains = np.array(domains)

    y_pred_bin = (y_pred > threshold).astype(int)

    # Filter columns where y_true or y_pred has no samples
    mask = (y_true.sum(axis=0) + y_pred_bin.sum(axis=0)) > 0
    y_true = y_true[:, mask]
    y_pred_bin = y_pred_bin[:, mask]
    filtered_labels = np.array(mlb.classes_)[mask]

    macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    exact = (y_pred_bin == y_true).all(axis=1).mean()

    print(f"\n{label} (Fixed Threshold={threshold:.2f}):")
    print(f"Macro F1: {macro:.3f}")
    print(f"Micro F1: {micro:.3f}")
    print(f"Exact Match: {exact:.3f}")

    print("\n----------------------------")
    print("Classification Report (All Domains)")
    print("----------------------------")
    overall_report = classification_report(y_true, y_pred_bin, target_names=filtered_labels, zero_division=0, output_dict=False)
    print(overall_report)

    domain_reports = {}

    print("\n----------------------------")
    print("Per-Domain Breakdown")
    print("----------------------------")
    for domain in np.unique(domains):
        idx = np.where(domains == domain)[0]
        y_true_d = y_true[idx]
        y_pred_d = y_pred_bin[idx]

        macro_d = f1_score(y_true_d, y_pred_d, average="macro", zero_division=0)
        micro_d = f1_score(y_true_d, y_pred_d, average="micro", zero_division=0)
        exact_d = (y_pred_d == y_true_d).all(axis=1).mean()

        print(f"\nDomain: {domain}")
        print(f"Macro F1: {macro_d:.3f}")
        print(f"Micro F1: {micro_d:.3f}")
        print(f"Exact Match: {exact_d:.3f}")

        print("Classification Report:")
        report = classification_report(y_true_d, y_pred_d, target_names=filtered_labels, zero_division=0)
        print(report)
        domain_reports[domain] = report

    return {
        "macro": macro,
        "micro": micro,
        "exact": exact,
        "threshold": threshold,
        "labels_used": filtered_labels.tolist(),
        "overall_report": classification_report(y_true, y_pred_bin, target_names=filtered_labels, zero_division=0, output_dict=True),
        "domain_reports": domain_reports
    }




# ==========================
# UTILS
# ==========================

def predict_proba(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            probs.extend(torch.sigmoid(outputs).cpu().numpy())
    return np.array(probs)

def evaluate_threshold_sweep(y_true, y_pred, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_thresh = 0.5
    best_f1 = 0
    results = []

    for thresh in thresholds:
        y_pred_bin = (y_pred > thresh).astype(int)
        macro = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
        micro = f1_score(y_true, y_pred_bin, average='micro', zero_division=0)
        exact = (y_pred_bin == y_true).all(axis=1).mean()

        results.append((thresh, macro, micro, exact))
        if macro > best_f1:
            best_f1 = macro
            best_thresh = thresh

    print("Threshold sweep results:")
    for t, macro, micro, exact in results:
        print(f"Thresh {t:.2f} | Macro F1: {macro:.3f} | Micro F1: {micro:.3f} | Exact Match: {exact:.3f}")

    print(f"\n Best threshold = {best_thresh:.2f} with Macro F1 = {best_f1:.3f}")
    return best_thresh


#### MTL

from sklearn.metrics import f1_score
import torch
import numpy as np
from tqdm import tqdm

def evaluate_mtl_task(model, loader, df_source, y_true, mlb, task, label="TEST", threshold=0.35, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    y_pred, domains = [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Evaluating {label} [{task}]")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask, task=task)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_pred.extend(probs)

            start = i * loader.batch_size
            end = start + len(probs)
            domains.extend(df_source["Domain"].iloc[start:end].tolist())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    domains = np.array(domains)

    y_pred_bin = (y_pred > threshold).astype(int)

    if task == "narrative_classification":
        mask = (y_true.sum(axis=0) + y_pred_bin.sum(axis=0)) > 0
        y_true = y_true[:, mask]
        y_pred_bin = y_pred_bin[:, mask]
        filtered_labels = np.array(mlb.classes_)[mask]
    else:
        filtered_labels = mlb.classes_

    macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    exact = (y_pred_bin == y_true).all(axis=1).mean()

    print(f"\n{label} ({task}) [Threshold={threshold:.2f}]")
    print(f"Macro F1: {macro:.3f}")
    print(f"Micro F1: {micro:.3f}")
    print(f"Exact Match: {exact:.3f}")

    print("\n----------------------------")
    print("Per-Domain Breakdown")
    print("----------------------------")
    for domain in np.unique(domains):
        idx = np.where(domains == domain)[0]
        y_true_d = y_true[idx]
        y_pred_d = y_pred_bin[idx]

        macro_d = f1_score(y_true_d, y_pred_d, average="macro", zero_division=0)
        micro_d = f1_score(y_true_d, y_pred_d, average="micro", zero_division=0)
        exact_d = (y_pred_d == y_true_d).all(axis=1).mean()

        print(f"\nDomain: {domain}")
        print(f"Macro F1: {macro_d:.3f}")
        print(f"Micro F1: {micro_d:.3f}")
        print(f"Exact Match: {exact_d:.3f}")

    return {
        "macro": macro,
        "micro": micro,
        "exact": exact,
        "labels_used": filtered_labels.tolist()
    }


def evaluate_mtl_all_tasks(
    model,
    task_loaders,
    task_dfs,
    task_targets,
    task_mlbs,
    domain_list,
    device,
    load_from_disk=False  # <- default to False
):
    all_results = {}
    for task in task_loaders:
        print(f"\n--- Task: {task.upper()} ---")

        if load_from_disk:
            model_path = f"{task}_MTL_{'-'.join(domain_list)}_to_{'-'.join(domain_list)}.pt"
            model.load_state_dict(torch.load(model_path))
            model.to(device)

        results = evaluate_mtl_task(
            model=model,
            loader=task_loaders[task],
            df_source=task_dfs[task],
            y_true=task_targets[task],
            mlb=task_mlbs[task],
            task=task,
            device=device
        )
        all_results[task] = results
    return all_results





def evaluate_flat_custom(model, loader, df_source, mlb, device, label="TEST", threshold=0.5, task=None):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_bin = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch[f"{task}_labels"].cpu().numpy()

            #  Call model differently depending on whether it's MTL or STL
            if task is not None:
                outputs = model(input_ids, attention_mask, task=task)
            else:
                outputs = model(input_ids, attention_mask)

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds_bin = (probs >= threshold).astype(int)

            y_true.extend(labels)
            y_pred.extend(probs)
            y_pred_bin.extend(preds_bin)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_bin = np.array(y_pred_bin)

    exact_match = (y_true == y_pred_bin).all(axis=1).mean()

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_bin": y_pred_bin,
        "exact": exact_match
    }


from collections import defaultdict

def compute_coarse_from_fine_report(y_true, y_pred, mlb_classes, taxonomy_map):
    report = classification_report(y_true, y_pred, target_names=mlb_classes, output_dict=True, zero_division=0)

    coarse_stats = defaultdict(lambda: {"precision": [], "recall": [], "f1": [], "support": 0})

    for fine_label, coarse_label in taxonomy_map.items():
        if fine_label not in report:
            continue
        stats = report[fine_label]
        coarse_stats[coarse_label]["precision"].append(stats["precision"])
        coarse_stats[coarse_label]["recall"].append(stats["recall"])
        coarse_stats[coarse_label]["f1"].append(stats["f1-score"])
        coarse_stats[coarse_label]["support"] += stats["support"]

    coarse_results = {}
    f1s = []
    for c, stats in coarse_stats.items():
        p = np.mean(stats["precision"]) if stats["precision"] else 0
        r = np.mean(stats["recall"]) if stats["recall"] else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        coarse_results[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": stats["support"]}
        f1s.append(f1)

    macro_coarse_f1 = np.mean(f1s)
    return coarse_results, round(macro_coarse_f1, 4)



def compute_fine_vs_coarse_metrics(y_true, y_pred, label_list, coarse_label_list):
    """
    Computes macro/micro F1 for all labels and for a coarse subset.

    Args:
        y_true: binary np.ndarray
        y_pred: binary np.ndarray
        label_list: list of all label names (mlb.classes_)
        coarse_label_list: list of labels considered 'coarse'

    Returns:
        dict with fine and coarse F1s
    """
    # Fine-grained
    macro_fine = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_fine = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # Coarse-grained (subset selection)
    indices = [i for i, label in enumerate(label_list) if label in coarse_label_list]
    if not indices:
        raise ValueError("No overlapping coarse labels found in label list")

    y_true_coarse = y_true[:, indices]
    y_pred_coarse = y_pred[:, indices]

    macro_coarse = f1_score(y_true_coarse, y_pred_coarse, average="macro", zero_division=0)
    micro_coarse = f1_score(y_true_coarse, y_pred_coarse, average="micro", zero_division=0)

    return {
        "macro_fine": macro_fine,
        "micro_fine": micro_fine,
        "macro_coarse": macro_coarse,
        "micro_coarse": micro_coarse
    }



# Coarse label groups for each task
def get_coarse_label_list(task):
    if task == "entity_framing":
        return [
            "Protagonist",
            "Antagonist",
            "Innocent"
        ]
    
    elif task == "narrative_classification":
        return [
            "URW: Blaming the war on others rather than the invader",
            "URW: Discrediting Ukraine",
            "URW: Russia is the Victim",
            "URW: Praise of Russia",
            "URW: Overpraising the West",
            "URW: Speculating war outcomes",
            "URW: Discrediting the West, Diplomacy",
            "URW: Negative Consequences for the West",
            "URW: Distrust towards Media",
            "URW: Amplifying war-related fears",
            "URW: Hidden plots by secret schemes of powerful groups",

            "CC: Criticism of climate policies",
            "CC: Criticism of institutions and authorities",
            "CC: Climate change is beneficial",
            "CC: Downplaying climate change",
            "CC: Questioning the measurements and science",
            "CC: Criticism of climate movement",
            "CC: Controversy about green technologies",
            "CC: Hidden plots by secret schemes of powerful groups",
            "CC: Amplifying Climate Fears",
            "CC: Green policies are geopolitical instruments"
        ]
    
    else:
        raise ValueError(f"Unknown task: {task}")

