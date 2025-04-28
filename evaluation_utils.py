import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score,classification_report




# ==========================
# FIXED THRESHOLD EVALUATION
# ==========================

def evaluate_flat(loader, df_source, mlb, label="TEST", threshold=0.25):

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

    # filter columns where y_true or y_pred has no samples (i.e., unseen label)
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

def evaluate_per_domain_flat(val_loader, df_val, test_loader, df_test, mlb, threshold=0.25):
    """
        REQUIRES the 'evaluate' function

        Evaluates model performance separately on validation and test domains using a fixed threshold.

        Args:
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
    val_results = evaluate_flat(val_loader, df_val.reset_index(drop=True), mlb, label="VALIDATION", threshold=threshold)

    print("\n=========================")
    print("Test (Fixed Threshold)")
    print("=========================")
    test_results = evaluate_flat(test_loader, df_test.reset_index(drop=True), mlb, label="TEST", threshold=threshold)

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

def evaluate_per_class_flat(loader, df_source, mlb, label="TEST", threshold=0.25):
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
# FIXED THRESHOLD EVALUATION (HIERARCHY-AWARE)
# ==========================

def evaluate_hierarchy(loader, df_source, mlb, label="TEST", threshold=0.25): 
    model.eval()
    y_true, y_pred, domains = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Evaluating " + label)):
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

    #  Enforce hierarchy before thresholding
    y_pred_tensor = torch.tensor(y_pred).float()
    y_pred_tensor = apply_hierarchical_constraints(y_pred_tensor, child_to_parent, label_to_index)
    y_pred = y_pred_tensor.cpu().numpy()

    y_pred_bin = (y_pred > threshold).astype(int)

    #  Filter out completely unseen labels
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
        "threshold": threshold,
        "labels_used": filtered_labels.tolist()
    }


def evaluate_per_domain_hierarchy(val_loader, df_val, test_loader, df_test, mlb, threshold=0.25):
    print("\n=========================")
    print("Validation (Fixed Threshold)")
    print("=========================")
    val_results = evaluate_hierarchy(val_loader, df_val.reset_index(drop=True), mlb, label="VALIDATION", threshold=threshold)

    print("\n=========================")
    print("Test (Fixed Threshold)")
    print("=========================")
    test_results = evaluate_hierarchy(test_loader, df_test.reset_index(drop=True), mlb, label="TEST", threshold=threshold)

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

def evaluate_mtl_task(model, loader, df_source, y_true, mlb, task, label="TEST", threshold=0.25, device=None):
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



def evaluate_mtl_hierarchical_task(model, test_loader, df_test, y_test, mlb, task_name, child_to_parent, label_to_index, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask, task=task_name)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_pred.extend(probs)

    y_pred = np.array(y_pred)
    y_pred_bin = (y_pred > 0.25).astype(float)
    
    # Apply hierarchical constraints
    y_pred_bin = apply_hierarchical_constraints_mtl(y_pred_bin, child_to_parent, label_to_index).astype(int)

    print(f"\n[Hierarchical Evaluation] Task: {task_name}")
    macro_f1 = f1_score(y_test, y_pred_bin, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}")

    # ===============================
    # PER DOMAIN EVALUATION
    # ===============================
    print("\n[Per-Domain Macro F1]")
    for domain in df_test["Domain"].unique():
        idxs = df_test[df_test["Domain"] == domain].index
        macro_f1_dom = f1_score(y_test[idxs], y_pred_bin[idxs], average="macro", zero_division=0)
        print(f"{domain}: {macro_f1_dom:.4f}")

    # ===============================
    # PER CLASS EVALUATION
    # ===============================
    print("\n[Per-Class F1 Score]")
    report = classification_report(y_test, y_pred_bin, target_names=mlb.classes_, zero_division=0, output_dict=False)
    print(report)



def evaluate_mtl_hierarchical_all_tasks(
    model,
    test_loaders,
    df_tests,
    y_tests,
    mlbs,
    child_to_parent_map,
    label_to_index_map,
    device
):
    print("\n========== MTL Hierarchical Evaluation ==========\n")
    for task in test_loaders:
        evaluate_mtl_hierarchical_task(
            model=model,
            test_loader=test_loaders[task],
            df_test=df_tests[task],
            y_test=y_tests[task],
            mlb=mlbs[task],
            task_name=task,
            child_to_parent=child_to_parent_map[task],
            label_to_index=label_to_index_map[task],
            device=device
        )


def apply_hierarchical_constraints_mtl(preds, child_to_parent, label_to_index):
    """
    preds = tensor of shape (batch_size, num_classes), after sigmoid
    child_to_parent = dict[str, str]
    label_to_index = dict[str, int]
    """
    for child, parent in child_to_parent.items():
        if child in label_to_index and parent in label_to_index:
            child_idx = label_to_index[child]
            parent_idx = label_to_index[parent]
            preds[:, child_idx] *= preds[:, parent_idx] >= 0.5
    return preds

