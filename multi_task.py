import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from itertools import cycle


# ==========================
# MODEL CLASS
# ==========================

class MultiTaskTransformer(nn.Module):
    def __init__(self, model_name, num_classes_dict):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        hidden_size = self.encoder.config.hidden_size

        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            "narrative_classification": nn.Linear(hidden_size, num_classes_dict["narrative_classification"]),
            "entity_framing": nn.Linear(hidden_size, num_classes_dict["entity_framing"])
        })

    def forward(self, input_ids, attention_mask, task):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        return self.task_heads[task](pooled_output)

# ==========================
# TRAINING LOOP
# ==========================

def train_mtl_flat(model, loaders, val_data, mlbs, optimizer, criterion, device, epochs, train_domain, test_domain):
    from itertools import cycle
    from sklearn.metrics import f1_score
    import numpy as np
    import torch

    best_macro_f1 = {task: 0.0 for task in loaders.keys()}
    task_names = list(loaders.keys())
    max_len = max(len(loader) for loader in loaders.values())
    iters = {task: cycle(loaders[task]) for task in task_names}

    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}...")
        model.train()
        total_loss = 0.0

        for _ in range(max_len):
            optimizer.zero_grad()
            loss = 0.0

            for task in task_names:
                batch = next(iters[task])
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch[f"{task}_labels"].to(device)

                outputs = model(input_ids, attention_mask, task=task)
                task_loss = criterion(outputs, labels)
                loss += task_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\nEpoch {epoch + 1} - Average Loss: {total_loss / max_len:.4f}")

        model.eval()
        for task in task_names:
            val_loader, df_val, y_val, mlb = val_data[task]
            print(f"\nValidating task: {task}")
            y_pred = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids, attention_mask, task=task)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    y_pred.extend(probs)

            y_pred = np.array(y_pred)
            y_pred_bin = (y_pred > 0.25).astype(int)
            macro = f1_score(y_val, y_pred_bin, average="macro", zero_division=0)
            print(f"[{task}] Macro F1: {macro:.4f}")

            if macro > best_macro_f1[task]:
                best_macro_f1[task] = macro
                save_path = f"{task}_MTL_{'-'.join(train_domain)}_to_{'-'.join(test_domain)}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"Best model for task '{task}' saved to {save_path}")



def train_mtl_hierarchical(model, loaders, val_data, child_to_parent_map, label_to_index_map,
                            optimizer, criterion, device, epochs, train_domain, test_domain):
    from itertools import cycle
    best_macro_f1 = {task: 0.0 for task in loaders.keys()}
    max_len = max(len(loader) for loader in loaders.values())
    task_names = list(loaders.keys())

    for epoch in range(epochs):
        print(f"\n Starting Epoch {epoch + 1}/{epochs}...") 
        model.train()
        total_loss = 0.0
        iters = {task: cycle(loaders[task]) for task in task_names}

        for _ in range(max_len):
            optimizer.zero_grad()
            loss = 0.0

            for task in task_names:
                batch = next(iters[task])
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch[f"{task}_labels"].to(device)

                outputs = model(input_ids, attention_mask, task=task)
                task_loss = hierarchical_loss_mtl(
                    outputs,
                    labels,
                    child_to_parent_map[task],
                    label_to_index_map[task]
                )

                loss += task_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max_len
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        model.eval()
        for task in val_data:
            val_loader, df_val, y_val, mlb = val_data[task]
            threshold = 0.25
            y_pred = predict_proba(model, val_loader, task, device)
            y_pred_bin = (y_pred > threshold).astype(float)
            y_pred_bin = apply_hierarchical_constraints_mtl(y_pred_bin, child_to_parent_map[task], label_to_index_map[task])
            y_pred_bin = y_pred_bin.astype(int)

            macro = f1_score(y_val, y_pred_bin, average="macro", zero_division=0)
            print(f"[{task}] Macro F1: {macro:.4f}")

            if macro > best_macro_f1[task]:
                best_macro_f1[task] = macro
                save_path = f"{task}_MTL_{'-'.join(train_domain)}_to_{'-'.join(test_domain)}.pt"
                torch.save(model.state_dict(), save_path)
                print(f" Best model for task '{task}' saved to {save_path}")



def hierarchical_loss_mtl(logits, targets, child_to_parent, label_to_index):
    """
    logits, targets = torch tensors of shape (batch_size, num_classes)
    child_to_parent = dict[str, str]
    label_to_index = dict[str, int]
    """
    bce = nn.BCEWithLogitsLoss(reduction='none')
    base_loss = bce(logits, targets)

    # Mask children losses where the parent is not active
    for child, parent in child_to_parent.items():
        if child in label_to_index and parent in label_to_index:
            child_idx = label_to_index[child]
            parent_idx = label_to_index[parent]
            mask = targets[:, parent_idx]
            base_loss[:, child_idx] *= mask  # only keep loss where parent is present

    return base_loss.mean()


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


def predict_proba(model, loader, task, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask, task=task)
            probs.extend(torch.sigmoid(outputs).cpu().numpy())
    return np.array(probs)



# MTL ADAPTER


class TaskAdapter(nn.Module):
    """PAL-like adapter: small bottleneck MLP acting as residual to the encoder output."""
    def __init__(self, hidden_size, bottleneck_dim=128):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)

    def forward(self, x):
        return self.up_project(self.activation(self.down_project(x)))


class AdapterMultiTaskTransformer(nn.Module):
    def __init__(self, model_name, num_classes_dict, adapter_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)

        # Task adapters (PALs)
        self.adapters = nn.ModuleDict({
            "narrative_classification": TaskAdapter(self.hidden_size, adapter_dim),
            "entity_framing": TaskAdapter(self.hidden_size, adapter_dim)
        })

        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            "narrative_classification": nn.Linear(self.hidden_size, num_classes_dict["narrative_classification"]),
            "entity_framing": nn.Linear(self.hidden_size, num_classes_dict["entity_framing"])
        })

    def forward(self, input_ids, attention_mask, task):
        # BERT encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

        # Add residual adapter
        adapter_output = self.adapters[task](pooled_output)
        adapted_output = pooled_output + adapter_output

        adapted_output = self.dropout(adapted_output)
        return self.task_heads[task](adapted_output)







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


