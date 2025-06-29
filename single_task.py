import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm



def predict_proba(model, loader, task, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask, task=task)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(probs)

    return np.array(preds)


# ==========================
# DATASET CLASS
# ==========================

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
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

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }



# ==========================
# MODEL CLASS
# ==========================

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, force_download=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)





def train_single_task_model(model,
                            train_loader,
                            val_loader,
                            y_val,
                            MODEL_PATH,
                            LEARNING_RATE=2e-5,
                            EPOCHS=3,
                            device=None,
                            predict_proba=None,
                            evaluate_threshold_sweep=None):
    """
    Trains a single-task multi-label classifier using BCEWithLogitsLoss.
    Preserves detailed logging and saves best model based on macro F1.

    Args:
        model (nn.Module): Transformer-based classifier.
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        y_val (np.array): Binary matrix of true labels for validation.
        MODEL_PATH (str): File path to save the best model.
        LEARNING_RATE (float): Learning rate for optimizer.
        EPOCHS (int): Number of training epochs.
        device (torch.device): CUDA or CPU device.
        predict_proba (function): Function to generate probabilities from model.
        evaluate_threshold_sweep (function): Function to find best threshold.

    Returns:
        nn.Module: Trained model (best epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_macro_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\nEpoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")

        # Validation
        val_probs = predict_proba(model, val_loader, device)
        threshold = evaluate_threshold_sweep(y_val, val_probs)
        y_val_pred = (val_probs > threshold).astype(int)
        macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        print(f"Validation Macro F1 (Epoch {epoch + 1}): {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model (Epoch {epoch + 1}) to {MODEL_PATH}")

    return model





