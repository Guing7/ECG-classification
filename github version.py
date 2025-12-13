import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from plot_best_curve import plot_best_curves

# 1. prepare_lead_folder
def prepare_lead_folder(lead_idx):
    folder = f"results/lead{lead_idx}"
    os.makedirs(folder, exist_ok=True)
    return folder

# 2. lead_idx and save_dir
lead_idx = 1
save_dir = prepare_lead_folder(lead_idx)
META_COLS = [
    "PatientAge", "Gender", "VentricularRate", "AtrialRate",
    "QRSDuration", "QTInterval", "QTCorrected", "RAxis", "TAxis",
    "QRSCount", "QOnset", "QOffset", "TOffset"
]

MERGE_MAP = {
    "AF": ["AF"], "SB": ["SB"], "SVT": ["SVT"],
    "ST": ["ST"], "SR": ["SR"], "AFIB": ["AFIB"], "SI": ["SA"]
}
RHYTHM_TO_GROUP = {r: g for g, rs in MERGE_MAP.items() for r in rs}
df = pd.read_excel("Diagnostics.xlsx")
classes = sorted(df["Rhythm"].unique())
label_map = {c: i for i, c in enumerate(classes)}

# 1. save config
def prepare_lead_folder(lead_idx):
    folder = f"results/lead{lead_idx}"
    os.makedirs(folder, exist_ok=True)
    return folder

# 2. random seed
def set_seed(seed=16):
    import os, random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
set_seed(seed=304)
# 3. define
class ECG_Fusion_Model(nn.Module):
    def __init__(self, n_classes=7, tab_feat_dim=13, hidden_size=128, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.compress = nn.Conv1d(1, 64, kernel_size=21, stride=11)
        self.cnn = nn.Sequential(
            nn.Conv1d(64, 64, 7), nn.LeakyReLU(0.01),
            nn.MaxPool1d(2), nn.BatchNorm1d(64), nn.Dropout(0.4),
            nn.Conv1d(64, 128, 5), nn.LeakyReLU(0.01),
            nn.Conv1d(128, 512, 7), nn.LeakyReLU(0.01),
            nn.MaxPool1d(2), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Conv1d(512, 256, 13), nn.LeakyReLU(0.01),
            nn.Conv1d(256, 256, 9), nn.LeakyReLU(0.01),
            nn.MaxPool1d(2), nn.BatchNorm1d(256), nn.Dropout(0.4),
        )
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.attn_query = nn.Linear(self.rnn_out_dim, self.rnn_out_dim)
        self.residual_proj = nn.Linear(self.rnn_out_dim, 256)
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_feat_dim, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.LeakyReLU(0.1), nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
        self.attn_proj = nn.Linear(256, 128)
        self.norm = nn.LayerNorm(128)
        self.attn_out_proj = nn.Linear(128, 256)
        self.attn_alpha = nn.Parameter(torch.tensor(1.0))
    def forward(self, wave, tab):
        x = self.compress(wave)  # [B, 64, L]
        x = self.cnn(x)  # [B, 256, T]   T=43
        x_t = x.transpose(1, 2)  # [B, T, 256]
        lstm_out, _ = self.lstm(x_t)
        # =========MaxPool========
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # [B, 256]
        # =========Attention=========
        lstm_t = lstm_out  # [B, T, D]
        lstm_t = self.attn_proj(lstm_t)  # [B, T, 128]
        lstm_t = self.norm(lstm_t)
        query = lstm_t.mean(dim=1)  # [B, 128]
        score = torch.matmul(lstm_t, query.unsqueeze(-1)).squeeze(-1)
        score = score / 8.0
        attn = F.softmax(score, dim=-1).unsqueeze(-1)  # [B, T, 1]
        attn_out = (attn * lstm_t).sum(dim=1)  # [B, 128]
        attn_out = self.attn_out_proj(attn_out)  # [B, 256]
        wave_feat = x_max + self.attn_alpha * attn_out
        tab_feat = self.tab_mlp(tab)
        final = torch.cat([wave_feat, tab_feat], dim=-1)
        return self.classifier(final)

# 4. load data
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

def load_ecg_with_meta(data_dir="ECGDataNPY", lead_idx=1, max_len=5000):
    df = pd.read_excel("Diagnostics.xlsx")
    clean_df = pd.read_excel("CleanECGFiles.xlsx")
    clean_files = set(clean_df["FileName"].tolist())
    df = df[df["FileName"].isin(clean_files)].reset_index(drop=True)
    print(f"📌 After merging clean list: {len(df)} samples remain")
    df["Rhythm"] = df["Rhythm"].map(RHYTHM_TO_GROUP)
    df = df.dropna(subset=["Rhythm"]).reset_index(drop=True)
    df["Gender"] = df["Gender"].astype(str).str.upper().replace({
        "MALE": 1, "M": 1, "FEMALE": 0, "F": 0
    })
    df["Gender"] = pd.to_numeric(df["Gender"], errors="ignore")
    classes = sorted(df["Rhythm"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    signals, metas, labels = [], [], []
    dropped = []
    print(f"\n Loading Clean Dataset | Lead {lead_idx}")
    print(f" Using CleanECGFiles.xlsx → {len(df)} samples")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row["FileName"]
        path = os.path.join(data_dir, fname + ".npy")
        if not os.path.exists(path):
            dropped.append(fname)
            continue
        arr = np.load(path)
        ecg = arr[:, lead_idx].astype(float)
        # Normalize single signal
        m, s = ecg.mean(), ecg.std()
        if s < 1e-6:
            dropped.append(fname)
            continue
        ecg = (ecg - m) / (s + 1e-6)
        # Pad / crop
        pad = np.zeros(max_len)
        L = min(max_len, len(ecg))
        pad[:L] = ecg[:L]
        signals.append(pad)
        metas.append(row[META_COLS].astype(float).values)
        labels.append(label_map[row["Rhythm"]])
    signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
    metas = torch.tensor(metas, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    # Normalize metadata
    metas = (metas - metas.mean(0)) / (metas.std(0) + 1e-6)
    print(f"✔ Loaded: {len(signals)}")
    print(f"⚠ Skipped missing / zero-std: {len(dropped)}")
    return signals, metas, labels, label_map
# 2. feature extract
def extract_wave_features(model, wave, tab):
    x = model.compress(wave)
    x = model.cnn(x)
    x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
    x_t = x.transpose(1, 2)
    lstm_out, _ = model.lstm(x_t)
    lstm_t = model.attn_proj(lstm_out)
    lstm_t = model.norm(lstm_t)
    query = lstm_t.mean(dim=1)
    score = (lstm_t @ query.unsqueeze(-1)).squeeze(-1)
    attn = F.softmax(score / 8.0, dim=-1).unsqueeze(-1)
    attn_out = (attn * lstm_t).sum(dim=1)
    attn_out = model.attn_out_proj(attn_out)
    wave_feat = x_max + model.attn_alpha * attn_out
    tab_feat = model.tab_mlp(tab)
    final_feat = torch.cat([wave_feat, tab_feat], dim=-1)
    return final_feat
# 3. DataLoader
signals, metas, labels, label_map = load_ecg_with_meta()
idx_train, idx_temp = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=labels[idx_temp], random_state=42)
train_meta = metas[idx_train]
meta_mean = train_meta.mean(0)
meta_std = train_meta.std(0)
metas = (metas - meta_mean) / (meta_std + 1e-6)
train_loader = DataLoader(TensorDataset(signals[idx_train], metas[idx_train], labels[idx_train]), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(signals[idx_val], metas[idx_val], labels[idx_val]), batch_size=128)
test_loader = DataLoader(TensorDataset(signals[idx_test], metas[idx_test], labels[idx_test]), batch_size=128)
# 4. model & optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECG_Fusion_Model(n_classes=7, tab_feat_dim=13).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
# 5. train
epochs = 100
result = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
best_val_acc = 0
for epoch in range(1, epochs + 1):
    # Train phase
    model.train()
    total_loss = total_correct = total_samples = 0
    for wave, tab, y in train_loader:
        wave, tab, y = wave.to(device), tab.to(device), y.to(device)
        logits = model(wave, tab)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(1)
        total_correct += (pred == y).sum().item()
        total_samples += y.size(0)
    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / total_samples
    # Validation phase
    model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for wave, tab, y in val_loader:
            wave, tab, y = wave.to(device), tab.to(device), y.to(device)
            logits = model(wave, tab)
            loss = F.cross_entropy(logits, y)
            val_loss += loss.item()
            pred = logits.argmax(1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    # Save history
    result["train_acc"].append(train_acc)
    result["val_acc"].append(val_acc)
    result["train_loss"].append(train_loss)
    result["val_loss"].append(val_loss)
    # Check for best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_fusion_model.pth")
    print(f"[Epoch {epoch}/{epochs}]"
          f" TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f}"
          f" TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f}")
# 6. load best model
model.load_state_dict(torch.load("best_fusion_model.pth"))
model.eval()
test_loss = 0
test_correct = 0
test_total = 0
all_feats = []
all_preds = []
all_labels = []
with torch.no_grad():
    for wave, tab, y in test_loader:
        wave, tab, y = wave.to(device), tab.to(device), y.to(device)
        feat = extract_wave_features(model, wave, tab)
        logits = model.classifier(feat)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits, y)
        test_loss += loss.item()
        test_correct += (pred == y).sum().item()
        test_total += y.size(0)
        all_feats.append(feat.cpu())
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
test_loss /= len(test_loader)
test_acc = test_correct / test_total
print(f"\nTest Acc = {test_acc:.4f}, Test Loss = {test_loss:.4f}")
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import os
all_feats = torch.cat(all_feats).cpu().numpy()
all_preds = torch.cat(all_preds).cpu().numpy()
all_labels = torch.cat(all_labels).cpu().numpy()
# ============================
# confusion_matrix
# ============================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(label_map.values()), yticklabels=list(label_map.values()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Test) - Acc={test_acc:.4f}")
plt.tight_layout()
confusion_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300)
plt.close()
# ============================
#  t-SNE
# ============================
tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
X_2d = tsne.fit_transform(all_feats)
plt.figure(figsize=(6, 6))
for i, cls in enumerate(label_map.values()):
    idx = (all_labels == i)
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=25, label=cls, alpha=0.75)
plt.legend(loc='lower right', frameon=True, framealpha=0.85, fontsize=10)
plt.title("t-SNE Visualization")
plt.tight_layout()
tsne_path = os.path.join(save_dir, "tsne.png")
plt.savefig(tsne_path, dpi=300)
plt.close()
# ============================
# metrics
# ============================
def compute_metrics(cm):
    metrics = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        Sens = TP / (TP + FN + 1e-6) * 100
        Prec = TP / (TP + FP + 1e-6) * 100
        Spec = TN / (TN + FP + 1e-6) * 100
        F1 = 2 * Sens * Prec / (Sens + Prec + 1e-6)
        Acc = (TP + TN) / cm.sum() * 100
        metrics.append([Sens, Prec, Spec, F1, Acc])
    return np.array(metrics)
metrics = compute_metrics(cm)
df_perf = pd.DataFrame(metrics, columns=["Sensitivity (%)", "Precision (%)", "Specificity (%)", "F1-Score (%)", "Accuracy (%)"])
df_perf.insert(0, "Class", list(label_map.values()))
overall = df_perf.iloc[:, 1:].mean()
overall["Class"] = "Overall"
df_perf = pd.concat([df_perf, overall.to_frame().T], ignore_index=True)
df_perf = df_perf.round(2)
performance_path = os.path.join(save_dir, "performance.xlsx")
df_perf.to_excel(performance_path, index=False)
# ============================
#         train curve 1
# ============================
df_result = pd.DataFrame({
    "train_acc": result["train_acc"],
    "val_acc": result["val_acc"],
    "train_loss": result["train_loss"],
    "val_loss": result["val_loss"],
})
result_path = os.path.join(save_dir, "result.xlsx")
df_result.to_excel(result_path, index=False)
import matplotlib.pyplot as plt
# ============================
#         train curve 2
# ============================
epochs = len(result["train_acc"])
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), result["train_acc"], label="Train Accuracy", color='b')
plt.plot(range(1, epochs+1), result["val_acc"], label="Validation Accuracy", color='g')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), result["train_loss"], label="Train Loss", color='r')
plt.plot(range(1, epochs+1), result["val_loss"], label="Validation Loss", color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
training_curve_path = os.path.join(save_dir, "training_curve.png")
plt.savefig(training_curve_path, dpi=300)
plt.close()
print(f"✔ saved → {training_curve_path}")

