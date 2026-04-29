import mne
import numpy as np
import torch
import matplotlib.pyplot as plt
import umap
from braindecode.models import REVE
from huggingface_hub import login  

# %%
login(token="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")

# %%
fpath = 'D:/OS(CURRENT)/data/music/exp2/20.03_g1/Tumyalis_clear.fif'
raw = mne.io.read_raw_fif(fpath, preload=True)
raw.resample(200) 

# %%
# raw_f = raw.filter(l_freq=15, h_freq=25) 

# %%
window_duration = 2
epochs = mne.make_fixed_length_epochs(raw, duration=window_duration, preload=True)
data = epochs.get_data()  

mean = data.mean(axis=2, keepdims=True)
std = data.std(axis=2, keepdims=True)
std[std == 0] = 1.0  

data_zscored = (data - mean) / std
data_ready = np.clip(data_zscored, -15, 15)
eeg_tensor = torch.tensor(data_ready, dtype=torch.float32)

# %%
print("Загрузка весов REVE...")
model = REVE.from_pretrained(
    "brain-bzh/reve-base",  
    n_outputs=512,         
    n_chans=raw.info['nchan'],
    n_times=eeg_tensor.shape[2],
    sfreq=raw.info['sfreq'],
    chs_info=[{"ch_name": ch} for ch in raw.ch_names],
    attention_pooling=True  
)
model.eval() 

pos = model.get_positions(raw.ch_names)

# %%
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

batch_size = 100
dataset = TensorDataset(eeg_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

all_embeddings = []

with torch.no_grad():
    for batch in loader:
        X_batch = batch[0].to(device)
        
        batch_pos = pos.expand(X_batch.shape[0], -1, -1).to(device)                
        embeddings = model(X_batch, pos=batch_pos)
        
        all_embeddings.append(embeddings.cpu().numpy())

features_flat = np.concatenate(all_embeddings, axis=0)

# %%
reducer = umap.UMAP(
    n_components=3, 
    metric='cosine', 
    n_neighbors=20, 
    min_dist=0.1
)
embeddings_3d_umap = reducer.fit_transform(features_flat)

# %%
n_epochs_per_label = embeddings_3d_umap.shape[0]/22
descriptions = raw.annotations.description

# Генерируем массив строк
true_labels_str = np.repeat(descriptions, n_epochs_per_label)

# Переводим в числа для раскраски графика
unique_labels = list(dict.fromkeys(descriptions)) 
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
true_labels_int = np.array([label_to_idx[lbl] for lbl in true_labels_str])
print(f"Сгенерировано меток: {len(true_labels_int)}")

# =====================================================================
# 8. 3D ВИЗУАЛИЗАЦИЯ ПО ЭКСПЕРИМЕНТАЛЬНЫМ СОСТОЯНИЯМ
# =====================================================================
fig = plt.figure(figsize=(14, 10))  
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    embeddings_3d_umap[:, 0], 
    embeddings_3d_umap[:, 1], 
    embeddings_3d_umap[:, 2], 
    c=true_labels_int, 
    cmap='turbo',      
    s=25,              
    alpha=0.8          
)

# Создаем легенду
handles = []
for i, lbl in enumerate(unique_labels):
    color = plt.cm.turbo(i / (len(unique_labels) - 1))
    handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8)
    handles.append(handle)

ax.legend(handles, unique_labels, title="Экспериментальные состояния", 
          bbox_to_anchor=(1.05, 1), loc='upper left')

ax.set_title('Проекция эмбеддингов REVE (с Attention Pooling) с учетом стимулов', fontsize=14)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

plt.tight_layout()
plt.show()

# %%

# %%

# %%

