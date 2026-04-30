import os
import mne
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from braindecode.models import REVE
from transformers import AutoModel
from tqdm import tqdm  

# %% 1. Инициализация моделей
my_token = "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"

pos_bank_hf = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, token=my_token)
valid_positions = set(pos_bank_hf.position_names)

fpath = f'REVE_aos/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R01.edf'
raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
mne.rename_channels(raw.info, {ch: ch.replace('.', '').strip().upper() for ch in raw.ch_names})

channels_to_drop = [ch for ch in raw.ch_names if ch not in valid_positions]
raw.drop_channels(channels_to_drop)

print("Загрузка REVE через Braindecode...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = REVE.from_pretrained(
    "brain-bzh/reve-base", 
    n_outputs=2,
    token=my_token,
    n_times=4 * 200,
    attention_pooling=False,
    chs_info=[{"ch_name": ch} for ch in raw.ch_names],
).eval().to(device)

model.final_layer = torch.nn.Identity()

# %% 2. Извлечение признаков со всех испытуемых
runs = ['04', '06', '08', '10', '12', '14'] 
batch_size = 32 

all_features = []
all_labels = []
all_subjects = []

print("Начинаем обработку базы PhysioNet...")
# Оборачиваем в tqdm для визуализации прогресса
for sid in tqdm(range(1, 110), desc="Обработка испытуемых"):
    raws = []
    skip_subject = False
    
    for r in runs:
        fpath = f'REVE_aos/MNE-eegbci-data/files/eegmmidb/1.0.0/S{sid:03d}/S{sid:03d}R{r}.edf'
        if not os.path.exists(fpath):
            skip_subject = True; break
            
        raw_tmp = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        mne.rename_channels(raw_tmp.info, {ch: ch.replace('.', '').strip().upper() for ch in raw_tmp.ch_names})
        if len(raw.info["bads"]) > 0:
            raw.interpolate_bads()
        mapping = {'T1': 'Left', 'T2': 'Right'} if r in ['04', '08', '12'] else {'T1': 'BothFists', 'T2': 'BothFeet'}
        raw_tmp.annotations.rename(mapping)
        
        # Применяем band-pass фильтр: от 0.5 Гц до 64 Гц (специфика REVE для PhysioNet)
        if raw_tmp.info['lowpass'] > 64:
            raw_tmp.filter(l_freq=0.5, h_freq=64.0, verbose=False)
        else:
            raw_tmp.filter(l_freq=0.5, h_freq=None, verbose=False)
            
        raws.append(raw_tmp)
            
    if skip_subject or not raws: continue

    # 1. Сначала склеиваем данные ВСЕЙ сессии
    raw = mne.concatenate_raws(raws)
    
    # 2. Выполняем ресемплинг до 200 Гц ПОСЛЕ применения фильтра 64 Гц
    raw.resample(200, verbose=False)
    raw.set_eeg_reference(ref_channels='average', verbose=False)
    
    channels_to_drop = [ch for ch in raw.ch_names if ch not in valid_positions]
    raw.drop_channels(channels_to_drop)

    # 3. Применяем Z-нормализацию и клиппинг ко всей сессии испытуемого сразу
    def z_score_and_clip(x):
        mean = np.mean(x)
        std = np.std(x)
        return np.clip((x - mean) / (std + 1e-6), -15.0, 15.0)

    raw.apply_function(z_score_and_clip, channel_wise=True)
        
    channels_to_drop = [ch for ch in raw.ch_names if ch not in valid_positions]
    raw.drop_channels(channels_to_drop)
    
    positions = model.get_positions(raw.ch_names).to(device)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    target_events = {k: v for k, v in event_dict.items() if k in ['Left', 'Right', 'BothFists', 'BothFeet']}
    if not target_events: continue
        
    tmax = 4.0 - (1 / raw.info['sfreq'])
    epochs = mne.Epochs(raw, events, event_id=target_events, tmin=0, tmax=tmax, baseline=None, preload=True, verbose=False)
    
    data_tensor = torch.tensor(epochs.get_data(copy=True), dtype=torch.float32)
    
    class_map = {'Left': 0, 'Right': 1, 'BothFists': 2, 'BothFeet': 3}
    inv_event_dict = {v: k for k, v in target_events.items()}
    subj_labels = np.array([class_map[inv_event_dict[ev[-1]]] for ev in epochs.events])
    
    subj_features = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            X_batch = data_tensor[i:i+batch_size].to(device)
            
            current_pos = positions.unsqueeze(0).expand(X_batch.shape[0], -1, -1)
            raw_features = model(X_batch, pos=current_pos)
            
            # Возвращаем 3D структуру: (Batch, 248 токенов, 512 размерность)
            tokens_3d = raw_features.view(X_batch.shape[0], -1, 512)
            
            # Сохраняем сырые 3D тензоры
            subj_features.append(tokens_3d.cpu().numpy())
            
    # Аккуратно склеиваем батчи для одного испытуемого
    # np.concatenate вдоль нулевой оси (Batch)
    all_features.append(np.concatenate(subj_features, axis=0))
    all_labels.extend(subj_labels)
    all_subjects.extend([sid] * len(subj_labels))
        
# %% 3. Сборка, токенизированный PCA и UMAP
# 1. Собираем единый 3D тензор всех данных
X_tokens_3d = np.concatenate(all_features, axis=0) 
y_labels = np.array(all_labels)
y_subjects = np.array(all_subjects)

N_epochs, N_tokens, D_emb = X_tokens_3d.shape
print(f"\nСбор данных завершен! Форма всех токенов: {X_tokens_3d.shape}")

# 2. Применяем PCA к каждому токену индивидуально (сжатие 512 -> 50)
print("Применение PCA на уровне токенов (сжатие признаков 512 -> 50)...")
# Вытягиваем в 2D, чтобы PCA обработал все токены как независимые объекты
X_flat_for_pca = X_tokens_3d.reshape(-1, D_emb) 
pca = PCA(n_components=50, random_state=42)
X_pca_flat = pca.fit_transform(X_flat_for_pca) # Получаем (N_epochs * 248, 50)

# 3. Paper-style Flatten: собираем токены обратно в эпохи и вытягиваем
X_subspace = X_pca_flat.reshape(N_epochs, N_tokens * 50)
print(f"Форма итогового подпространства (Flatten) для UMAP: {X_subspace.shape}")

# 4. Строим UMAP на 12400-мерном подпространстве
print("Обучение UMAP на Flatten-пространстве (это может занять пару минут)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(X_subspace)

# %% 4. Отрисовка графиков
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# График 1: Испытуемые
sns.scatterplot(
    x=embedding_2d[:, 0], 
    y=embedding_2d[:, 1], 
    hue=y_subjects,
    palette=sns.color_palette("husl", len(np.unique(y_subjects))),
    ax=axes[0],
    s=30, alpha=0.7, edgecolor=None
)
axes[0].set_title(f"Подпространство REVE ({X_subspace.shape[1]}-D) - Испытуемые")
axes[0].legend_.remove() 

# График 2: Классы
task_names = []
for l in y_labels:
    if l == 0: task_names.append('Left Hand')
    elif l == 1: task_names.append('Right Hand')
    elif l == 2: task_names.append('Both Fists')
    else: task_names.append('Both Feet')

sns.scatterplot(
    x=embedding_2d[:, 0], 
    y=embedding_2d[:, 1], 
    hue=task_names,
    palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    ax=axes[1],
    s=30, alpha=0.7, edgecolor='k', linewidth=0.2
)
axes[1].set_title(f"Подпространство REVE ({X_subspace.shape[1]}-D) - Задачи")
axes[1].legend(title="Движение")

plt.tight_layout()
plt.show()

# %%

