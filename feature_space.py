import os
import mne
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from braindecode.models import REVE
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# %% 1. Инициализация моделей
my_token = "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"

print("Загрузка REVE через Braindecode...")
model = REVE.from_pretrained(
    "brain-bzh/reve-base", 
    n_outputs=512,
    token=my_token,
    attention_pooling=True
).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pos_bank_hf = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, token=my_token)
valid_positions = set(pos_bank_hf.position_names)

# %% 2. Извлечение признаков со всех испытуемых
# Отключаем классификатор, чтобы модель выдавала результат пулинга
model.final_layer = torch.nn.Identity()

runs = ['04', '08', '12'] # Задачи на воображение движений руками
batch_size = 8

all_features = []
all_labels = []
all_subjects = []

print("Начинаем обработку базы PhysioNet...")
for sid in range(1, 110):
    raws = []
    skip_subject = False
    
    # 2.1 Загрузка записей
    for r in runs:
        fpath = f'REVE_aos/MNE-eegbci-data/files/eegmmidb/1.0.0/S{sid:03d}/S{sid:03d}R{r}.edf'
        
        if not os.path.exists(fpath):
            skip_subject = True
            break
            
        try:
            raw_tmp = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
            mne.rename_channels(raw_tmp.info, {ch: ch.replace('.', '').strip().upper() for ch in raw_tmp.ch_names})
            raws.append(raw_tmp)
        except Exception as e:
            print(f"Ошибка чтения S{sid:03d}R{r}: {e}")
            skip_subject = True
            break
            
    if skip_subject or not raws:
        continue
        
    try:
        # 2.2 Склейка и предобработка
        raw = mne.concatenate_raws(raws)
        raw.resample(200, verbose=False)
        
        # Оставляем только известные каналы
        channels_to_drop = [ch for ch in raw.ch_names if ch not in valid_positions]
        raw.drop_channels(channels_to_drop)
        
        # встроенный метод получения тензора позиций
        positions = model.get_positions(raw.ch_names).to(device)
        
        # 2.3 Нарезка на эпохи
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        mapping = {}
        if 'T1' in event_dict: mapping['T1'] = event_dict['T1']
        if 'T2' in event_dict: mapping['T2'] = event_dict['T2']
        
        if not mapping:
            continue
            
        tmax = 4.0 - (1 / raw.info['sfreq'])
        epochs = mne.Epochs(raw, events, event_id=mapping, tmin=0, tmax=tmax, baseline=None, preload=True, verbose=False)
        
        data_tensor = torch.tensor(epochs.get_data(copy=True), dtype=torch.float32)
        subj_labels = np.array([0 if l == mapping.get('T1', -1) else 1 for l in epochs.events[:, -1]])
        
        # 2.4 Прогон через REVE (Braindecode way)
        subj_features = []
        with torch.no_grad():
            for i in range(0, len(data_tensor), batch_size):
                X_batch = data_tensor[i:i+batch_size].to(device)
                B = X_batch.shape[0]
                current_pos = positions.unsqueeze(0).expand(B, -1, -1)
                
                # Обычный прогон (без return_features=True).
                # Сигнал пройдет через энкодер, затем через включенный attention_pooling,
                # и выйдет через наш Identity-слой в нужном виде (B, 512)
                pooled_features = model(X_batch, pos=current_pos)
                
                subj_features.append(pooled_features.cpu().numpy())
                
        all_features.append(np.vstack(subj_features))
        all_labels.extend(subj_labels)
        all_subjects.extend([sid] * len(subj_labels))
        
        print(f"Субъект S{sid:03d} успешно обработан ({len(subj_labels)} эпох).")
        
    except Exception as e:
        print(f"Пропуск S{sid:03d} из-за ошибки обработки: {e}")

# Объединяем все данные
X_features = np.vstack(all_features)
y_labels = np.array(all_labels)
y_subjects = np.array(all_subjects)

print(f"\nСбор данных завершен! Итоговая матрица признаков: {X_features.shape}")

# %% 3. Построение UMAP
print("Обучение UMAP на всем массиве данных...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(X_features)

# Рисуем два графика рядом
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# График 1: Раскраска по ИСПЫТУЕМЫМ
sns.scatterplot(
    x=embedding_2d[:, 0], 
    y=embedding_2d[:, 1], 
    hue=y_subjects,
    palette=sns.color_palette("husl", len(np.unique(y_subjects))),
    ax=axes[0],
    s=30, 
    alpha=0.7, 
    edgecolor=None
)
axes[0].set_title("Скрытое пространство REVE (Цвет = Испытуемый)")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")
axes[0].legend_.remove() 

# График 2: Раскраска по КЛАССАМ ДВИЖЕНИЙ
task_names = ['Левая рука (T1)' if l == 0 else 'Правая рука (T2)' for l in y_labels]
sns.scatterplot(
    x=embedding_2d[:, 0], 
    y=embedding_2d[:, 1], 
    hue=task_names,
    palette=['#1f77b4', '#ff7f0e'],
    ax=axes[1],
    s=30, 
    alpha=0.7, 
    edgecolor='k',
    linewidth=0.2
)
axes[1].set_title("Скрытое пространство REVE (Цвет = Задача)")
axes[1].set_xlabel("UMAP 1")
axes[1].legend(title="Воображаемое движение")

plt.tight_layout()
plt.show()

# %% 4. Линейное пробирование (Linear Probing)
print("\n--- Выполнение линейного пробирования ---")
# Используем логистическую регрессию для оценки качества извлеченных признаков (насколько они линейно разделимы)
clf = LogisticRegression(max_iter=1000, random_state=42)

# Оценка качества модели с помощью 5-кратной кросс-валидации
scores = cross_val_score(clf, X_features, y_labels, cv=5, scoring='accuracy')

print(f"Точность (Accuracy) линейной пробы на кросс-валидации (5 фолдов): {scores}")
print(f"Средняя точность кросс-валидации: {scores.mean():.4f} ± {scores.std():.4f}")

# Обучаем модель на всех данных
clf.fit(X_features, y_labels)
y_pred = clf.predict(X_features)
train_acc = accuracy_score(y_labels, y_pred)
print(f"Точность (Accuracy) на всей выборке: {train_acc:.4f}")