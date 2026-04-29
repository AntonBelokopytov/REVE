# %% БЛОК 1: Импорты и загрузка базовой модели
import torch
from transformers import AutoModel
import mne
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import numpy as np

# Твой токен от HuggingFace
my_token = "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"

# Загружаем саму модель
model = AutoModel.from_pretrained(
    "brain-bzh/reve-base", 
    trust_remote_code=True, 
    token=my_token
)
model.eval()

# %% БЛОК 2: Подготовка данных ЭЭГ (загрузка и ресемплинг)
fpath = 'REVE_aos/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf'
raw = mne.io.read_raw_edf(fpath, preload=True)

# Модель обучалась на 200 Гц, поэтому обязательно меняем частоту
raw.resample(200)

# %% БЛОК 3: Выравнивание каналов и извлечение позиций электродов
# 1. Загружаем банк позиций
pos_bank = AutoModel.from_pretrained(
    "brain-bzh/reve-positions", 
    trust_remote_code=True, 
    token=my_token
)

# 2. Очищаем названия каналов текущей записи
clean_ch_names = [ch.replace('.', '').strip().upper() for ch in raw.ch_names]

# 3. Получаем тензор позиций для тех каналов, которые знает модель
positions = pos_bank(clean_ch_names)

# ВАЖНО: Добавляем батч-измерение для позиций (1, N_channels, 3)
batch_pos = positions.unsqueeze(0)

# 4. Находим оригинальные имена каналов, которых нет в словаре pos_bank
valid_positions = set(pos_bank.position_names)
channels_to_drop = [
    orig_ch for orig_ch, clean_ch in zip(raw.ch_names, clean_ch_names)
    if clean_ch not in valid_positions
]

print(f"Каналы, не найденные в словаре REVE (будут удалены): {channels_to_drop}")

# 5. Синхронизируем: удаляем неизвестные каналы из ЭЭГ
raw_filtered = raw.copy().drop_channels(channels_to_drop)

# 6. Вырезаем кусок данных (4 секунды при 200 Гц = 800 отсчетов)
window_samples = 4 * 200
eeg_data_filtered = raw_filtered.get_data(start=0, stop=window_samples)

# 7. Формируем итоговый батч ЭЭГ (1, N_channels, Time)
batch_eeg = torch.tensor(eeg_data_filtered, dtype=torch.float32).unsqueeze(0)

print(f"Размерность batch_eeg: {batch_eeg.shape}")
print(f"Размерность batch_pos: {batch_pos.shape}")

# %% БЛОК 4: Прогон через модель и сбор слоев (без хуков)
model.eval()

# Прогоняем данные со специальным флагом return_output=True
# В этом режиме REVE возвращает список (list) тензоров для каждого слоя
with torch.no_grad():
    outputs = model(batch_eeg, batch_pos, return_output=True)

# outputs — это список. Первый элемент outputs[0] — это вход в трансформер.
# outputs[1] до outputs[22] — это выходы после каждого из 22 слоев энкодера.
# Нам нужны только тензоры, переводим их в numpy.
layer_outputs = [out.detach().cpu().numpy() for out in outputs]

print(f"Количество собранных слоев: {len(layer_outputs)}")

# %% БЛОК 5: Вычисление метрики Procrustes и отрисовка графика
linearity_scores = []

# Считаем линейность (схожесть) между каждыми соседними слоями i и i+1
for i in range(len(layer_outputs) - 1):
    # Убираем батч (индекс 0), получаем 2D матрицу [Sequence_length, Hidden_dim]
    layer_a = layer_outputs[i][0]
    layer_b = layer_outputs[i + 1][0]
    
    # scipy.spatial.procrustes возвращает (matrix1, matrix2, disparity)
    # disparity — это ошибка несовпадения (чем меньше, тем больше матрицы похожи)
    _, _, disparity = procrustes(layer_a, layer_b)
    
    # Линейность считаем как 1 - disparity (чем ближе к 1, тем линейнее переход)
    similarity = 1.0 - disparity
    linearity_scores.append(similarity)

# Отрисовка
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(linearity_scores) + 1), linearity_scores, marker='o', linestyle='-', color='b')

plt.xlabel("Layer Transition (i -> i+1)")
plt.ylabel("Procrustes Linearity Score")
plt.title("REVE-Base: Layer-to-Layer Linearity (Procrustes Metric)")
plt.xticks(range(1, len(linearity_scores) + 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%

