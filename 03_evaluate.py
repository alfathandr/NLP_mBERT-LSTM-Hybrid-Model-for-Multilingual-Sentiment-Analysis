import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, TFBertModel
from tf_keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tf_keras.models import Model

MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LENGTH = 128

# Sesuaikan path ini dengan struktur di Google Drive Anda
DATASET_DIR = 'dataset'
SAVED_MODELS_DIR = 'saved_models'
TRAIN_FILE = 'train_balanced_3class.csv'

print("Memuat tokenizer, label encoder, dan data...")

tokenizer_path = os.path.join(SAVED_MODELS_DIR, 'tokenizer_3class')
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

with open(os.path.join(SAVED_MODELS_DIR, 'label_encoder_3class.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

df = pd.read_csv(os.path.join(DATASET_DIR, TRAIN_FILE))
df.dropna(inplace=True)

_, df_val = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df['label']
)

print(f"Jumlah data validasi untuk evaluasi: {len(df_val)}")


print("Membangun ulang arsitektur model...")

input_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')
bert_model = TFBertModel.from_pretrained(MODEL_NAME, from_pt=True)
sequence_output = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
bilstm = Bidirectional(LSTM(64))(sequence_output)
dropout = Dropout(0.3)(bilstm)
dense_output = Dense(num_classes, activation='softmax')(dropout)
model = Model(inputs=[input_ids, attention_mask], outputs=dense_output)

model_weights_path = os.path.join(SAVED_MODELS_DIR, 'mbert_lstm_3class.h5')
model.load_weights(model_weights_path)
print("Bobot model terbaik berhasil dimuat.")

print("Melakukan tokenisasi pada data validasi...")
X_val = tokenizer.batch_encode_plus(
    df_val['text'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors='tf'
)

print("Melakukan prediksi...")
y_pred_probs = model.predict({'input_ids': X_val['input_ids'], 'attention_mask': X_val['attention_mask']})
y_pred_encoded = np.argmax(y_pred_probs, axis=1)

y_true_encoded = label_encoder.transform(df_val['label'])

print("\n" + "="*60)
print("                    LAPORAN HASIL KLASIFIKASI")
print("="*60)

class_names = label_encoder.classes_
print(classification_report(y_true_encoded, y_pred_encoded, target_names=class_names))
print("="*60)

print("\nMenampilkan Confusion Matrix...")
cm = confusion_matrix(y_true_encoded, y_pred_encoded)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Label Sebenarnya (True)')
plt.xlabel('Label Prediksi (Predicted)')
plt.show()