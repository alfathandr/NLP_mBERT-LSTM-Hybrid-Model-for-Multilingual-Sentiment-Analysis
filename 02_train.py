import os
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from tf_keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping
from tf_keras.utils import to_categorical

MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

DATASET_DIR = 'dataset'
TRAIN_FILE = 'train_balanced_3class.csv'
SAVED_MODELS_DIR = 'saved_models'
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

print("Membaca dataset yang sudah seimbang...")
train_path = os.path.join(DATASET_DIR, TRAIN_FILE)
df = pd.read_csv(train_path)
df.dropna(inplace=True)

print(f"Total data yang dimuat: {len(df)}")

print("Membagi data menjadi 90% training dan 10% validation...")
df_train, df_val = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df['label']
)

print(f"Jumlah data training baru: {len(df_train)}")
print(f"Jumlah data validasi baru: {len(df_val)}")

print("Melakukan Label Encoding...")
label_encoder = LabelEncoder()
label_encoder.fit(df_train['label'])

y_train_encoded = label_encoder.transform(df_train['label'])
y_val_encoded = label_encoder.transform(df_val['label'])

num_classes = len(label_encoder.classes_)
y_train = to_categorical(y_train_encoded, num_classes=num_classes)
y_val = to_categorical(y_val_encoded, num_classes=num_classes)

print(f"Kelas yang ditemukan: {label_encoder.classes_} ({num_classes} kelas)")

with open(os.path.join(SAVED_MODELS_DIR, 'label_encoder_3class.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder tersimpan.")

print(f"Inisialisasi tokenizer dari '{MODEL_NAME}'...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def batch_tokenize(texts):
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )

print("Melakukan tokenisasi pada data training...")
X_train = batch_tokenize(df_train['text'])

print("Melakukan tokenisasi pada data validasi...")
X_val = batch_tokenize(df_val['text'])

tokenizer.save_pretrained(os.path.join(SAVED_MODELS_DIR, 'tokenizer_3class'))
print("Tokenizer tersimpan.")

print("Membangun arsitektur model hybrid mBERT-LSTM...")

input_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')

# ---> PERBAIKAN DI SINI: Tambahkan from_pt=True
bert_model = TFBertModel.from_pretrained(MODEL_NAME, from_pt=True)

bert_model.trainable = True
bert_output = bert_model(input_ids, attention_mask=attention_mask)
sequence_output = bert_output.last_hidden_state

bilstm = Bidirectional(LSTM(64, return_sequences=False))(sequence_output)
dropout_1 = Dropout(0.3)(bilstm)

dense_output = Dense(num_classes, activation='softmax')(dropout_1)

model = Model(inputs=[input_ids, attention_mask], outputs=dense_output)

optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

print("\nMempersiapkan callbacks...")
checkpoint_path = os.path.join(SAVED_MODELS_DIR, 'mbert_lstm_3class.h5')
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

print("\n--- MEMULAI PELATIHAN MODEL ---")
history = model.fit(
    {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},
    y_train,
    validation_data=({'input_ids': X_val['input_ids'], 'attention_mask': X_val['attention_mask']}, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[model_checkpoint, early_stopping]
)

print("\n🎉 Pelatihan model selesai!")
print(f"Model terbaik disimpan di: {checkpoint_path}")