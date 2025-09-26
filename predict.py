import os
import pickle
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tf_keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tf_keras.models import Model


print("Memuat artefak model... Harap tunggu.")

MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LENGTH = 128
SAVED_MODELS_DIR = 'saved_models' # Path relatif untuk lingkungan lokal

try:
    tokenizer_path = os.path.join(SAVED_MODELS_DIR, 'tokenizer_3class')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    with open(os.path.join(SAVED_MODELS_DIR, 'label_encoder_3class.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    num_classes = len(label_encoder.classes_)

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
    
    print("✅ Artefak berhasil dimuat. Model siap digunakan.")

except Exception as e:
    print(f"❌ Terjadi kesalahan saat memuat artefak: {e}")
    model = None

def predict_sentiment(text):
    """
    Fungsi untuk memprediksi sentimen dari satu kalimat.
    Input: text (string)
    Output: predicted_label (string)
    """
    if model is None:
        return "Model tidak berhasil dimuat."

    encoded_text = tokenizer.encode_plus(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    
    input_dict = {'input_ids': encoded_text['input_ids'], 'attention_mask': encoded_text['attention_mask']}
    pred_probs = model.predict(input_dict, verbose=0)
    
    pred_encoded = np.argmax(pred_probs, axis=1)
    predicted_label = label_encoder.inverse_transform(pred_encoded)
    
    return predicted_label[0]

if __name__ == "__main__" and model is not None:
    print("\n" + "="*50)
    print("       Model Prediksi Sentimen Interaktif")
    print("="*50)
    print("Ketik kalimat yang ingin Anda analisis.")
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri program.")
    print("-" * 50)

    while True:
        kalimat_input = input("Masukkan kalimat: ")

        if kalimat_input.lower() in ['exit', 'keluar']:
            print("\nTerima kasih telah menggunakan program ini. Sampai jumpa!")
            break

        if kalimat_input.strip():
            prediksi = predict_sentiment(kalimat_input)
            print(f"   Prediksi Sentimen -> {prediksi}\n")
        else:
            print("   Input tidak boleh kosong. Silakan coba lagi.\n")