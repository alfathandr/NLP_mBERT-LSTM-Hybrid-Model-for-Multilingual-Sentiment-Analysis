import pandas as pd
import os
from imblearn.over_sampling import RandomOverSampler

DATASET_DIR = 'dataset'
TRAIN_FILE = 'tamil_sentiment_full_train.csv' 
OUTPUT_FILE = 'train_balanced_3class.csv' # ---> DIUBAH: Nama file output baru

input_path = os.path.join(DATASET_DIR, TRAIN_FILE)
output_path = os.path.join(DATASET_DIR, OUTPUT_FILE)

if not os.path.exists(DATASET_DIR):
    print(f"Error: Folder '{DATASET_DIR}' tidak ditemukan.")
    print("Pastikan Anda menjalankan skrip ini dari direktori yang benar.")
    exit()

print(f"Membaca dataset dari: {input_path}")
try:
    df = pd.read_csv(
        input_path, 
        sep='\t', 
        header=None, 
        names=['text', 'label'],
        on_bad_lines='warn'
    )
except FileNotFoundError:
    print(f"Error: File '{input_path}' tidak ditemukan.")
    exit()

print("\n--- 5 Baris Pertama Data Asli ---")
print(df.head())
print("\nUkuran data asli:", df.shape)

df.dropna(subset=['text', 'label'], inplace=True)
print("Ukuran data setelah menghapus nilai kosong:", df.shape)

print("\nMemfilter data untuk 3 kelas (Positive, Negative, Mixed_feelings)...")

selected_labels = ['Positive', 'Negative', 'Mixed_feelings']
df = df[df['label'].isin(selected_labels)].copy() 

print(f"Ukuran data setelah filtering: {df.shape}")

print("Mengubah label 'Mixed_feelings' menjadi 'Neutral'...")
df['label'] = df['label'].replace({'Mixed_feelings': 'Neutral'})

print("Pemetaan label selesai.")


print("\n" + "="*50)
print("ANALISIS DISTRIBUSI KELAS (SEBELUM RANDOM OVER SAMPLING)")
print("="*50)
print("Jumlah data per kelas:")
print(df['label'].value_counts())
print("\nProporsi data per kelas (%):")
print(df['label'].value_counts(normalize=True) * 100)
print("="*50)


print("\nMemisahkan fitur (X) dan label (y)...")
X = df[['text']] 
y = df['label']

print("Menerapkan Random Over Sampling (ROS)...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

df_balanced = pd.DataFrame(X_resampled, columns=['text'])
df_balanced['label'] = y_resampled

print("Proses ROS selesai.")
print("\nUkuran data setelah di-resample:", df_balanced.shape)

print("\n" + "="*50)
print("ANALISIS DISTRIBUSI KELAS (SETELAH RANDOM OVER SAMPLING)")
print("="*50)
print("Jumlah data per kelas:")
print(df_balanced['label'].value_counts())
print("\nProporsi data per kelas (%):")
print(df_balanced['label'].value_counts(normalize=True) * 100)
print("="*50)

print(f"\nMenyimpan data yang sudah seimbang ke: {output_path}")
df_balanced.to_csv(output_path, index=False, encoding='utf-8')

print("\n🎉 Skrip persiapan data selesai dijalankan!")
print(f"Data latih yang seimbang kini siap digunakan di '{output_path}'.")