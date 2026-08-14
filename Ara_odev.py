"""
Proje: Müşteri Ayrılma (Churn) Tahmini - Makine Öğrenmesi Ara Ödevi
Amaç: Temel makine öğrenmesi boru hattını (pipeline) kurarak, müşterilerin 
      ayrılıp ayrılmayacağını (churn) tahmin etmek. Veri seti kod içinde sentetik 
      olarak üretilmekte, ön işleme adımlarından geçirilmekte ve sınıflandırma 
      modelleri ile test edilmektedir.

Kullanılan Kütüphaneler:
- pandas, numpy: Veri manipülasyonu ve sentetik veri üretimi
- scikit-learn: Ön işleme (StandardScaler, One-Hot), modelleme (LogReg, KNN, Tree) 
                ve metrik hesaplamaları (Confusion Matrix, F1, Accuracy vb.)
- matplotlib, seaborn: Sonuçların görselleştirilmesi

Çalıştırma Adımları:
1. Gerekli kütüphaneleri kurun: pip install -r requirements.txt
2. Terminal veya komut satırından betiği çalıştırın: python churn_prediction.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. VERİ OLUŞTURMA ---
print("--- 1. Veri Üretimi ve İnceleme ---")
np.random.seed(42)
n_rows = 200

data = {
    'yas': np.random.randint(18, 65, n_rows),
    'gelir': np.random.randint(3000, 25000, n_rows),
    'abonelik_suresi': np.random.randint(1, 60, n_rows),
    'destek_talebi_sayisi': np.random.randint(0, 10, n_rows),
    'sehir': np.random.choice(['Istanbul', 'Ankara', 'Izmir'], n_rows),
    'uyelik_tipi': np.random.choice(['Standart', 'Premium'], n_rows),
}
df = pd.DataFrame(data)

# Churn hedefini destek talebi ve abonelik süresine bağlı olarak sentetik üretelim
olasilik = (df['destek_talebi_sayisi'] * 0.1) - (df['abonelik_suresi'] * 0.005)
df['churn'] = (olasilik > np.median(olasilik)).astype(int)

# --- 2. EKSİK VERİ KONTROLÜ VE VERİ İNCELEME ---
# Bilinçli olarak birkaç eksik veri (NaN) ekleyelim
df.loc[5:10, 'gelir'] = np.nan
df.loc[15:20, 'yas'] = np.nan

print(f"Veri Seti Boyutu: {df.shape}")
print("\nİlk 5 Satır:\n", df.head())
print("\nHedef Değişken Dağılımı:\n", df['churn'].value_counts())
print("\nEksik Değerler (Doldurmadan Önce):\n", df.isnull().sum())

# Eksik verileri medyan ile doldurma
df['gelir'] = df['gelir'].fillna(df['gelir'].median())
df['yas'] = df['yas'].fillna(df['yas'].median())

# --- 3. ÖZNİTELİK ÜRETME (FEATURE ENGINEERING) ---
print("\n--- 2. Öznitelik Üretme ve Ön İşleme ---")
# Yeni öznitelik: Müşteri hiç destek talebinde bulundu mu? (0: Hayır, 1: Evet)
df['destek_talebi_var_mi'] = (df['destek_talebi_sayisi'] > 0).astype(int)

# --- 4. KATEGORİK DEĞİŞKENLERİ DÖNÜŞTÜRME (ONE-HOT ENCODING) ---
df = pd.get_dummies(df, columns=['sehir', 'uyelik_tipi'], drop_first=True)

# --- 5. VERİYİ BÖLME (TRAIN - VALIDATION - TEST) ---
# %60 Train, %20 Validation, %20 Test olacak şekilde stratify kullanarak bölüyoruz
X = df.drop('churn', axis=1)
y = df['churn']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print(f"Train seti: {X_train.shape[0]} satır")
print(f"Validation seti: {X_val.shape[0]} satır")
print(f"Test seti: {X_test.shape[0]} satır")

# --- 6. ÖLÇEKLEME (SCALING) ---
scaler = StandardScaler()
# Sadece sayısal sütunları ölçekliyoruz
num_cols = ['yas', 'gelir', 'abonelik_suresi', 'destek_talebi_sayisi']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# --- 7. MODEL EĞİTİMİ VE VALIDATION KARŞILAŞTIRMASI ---
print("\n--- 3. Model Eğitimi ve Validation ---")
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42)
}

best_model_name = ""
best_val_score = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"{name} Validation Accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_score:
        best_val_score = val_acc
        best_model_name = name
        best_model = model

print(f"\nEn iyi performans gösteren model: {best_model_name}")

# --- 8. TEST VERİSİNDE DEĞERLENDİRME ---
print("\n--- 4. Test Seti Değerlendirmesi ---")
y_test_pred = best_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred):.4f}")

# --- Bonus: Confusion Matrix Görselleştirme ---
# Özel renk paleti yapılandırması
plt.figure(figsize=(6, 4), facecolor='#1e2a38')
ax = plt.axes()
ax.set_facecolor('#1e2a38')

cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            cbar=False, ax=ax, 
            annot_kws={"color": "white", "size": 14})

ax.set_title(f'{best_model_name} Confusion Matrix', color='white', pad=15)
ax.set_xlabel('Tahmin Edilen (Predicted)', color='white')
ax.set_ylabel('Gerçek (Actual)', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

plt.tight_layout()
plt.savefig('confusion_matrix.png', facecolor='#1e2a38')
print("\n'confusion_matrix.png' grafiği çalışma dizinine kaydedildi.")

# --- 9. KISA YORUM ÇIKTISI ---
print("\n--- 5. Sonuç Yorumu ---")
print(f"Eğitilen modeller arasında {best_model_name} validation setinde en yüksek "
      f"doğruluğu ({best_val_score:.2f}) gösterdi. Sentetik veri setimizde churn "
      f"hedefi doğrudan destek talebi ve abonelik süresi ile matematiksel olarak "
      f"ilişkilendirildiği için, veri yapısındaki doğrusal (linear) eğilimleri veya "
      f"kesin kuralları yakalayan modeller (LogReg/Decision Tree) genellikle "
      f"mesafe tabanlı (KNN) modellere göre daha tutarlı sonuçlar vermiştir. "
      f"Test setindeki F1-Skoru ({f1_score(y_test, y_test_pred):.2f}) "
      f"modelin genel başarısını doğrulamaktadır.")