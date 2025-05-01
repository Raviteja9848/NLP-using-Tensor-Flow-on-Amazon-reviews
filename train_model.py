import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# 1️⃣ Load dataset and map to 3 classes
df = pd.read_csv("amazon_alexa.tsv", delimiter='\t', quoting=3, usecols=['rating', 'verified_reviews'])
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])

def map_sentiment(rating):
    if rating > 3:
        return 2  # Positive
    elif rating == 3:
        return 1  # Neutral
    else:
        return 0  # Negative

df['Sentiment'] = df['rating'].apply(map_sentiment)
df["verified_reviews"] = df["verified_reviews"].fillna("")
df = df[['Sentiment', 'verified_reviews']]

# 2️⃣ Tokenization
max_words = 20000
max_len = 150

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df["verified_reviews"])
sequences = tokenizer.texts_to_sequences(df["verified_reviews"])
X_padded = pad_sequences(sequences, maxlen=max_len, padding='post')
y = df["Sentiment"].values

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42, stratify=y)

# 4️⃣ Compute Class Weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',  # Use keyword argument for class_weight
    classes=np.unique(y_train),  # Classes to compute weight for
    y=y_train  # The target labels
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 5️⃣ Build BiLSTM Multi-Class Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 6️⃣ Train the Model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,  # Apply class weights during training
    callbacks=[early_stop]
)

# 7️⃣ Evaluation
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# 8️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Neg', 'Neutral', 'Pos'], yticklabels=['Neg', 'Neutral', 'Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 9️⃣ Save model and tokenizer
model.save("sentiment_bilstm_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Training complete. 3-class model and tokenizer saved.")