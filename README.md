# 🧠 Twitter Sentiment Analysis using LSTM

This project demonstrates a deep learning approach to sentiment analysis on Twitter data using an LSTM-based neural network. A user-friendly Streamlit web application is included to interact with the trained model.

---

## 📂 Dataset
- **File:** `Twitter_Data.csv`
- **Columns:** `text`, `sentiment`
- **Sentiment Classes:** Positive, Negative, Neutral

---

## 🛠️ Preprocessing Steps
- Removed missing values.
- Cleaned non-alphabetic characters and converted text to lowercase.
- Tokenized and padded text sequences.
- Encoded sentiment labels into numeric classes.
- One-hot encoded target labels for multi-class classification.

---

## 🧪 Model Architecture
- `Embedding Layer` (input_dim=10000, output_dim=128)
- `LSTM Layer` (128 units, return_sequences=True)
- `GlobalMaxPooling1D`
- `Dense` layer with 64 units (ReLU)
- `Dropout` layer
- `Dense` layer with 16 units (ReLU)
- `Dropout` layer
- Output `Dense` layer with 3 units (Softmax for 3-class classification)

---

## 🎯 Training
- **Optimizer:** RMSprop (learning_rate=0.001, rho=0.7, momentum=0.5)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **EarlyStopping:** Based on validation accuracy (patience=3)

---

## 📈 Evaluation
Evaluated using:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)

---

## 💾 Model Artifacts
- `Sentiment_DL_Model.h5` — Trained Keras LSTM model
- `Sentiment_tokenizer.joblib` — Trained Tokenizer used for input text preprocessing

---

## 🌐 Streamlit App
A lightweight web app is created using **Streamlit** in `app.py`:
- Input a text string
- The model returns sentiment: **Positive**, **Negative**, or **Neutral**

---

## ▶️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/codedbykenneth/twitter-sentiment-lstm.git
cd twitter-sentiment-lstm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python main.py
```

### 4. Launch Streamlit App
```bash
streamlit run app.py
```

---

## 📜 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 📧 Contact
For questions or feedback, feel free to raise an issue or contact via GitHub.

