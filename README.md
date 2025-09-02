# 🤖 Chatbot with Python (LSTM)

This project is a **chatbot built with Python and Keras** using an LSTM (Long Short-Term Memory) neural network.  
It learns from a given text corpus and can generate responses or predict the next word/character.  

---

## 🚀 Features
- Preprocessing of input text (tokenization, sequence generation).
- LSTM-based neural network for sequence prediction.
- Training with categorical cross-entropy loss and RMSprop optimizer.
- Saving/loading of model and training history.
- Ability to predict next words/characters given an input prompt.
- Support for unknown tokens (`<UNK>`).
- Simple visualization of accuracy and loss.

---

## 📂 Project Structure
chatbot/
│── data/ # Training data (text corpus)
│── model/ # Saved model files (.keras format)
│── notebooks/ # Colab / Jupyter notebooks
│── history.p # Training history (pickled)
│── chatbot.py # Main training & chatbot logic
│── utils.py # Helper functions (input prep, prediction, etc.)
│── requirements.txt # Python dependencies
│── README.md # Project documentation

---

## 🛠️ Installation

1. Clone this repository:
   cd chatbot-lstm
   
Install dependencies:
pip install -r requirements.txt

If using Google Colab, upload your dataset with:

from google.colab import files
files.upload()

📊 Training
Run the training script:
python chatbot.py

This will:
-> Train the LSTM model on your dataset.
-> Save the trained model as keras_next_word_model.keras.
-> Save the training history as history.p.

💡 Usage
Load model and history:

from keras.models import load_model
import pickle

model = load_model('keras_next_word_model.keras')
history = pickle.load(open("history.p", "rb"))

Prepare input for prediction:
from utils import prepare_input
x = prepare_input("this is an example input")
prediction = model.predict(x)

Plot accuracy/loss:
import matplotlib.pyplot as plt
plt.plot(history['accuracy'], label='train')
plt.plot(history['val_accuracy'], label='validation')
plt.legend()
plt.show()

📈 Example Output
Input: "this is an example"
Predicted next word: "of"

✅ Requirements
Python 3.8+
TensorFlow / Keras
NumPy
Matplotlib
Pickle
Install all with:
pip install tensorflow numpy matplotlib

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss.

📜 License
This project is licensed under the MIT License.
