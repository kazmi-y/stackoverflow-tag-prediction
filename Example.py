from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and preprocessing objects
model = load_model('models/stack_overflow_lstm_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('models/mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

def predict_tags(text, max_len=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    tags = mlb.inverse_transform((pred > 0.5).astype(int))
    return tags

# Example
print(predict_tags("How do I connect to SQL Server using C#?"))
