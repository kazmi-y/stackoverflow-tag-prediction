# Stack Overflow Multi-label Tag Prediction with LSTM

## Overview

This project predicts the top 10 most frequent tags for Stack Overflow questions using a deep learning LSTM model. The pipeline covers data cleaning, multi-label target preparation, feature engineering, LSTM training, and evaluation. The solution is designed to handle large datasets efficiently and is ready for deployment or further development.


### Data Overview:

We loaded and explored two datasets: Questions.csv and Tags.csv.
The Questions dataset contains information about Stack Overflow questions, including Title, Body, CreationDate, Score, etc.
The Tags dataset links question IDs to their respective tags.
There are over 1.2 million unique questions and 37,000 unique tags.
Missing Values and Duplicates:

The Questions dataset has missing values for OwnerUserId and ClosedDate.
The Tags dataset has a small number of missing Tag values and one duplicate row.
Tag Distribution:

The top 10 most frequent tags were identified and visualized: 'javascript', 'java', 'c#', 'php', 'android', 'jquery', 'python', 'html', 'c++', and 'ios'.
Most questions have between 2 and 4 tags, with a maximum of 5 tags per question.
Text Analysis:

We analyzed the length of question titles and bodies. Titles are generally much shorter than bodies.
No empty titles or bodies were found.
Preprocessing and Model Preparation:

We focused on questions associated with the top 10 tags.
HTML was removed from the question bodies, and the text was cleaned by converting to lowercase and removing non-alphabetic characters.
Cleaned titles and bodies were combined for modeling.
Multi-label binarization was applied to the top 10 tags to create target variables for the model.
The text data was tokenized and padded for input into the LSTM model.
The data was split into training and validation sets.
Model Training and Evaluation:

A Sequential LSTM model was built for multi-label classification.
The model was trained for 5 epochs.
The classification report shows the precision, recall, and F1-score for each of the top 10 tags on the validation set.
The Hamming Loss and Subset Accuracy were also calculated to evaluate the multi-label classification performance.
Model and Tokenizer Saving:

The trained LSTM model and the tokenizer object were saved for future use.
---

## How to Load and Use the Model for Inference

1. **Clone or Download the Repository**

2. **Install Dependencies**  
   Make sure you have Python 3.8+ and the following packages installed:
   - `tensorflow` (2.x)
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `pickle` (standard library, for loading objects)
   - `bs4` (BeautifulSoup, for text cleaning)
   - `lxml` (for fast HTML parsing)

   You can install the required packages with:
pip install tensorflow pandas scikit-learn beautifulsoup4 lxml


3. **Load the Model and Preprocessing Objects**
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

Load objects
model = load_model('stack_overflow_lstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
tokenizer = pickle.load(f)

with open('mlb.pkl', 'rb') as f:
mlb = pickle.load(f)

Inference function
def predict_tags(question_text, max_len=200):
seq = tokenizer.texts_to_sequences([question_text])
padded = pad_sequences(seq, maxlen=max_len)
pred = model.predict(padded)
tags = mlb.inverse_transform((pred > 0.5).astype(int))
return tags

Example usage
question = "How do I connect to SQL Server using C#?"
print("Predicted tags:", predict_tags(question))


---

## Dependencies and Environment Setup

- **Python**: 3.8 or above
- **TensorFlow**: 2.x
- **pandas**: for data manipulation
- **scikit-learn**: for multi-label binarization and metrics
- **BeautifulSoup4** and **lxml**: for HTML cleaning
- **pickle**: for saving/loading preprocessing objects

Install all dependencies with:

pip install tensorflow pandas scikit-learn beautifulsoup4 lxml


---

## Approach and Results

### **Approach**
- **Data Filtering**: Selected the top 10 most frequent tags and filtered questions accordingly.
- **Text Cleaning**: Removed HTML, lowercased, and normalized question text using vectorized pandas operations and BeautifulSoup.
- **Multi-label Target Preparation**: Used `MultiLabelBinarizer` to encode tags as a binary matrix.
- **Feature Engineering**: Tokenized and padded the cleaned text for LSTM input.
- **Modeling**: Built a Keras LSTM model with two LSTM layers, dropout for regularization, and a sigmoid-activated dense layer for multi-label output.
- **Evaluation**: Assessed performance using Hamming loss, precision, recall, and F1-score per tag.

### **Results**
- The LSTM model achieved strong performance on the validation set, with high F1-scores for the most frequent tags and low Hamming loss.
- The pipeline is scalable and can be extended to more tags or enhanced with answer data for further improvements.

---

## Notes

- The model and preprocessing objects (`.h5`, `.pkl`) must be present in the project directory for inference.
- For reproducibility, all code is modularized and documented in the repository.
- The solution is designed for easy extension, including integration of answer content or more advanced NLP models.

---

**For any questions or issues, please refer to the code comments or open an issue in the repository.**
