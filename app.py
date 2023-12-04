from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AdamW

app = Flask(__name__)

# Load the BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Define the BERT architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):  
        super(BERT_Arch, self).__init__()
        self.bert = bert   
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)  # Adjusted dimensions
        self.fc2 = nn.Linear(512, 2)    # Adjusted dimensions
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.dropout(cls_hs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Instantiate the model
model = BERT_Arch(bert)

# Load the pre-trained model weights
model.load_state_dict(torch.load('c2_new_model_weights.pt', map_location=torch.device('cpu')))  # Use 'cpu' for loading on any device
model.eval()

# Sample data for demonstration purposes (replace with actual data)
sample_data = pd.DataFrame({"title": ["Your news title goes here"]})

# Tokenize and encode the sample data
tokens_sample = tokenizer.batch_encode_plus(
    sample_data['title'].tolist(),
    max_length=15,
    pad_to_max_length=True,
    truncation=True
)

# Convert to tensors
sample_seq = torch.tensor(tokens_sample['input_ids'])
sample_mask = torch.tensor(tokens_sample['attention_mask'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input title from the form
        title = request.form['title']

        # Tokenize and encode the input title
        tokens = tokenizer.batch_encode_plus(
            [title],
            max_length=15,
            pad_to_max_length=True,
            truncation=True
        )

        # Convert to tensors
        input_seq = torch.tensor(tokens['input_ids'])
        input_mask = torch.tensor(tokens['attention_mask'])

        # Make predictions
        with torch.no_grad():
            output = model(input_seq, input_mask)
            prediction_prob = torch.exp(output)[0].tolist()  # Convert logits to probabilities
            prediction = np.argmax(prediction_prob)
            class_names = ['True', 'Fake']
            predicted_class = class_names[prediction]

        return render_template('index.html', title=title, predicted_class=predicted_class, prediction_prob=prediction_prob)

if __name__ == '__main__':
    app.run(debug=True)

