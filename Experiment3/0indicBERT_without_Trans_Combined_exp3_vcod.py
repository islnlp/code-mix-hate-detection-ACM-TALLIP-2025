# Importing Libraries

import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from transformers import BertModel, BertConfig
from tqdm import tqdm
import logging
import time
import os

#Arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_lang', type=str, default='hi+en', help='For exp.3 train lang: hi, en, hi+en')
args = parser.parse_args()

# saving to log files

logging.basicConfig(filename=f'/data1/aakash/Codemix/Aakash_02/logs/mBERT(Com)_{time.asctime().replace(" ","_")}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a stream handler to print log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading dataset

Cod=pd.read_csv('/data1/aakash/Codemix/Aakash_02/Codes/Dataset/1HateSpeech_Codemix.csv') #codemix
Cod=Cod.dropna()

#Splitting
train_df, remaining_df = train_test_split(Cod, test_size=0.30, random_state=random_seed, stratify=Cod['Tag'])
test_df, val_df = train_test_split(remaining_df, test_size=0.5, random_state=random_seed, stratify=remaining_df['Tag'])


eng = pd.read_csv('/data1/aakash/Codemix/Aakash_02/Codes/Dataset/1Hatespeech_English(new).csv')


hind = pd.read_csv('/data1/aakash/Codemix/Aakash_02/Codes/Dataset/1HateSpeech_Hindi.csv')




# Select rows with the specified label

#da = eng[eng['Tag'] == 1].iloc[:200]  #Default value. You can check for other values by below lines.
#da = eng[eng['Tag'] == 1].iloc[:400]
#da = eng[eng['Tag'] == 1].iloc[:600]
#da = eng[eng['Tag'] == 1].iloc[:800]
#da = eng[eng['Tag'] == 1].iloc[:1000]
#da = eng[eng['Tag'] == 1].iloc[:1200]
#da = eng[eng['Tag'] == 1].iloc[:1400]
da = eng[eng['Tag'] == 1].iloc[:1416]
#da = eng[eng['Tag'] == 1].iloc[:2261] #highest possible sample
da = da.reset_index(drop=True)

#db = eng[eng['Tag']==0].iloc[:200]    #Default value. You can check for other values by below lines.
#db = eng[eng['Tag'] == 0].iloc[:400]
#db = eng[eng['Tag'] == 0].iloc[:600]
#db = eng[eng['Tag'] == 0].iloc[:800]
#db = eng[eng['Tag'] == 0].iloc[:1000]
#db = eng[eng['Tag'] == 0].iloc[:1200]
#db = eng[eng['Tag'] == 0].iloc[:1400]
db = eng[eng['Tag'] == 0].iloc[:1416]
#db = eng[eng['Tag'] == 0].iloc[:2261]  #highest possible sample
db = db.reset_index(drop=True)

eng_com= pd.concat([da, db])#train_dx
eng_com = eng_com.reset_index(drop=True)

# Select rows with the specified label
#dk = hind[hind['Tag'] == 1].iloc[:200]  #Default value. You can check for other values by below lines.
#dk = hind[hind['Tag'] == 1].iloc[:400]
#dk = hind[hind['Tag'] == 1].iloc[:600]
#dk = hind[hind['Tag'] == 1].iloc[:800]
#dk = hind[hind['Tag'] == 1].iloc[:1000]
#dk = hind[hind['Tag'] == 1].iloc[:1200]
#dk = hind[hind['Tag'] == 1].iloc[:1400]
dk = hind[hind['Tag'] == 1].iloc[:1416]
dk = dk.reset_index(drop=True)

#dm = hind[hind['Tag']==0].iloc[:200]     #Default value. You can check for other values by below lines.
#dm = hind[hind['Tag'] == 0].iloc[:400]
#dm = hind[hind['Tag'] == 0].iloc[:600]
#dm = hind[hind['Tag'] == 0].iloc[:800]
#dm = hind[hind['Tag'] == 0].iloc[:1000]
#dm = hind[hind['Tag'] == 0].iloc[:1200]
#dm = hind[hind['Tag'] == 0].iloc[:1400]
dm = hind[hind['Tag'] == 0].iloc[:1416]
dm = dm.reset_index(drop=True)

hind_com= pd.concat([dk, dm])#train_dy
hind_com = hind_com.reset_index(drop=True)

# select which the data to train (Exp2)
train_lang = args.train_lang

if train_lang == 'hi':
    dfs = [hind_com]
elif train_lang == 'en':
    dfs = [eng_com]
else:
    dfs = [eng_com,hind_com]

# # Create a list of dataframes
# dfs = [eng_com,hind_com]

# Shuffle the order of dataframes randomly
random.shuffle(dfs)

# Concatenate dataframes
t_com = pd.concat(dfs)

# Reset the index
t_com = t_com.reset_index(drop=True)

t_com.Tag.value_counts()
val_df.Tag.value_counts()


# Creating custom dataset

class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        try:
            sentence = str(self.sentences[idx])
            label = self.labels[idx]

            encoding = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Problematic sentence: {self.sentences[idx]}")
            print(f"Problematic label: {self.labels[idx]}")
            raise e
        

# Model

from transformers import BertModel
from transformers import AutoModel, AutoTokenizer

class CustomMBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomMBERTModel, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bert = AutoModel.from_pretrained('ai4bharat/indic-bert', output_hidden_states=True)

        # # Adding Transformer layers
        # self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=8), num_layers=4)

        # Adding Linear layer
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        
        # freeze bottom 10 layers
        frozen_layers = tuple([layer.detach() for layer in all_hidden_states[:-2]])  
        non_frozen_layers = all_hidden_states[-2:]  # Keep the rest trainable
        
        # Combine frozen and non-frozen layers
        combined_layers = frozen_layers + non_frozen_layers

        # Use only the output of the last layer (layer 12) for further processing
        last_layer_output = combined_layers[-1]
        
        # # Pass the hidden states through the TransformerEncoder
        # transformer_outputs = self.transformer_encoder(last_layer_output)

        #pooled_output= transformer_outputs[:,0,:]
        pooled_output= last_layer_output[:,0,:]


        pooled_output=torch.squeeze(pooled_output,dim=1)

        #print('p-shape:', pooled_output.shape)

        logits = self.linear(pooled_output)

        #print('l-shape:', logits.shape)
        return logits


model = CustomMBERTModel(num_labels=2)
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')

model.to(device)
print("")


# Dataset Preparation

max_len = 128  # Maximum sequence length

# Create training, validation, and test datasets

train_dataset = CustomDataset(t_com['Sentence'].values, t_com['Tag'].values, tokenizer, max_len)
#train_dataset = CustomDataset(eng_com['Sentence'].values, eng_com['Tag'].values, tokenizer, max_len)
#train_dataset = CustomDataset(hind_com['Sentence'].values, hind_com['Tag'].values, tokenizer, max_len)
val_dataset = CustomDataset(val_df['Sentence'].values, val_df['Tag'].values, tokenizer, max_len)
test_dataset = CustomDataset(test_df['Sentence'].values, test_df['Tag'].values, tokenizer, max_len)
#test_dataset = CustomDataset(test_dt['Sentence'].values, test_dt['Tag'].values, tokenizer, max_len)


# Creating Dataloader

batch_size = 64
epochs = 50
learning_rate = 2e-5

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)


# Training

a=2800
b=2800
weightage= torch.tensor([(a/(a+b))*2, (b/(a+b))*2]).to(device)

criterion = nn.CrossEntropyLoss(weight=weightage)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_val_f1_score = 0
patience = 4  # Number of epochs to wait for improvement
counter = 0  # Counter to keep track of epochs without improvement


#best_model_save_path
tempdir = '/data1/aakash/Codemix/Aakash_02/.model/'
best_model_params_path = os.path.join(tempdir, f"best_model_mbert(cm_ratio)_Com.pt")


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask=attention_mask)  #([64,2])
        
        loss = criterion(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    print(f'Epoch {epoch + 1}/{epochs}')
    logging.info(f'Train loss: {avg_train_loss:.4f}')

    train_losses.append(avg_train_loss)


    # Validation

    # val_weightage=torch.tensor([249/(437+249),437/(438+247)]).to(device)

    # criterion = nn.CrossEntropyLoss(weight=val_weightage)
    criterion = nn.CrossEntropyLoss()


    model.eval()
    val_predictions = []
    val_labels = []
    val_loss = 0

    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        predicted_labels = torch.argmax(logits, dim=1)

        val_predictions.extend(predicted_labels.detach().cpu().numpy())
        val_labels.extend(labels.detach().cpu().numpy())
        
        val_loss += criterion(logits, labels).item()

    epoch_val_accuracy = accuracy_score(val_labels, val_predictions)
    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation accuracy: {epoch_val_accuracy:.4f}')
    logging.info(f'Validation loss: {avg_val_loss:.4f}')

    val_f1_score = f1_score(val_labels, val_predictions, average='macro')

    classification_report_epoch = classification_report(val_labels, val_predictions)
    logging.info(f'Classification Report per Epoch {epoch+1}:')
    logging.info(classification_report_epoch)
    
    val_losses.append(avg_val_loss)

    # Early stopping

    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     torch.save(model.state_dict(), best_model_params_path) # Saving best model
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print(f'Early stopping at epoch {epoch + 1}')
    #         break

    if val_f1_score > best_val_f1_score:
        best_val_f1_score = val_f1_score
        torch.save(model.state_dict(), best_model_params_path)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    scheduler.step()


# Plotting the train and validation loss

# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# Evaluation

total_eval_accuracy = 0
predictions = []
true_labels = []

best_model = CustomMBERTModel(num_labels=2).to(device)
best_model.load_state_dict(torch.load(best_model_params_path))
best_model.eval()

for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        logits = best_model(input_ids, attention_mask=attention_mask)

    #logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1)
    accuracy = (predicted_labels == labels).float().mean()
    total_eval_accuracy += accuracy.item()

    predictions.extend(predicted_labels.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

avg_test_accuracy = total_eval_accuracy / len(test_dataloader)
logging.info(f'Test accuracy: {avg_test_accuracy:.4f}')


classification_report_output = classification_report(true_labels, predictions)
logging.info('Classification Report:')
logging.info(classification_report_output)

# Append the classification report to output.txt
with open('/data1/aakash/Codemix/Aakash_02/bash/output.txt', 'a') as f:
    f.write(f"0indicBERT_without_Trans_Combined_exp3_vcod_{train_lang}.py \n")
    f.write(classification_report_output)
    f.write('\n')  # Add a newline for separation


# # Create a DataFrame from the list
# df = pd.DataFrame({'Predicted Labels': predictions})

# # Save the DataFrame to a CSV file
# output_filename = 'predicted_labels_combined(mbert_without_trans_1400).csv'
# df.to_csv(output_filename, index=False)
