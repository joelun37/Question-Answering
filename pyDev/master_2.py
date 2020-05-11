# Libraries
import autoreload
%load_ext autoreload

import re
import json 
import torch
import numpy as np
import matplotlib.pyplot as plt 
from transformers import BertTokenizer
from transformers import BertModel

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set parameters and paths
pyData = "/Volumes/750GB-HDD/root/Question-Answering/pyData/"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Import data

nq_train_txt_short = pyData + "tensorflow2-question-answering/simplified-nq-train-short.json"

# nq_train_txt_short_dict = raw_NQ_data_dict(nq_train_txt_short)

# nq_train_txt_short_dict["annotations"]

dict_obs = {}
dict_obs["question"] = []
dict_obs["long_answer"] = []
dict_obs["transformed_question"] = []
dict_obs["start_transformed_long_answer"] = []
dict_obs["end_transformed_long_answer"] = []

with open(nq_train_txt_short, 'r') as f:
    counter = 0
    for line in f:
        obs = json.loads(line)
        long_answer_candidate_index = obs["annotations"][0]["long_answer"]["candidate_index"]
        long_answer_candidate_index = obs["annotations"][0]["long_answer"]["candidate_index"]
        long_answer_start = obs["long_answer_candidates"][long_answer_candidate_index]["start_token"]
        long_answer_end = obs["long_answer_candidates"][long_answer_candidate_index]["end_token"]
        long_answer =""
        
        for i in range(long_answer_start, long_answer_end):
            long_answer += obs["document_text"].split()[i] + " "
        
        if len(remove_html_tags(long_answer)) <= 512:
            dict_obs["question"].append(obs["question_text"])
            dict_obs["long_answer"].append(remove_html_tags(long_answer))

            question_inputs = tokenizer.encode_plus(obs["question_text"], add_special_tokens=True, return_tensors="pt")
            question_input_ids = question_inputs["input_ids"].tolist()[0]
            question_text_tokens = tokenizer.convert_ids_to_tokens(question_input_ids)

            long_answer_inputs = tokenizer.encode_plus(remove_html_tags(long_answer), add_special_tokens=True, return_tensors="pt")
            long_answer_input_ids = long_answer_inputs["input_ids"].tolist()[0]
            long_answer_text_tokens = tokenizer.convert_ids_to_tokens(long_answer_input_ids)

            question_outputs = model(**question_inputs)
            long_answer_outputs = model(**long_answer_inputs)

           # Last hidden states
            question_last_hidden_states = question_outputs[0].detach().numpy().reshape(len(question_text_tokens), 768)
            long_answer_last_hidden_states = long_answer_outputs[0].detach().numpy().reshape(len(long_answer_text_tokens), 768)

           # Transformed
            transformed_question = question_last_hidden_states[0]
            start_transformed_long_answer = long_answer_last_hidden_states[0]
            end_transformed_long_answer = long_answer_last_hidden_states[len(long_answer_text_tokens) - 1]

            # Append transformed texts
            dict_obs["transformed_question"].append(transformed_question)
            dict_obs["start_transformed_long_answer"].append(start_transformed_long_answer)
            dict_obs["end_transformed_long_answer"].append(end_transformed_long_answer)

        # print(f"\n \n Question text: {obs['question_text']} \n  Length: {len(remove_html_tags(long_answer))} \n Long answer: {remove_html_tags(long_answer)} \n \n ")

        if len(remove_html_tags(long_answer)) <= 512:
            counter += 1

    print(f"The number of reasonably long answers is {counter}")


# Create X
X = dict_obs["transformed_question"][0].reshape(1, -1)
for q in range(1, len(dict_obs["transformed_question"])):
  X = np.vstack((X, dict_obs["transformed_question"][q].reshape(1, -1)))

# 44 axes explains 70%
reduced_X = reduce_to_k_dim_PCA(X, k=295)

# Create y1
y1 = dict_obs["start_transformed_long_answer"][0].reshape(1, -1)
for q in range(1, len(dict_obs["start_transformed_long_answer"])):
  y1 = np.vstack((y1, dict_obs["start_transformed_long_answer"][q].reshape(1, -1)))

# 41 axes explain 70%
reduced_y1 = reduce_to_k_dim_PCA(y1, k=295)

# Create y2
y2 = dict_obs["end_transformed_long_answer"][0].reshape(1, -1)
for q in range(1, len(dict_obs["end_transformed_long_answer"])):
  y2 = np.vstack((y2, dict_obs["end_transformed_long_answer"][q].reshape(1, -1)))

# 50 axes explain 85%
reduced_y2 = reduce_to_k_dim_PCA(y2, k=295)


for i in range(10):
    print(f"Question: {dict_obs['question'][i]} \n Transformed Question: \n {dict_obs['transformed_question'][i][0:10]}")

dict_obs['transformed_question'][0] == dict_obs['transformed_question'][1]


# Create y_start

len(dict_obs["long_answer"])

# Set text
question = dict_obs["question"][0]
print(question)

# Tokenize text
inputs = tokenizer.encode_plus(question, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Convert inputs to tensor
# input_ids_tensor = torch.tensor(inputs).unsqueeze(0)

# Model
outputs = model(inputs)

# Last hidden states
last_hidden_states = outputs[0].detach().numpy().reshape(len(text_tokens), 768)

questions = last_hidden_states[0]

