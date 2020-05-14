# Third attempt

# FOR GCP
# import tensorflow as tf

# # Get the GPU device name.
# device_name = tf.test.gpu_device_name()

# # The device name should look like the following:
# if device_name == '/device:GPU:0':
#     print('Found GPU at: {}'.format(device_name))
# else:
#     raise SystemError('GPU device not found')
# import torch

# # If there's a GPU available...
# if torch.cuda.is_available():    

#     # Tell PyTorch to use the GPU.    
#     device = torch.device("cuda")

#     print('There are %d GPU(s) available.' % torch.cuda.device_count())

#     print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
# else:
#     print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")

import autoreload
%load_ext autoreload

import re
import json 
import torch
import numpy as np
import matplotlib.pyplot as plt 
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering

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

dict_obs = {}
with open(nq_train_txt_short, 'r') as f:
    for line in f:
        obs = json.loads(line)
        dict_obs[obs["example_id"]] = obs

for i,k in enumerate(dict_obs):
    print(dict_obs[k]["question_text"])

parts_dict = {}
window_size = 509 - len(obs["question_text"])
step_size = 128

"""
The loop below contains a minor bug. It does not take into account the last window
because it goes beyong the end of the paragraph.
A quick way around this issue for shorter paragraphs is to take 
shorter windows and step sizes.

Since we are dealng with long Wikipedia pages, we always set the window to 520.
As a result we do not need attention masks because our data won't contain paddings.
"""
sentence=obs["document_text"]
lst_sentence = sentence.split()
parts_dict[obs["example_id"]] = []
for i in range(len(lst_sentence)):
    if  ((window_size+(i*step_size) - 1) < len(lst_sentence)):
        part_of_sentence = [lst_sentence[word] for word in range(i*step_size, window_size+(i*step_size))]
        parts_dict[obs["example_id"]].append(" ".join(part_of_sentence))

input_ids = []

for i in range(len(parts_dict[obs["example_id"]])):
    encoded_dict = tokenizer.encode_plus(obs["question_text"],
                        parts_dict[obs["example_id"]][i],                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 520,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        # return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt'     # Return pytorch tensors.
                )
    input_ids.append(encoded_dict["input_ids"])
    # input_ids = tokenizer.encode(obs["question_text"], parts_dict[obs["example_id"]][i])

input_ids = []
input_ids.append(encoded_dict["input_ids"])

for i in range(len(input_ids)):
    print(input_ids[i].shape[1])

# Transform dictionary of observations
transformed_dict_obs = {}
parts_dict = {}
for key in dict_obs:
    transformed_dict_obs[key]={}
    obs = dict_obs[key]
    sentence=obs["document_text"]
    lst_sentence = sentence.split()
    parts_dict[obs["example_id"]] = {}
    parts_dict[obs["example_id"]]["question_text"] = obs["question_text"]
    parts_dict[obs["example_id"]]["passages"] = []
    parts_dict[obs["example_id"]]["answer_start_ix"] = []
    parts_dict[obs["example_id"]]["answer_end_ix"] = []
    la_start = dict_obs[key]["annotations"][0]["long_answer"]["start_token"]
    la_end = dict_obs[key]["annotations"][0]["long_answer"]["end_token"]
    parts_dict[obs["example_id"]]["long_answer"] = lst_sentence[la_start:la_end]
    for i in range(len(lst_sentence)):
        if  ((window_size+(i*step_size) - 1) < len(lst_sentence)):
            part_of_sentence = [lst_sentence[word] for word in range(i*step_size, window_size+(i*step_size))]
            parts_dict[obs["example_id"]]["passages"].append(" ".join(part_of_sentence))
            if (i*step_size <= la_start) and (window_size+(i*step_size) >= la_end):
                start_ix = (la_start - i*step_size)
                end_ix = (la_end - i*step_size)
            else:
                start_ix = 0
                end_ix = 0
            parts_dict[obs["example_id"]]["answer_start_ix"].append(start_ix)
            parts_dict[obs["example_id"]]["answer_end_ix"].append(end_ix)

    for k in parts_dict:
        print("\n")
        print(k)
        print("\n")
        for i in range(len(parts_dict[k]["answer_start_ix"])):
            if parts_dict[k]["answer_start_ix"][i] != 0:
                start = parts_dict[k]["answer_start_ix"][i]
                end = parts_dict[k]["answer_end_ix"][i]
                print('\n')
                print(parts_dict[k]["question_text"])
                print('\n')
                print(parts_dict[k]["passages"][i])
                print('\n')
                print(parts_dict[k]["passages"][i].split()[start:end])
                print('\n')
                print(parts_dict[k]["long_answer"])

    input_ids = []

    for i in range(len(parts_dict[obs["example_id"]])):
        encoded_dict = tokenizer.encode_plus(obs["question_text"],
                            parts_dict[obs["example_id"]][i],                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 520,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            # return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt'     # Return pytorch tensors.
                    )
        input_ids.append(encoded_dict["input_ids"])
    transformed_dict_obs[key]["input_ids"] = input_ids
    transformed_dict_obs[key]["question_text"] = obs["question_text"]
    transformed_dict_obs[key]["annotations"] = obs["annotations"]

    la_start = transformed_dict_obs[key]["annotations"][0]["long_answer"]["start_token"]
    la_end = transformed_dict_obs[key]["annotations"][0]["long_answer"]["end_token"]
    long_answer = []
    for word_ix in range(la_start, la_end):
        long_answer.append(lst_sentence[word_ix])
    transformed_dict_obs[key]["long_answer"] = long_answer
    # transformed_dict_obs[key]["input_ids"] = input_ids
    # transformed_dict_obs[key]["question_text"] = obs["question_text"]
    # transformed_dict_obs[key]["annotations"] = obs["annotations"]

for key in transformed_dict_obs.keys():
    print(transformed_dict_obs[key]["question_text"])
    print(transformed_dict_obs[key]["long_answer"])

input_ids[1].shape[1]

# add start and end token spots to the dictionary
# then try BertQuestionsAnswerin
# Then try adding the third parameter
# Hopefully, it'll suffice to add self.param maybe add to the output as well

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])

sep_index = input_ids.tolist().index(tokenizer.sep_token_id)

# The number of segment A tokens includes the [SEP] token istelf.
num_seg_a = sep_index + 1

# The remainder are segment B.
num_seg_b = len(input_ids) - num_seg_a

# Construct the list of 0s and 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)

# Run our example through the model.
start_scores, end_scores = model(encoded_dict["input_ids"], # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
