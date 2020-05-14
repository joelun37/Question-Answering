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

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])


parts_dict = {}
window_size = 
step_size = 128
example_id = 1

sentence = "Sheets of empty canvas untouched sheets of clay Were laid spread out before me as her body once did All five horizons revolved around her soul as the earth to the sun Now the air I tasted and breathed has taken a turn ooh And all I taught her was everything Ooh I know she gave me all that she wore And now my bitter hands chafe beneath the clouds Of what was everything Oh, the pictures have all been washed in black Tattooed everything Give me fuel give me fire"

sentence=obs["document_text"]
lst_sentence = sentence.split()
parts_dict[obs["example_id"]] = []
for i in range(len(lst_sentence)):
    if  ((window_size+(i*step_size) - 1) < len(lst_sentence)):
        part_of_sentence = [lst_sentence[word] for word in range(i*step_size, window_size+(i*step_size))]
        parts_dict[example_id].append(" ".join(part_of_sentence))

for i, word in enumerate(sentence):
    for window in range(window_size):
        if ((i + window + 1) < len(sentence)):
            my_dict[word].append(sentence[i + window + 1])
        if ((i - window -1) > -1):
            my_dict[word].append(sentence[i - window - 1])

for i in range(len(parts_dict[-5739403430449762964])):
    print("\n")
    print(parts_dict[-5739403430449762964][i])


parts_dict = {}
window_size = 509 - len(obs["question_text"])
step_size = 128
# example_id = 1

input_ids = []

len(obs["question_text"])


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

input_ids[1].shape[1]

# add start and end token spots to the dictionary
# then try BertQuestionsAnswerin
# Then try adding the third parameter
# Hopefully, it'll suffice to add self.param maybe add to the output as well

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
