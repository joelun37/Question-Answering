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
import random
import matplotlib.pyplot as plt 
import time
import datetime


from transformers import BertTokenizer
from transformers import BertForQuestionAnswering, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# utils
def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found

# Set parameters and paths
pyData = "/Volumes/750GB-HDD/root/Question-Answering/pyData/"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model = BertModel.from_pretrained('bert-base-uncased')

# Import data
nq_train_txt_short = pyData + "tensorflow2-question-answering/simplified-nq-train-short.json"

parts_dict = {}

"""
The loop below contains a minor bug. It does not take into account the last window
because it goes beyong the end of the paragraph.
A quick way around this issue for shorter paragraphs is to take 
shorter windows and step sizes.

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
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt'     # Return pytorch tensors.
                )
    input_ids.append(encoded_dict["input_ids"])
    # input_ids = tokenizer.encode(obs["question_text"], parts_dict[obs["example_id"]][i])

input_ids = []
input_ids.append(encoded_dict["input_ids"])

for i in range(len(input_ids)):
    print(input_ids[i].shape[1])



# Transform dictionary of observations
dict_obs = {}
with open(nq_train_txt_short, 'r') as f:
    for line in f:
        obs = json.loads(line)
        dict_obs[obs["example_id"]] = obs

transformed_dict_obs = {}
step_size = 128
parts_dict = {}
obs={}
for key in dict_obs:
    if dict_obs[key]["annotations"][0]["long_answer"]["candidate_index"] != -1:
        transformed_dict_obs[key]={}
        obs = dict_obs[key]
        window_size = 509 - len(obs["question_text"])
        if dict_obs[key]["annotations"][0]["long_answer"]["end_token"] - dict_obs[key]["annotations"][0]["long_answer"]["start_token"] <= window_size:
            sentence=obs["document_text"]
            lst_sentence = sentence.split()
            # Create dictionary entry
            parts_dict[obs["example_id"]] = {}
            parts_dict[obs["example_id"]]["question_text"] = obs["question_text"]
            parts_dict[obs["example_id"]]["passages"] = []
            parts_dict[obs["example_id"]]["answer_start_ix"] = []
            parts_dict[obs["example_id"]]["answer_end_ix"] = []
            la_start = dict_obs[key]["annotations"][0]["long_answer"]["start_token"]
            la_end = dict_obs[key]["annotations"][0]["long_answer"]["end_token"]
            parts_dict[obs["example_id"]]["long_answer"] = lst_sentence[la_start:la_end]

            if len(lst_sentence) > window_size:
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
            else:
                parts_dict[obs["example_id"]]["passages"].append(obs["document_text"])
                parts_dict[obs["example_id"]]["answer_start_ix"].append(la_start)
                parts_dict[obs["example_id"]]["answer_end_ix"].append(la_end)

for key in parts_dict.keys():
    print(key)
    print(len(parts_dict[key]["answer_start_ix"]))
    print(len(parts_dict[key]["answer_end_ix"]))
    print(len(parts_dict[key]["passages"]))
        # else:
        #     print("That guy")
        #     print(f"LASTART is {la_start}")
        #     parts_dict[obs["example_id"]]["passages"].append(" ".join(lst_sentence))
        #     start_ix = la_start
        #     end_ix = la_end
        #     parts_dict[obs["example_id"]]["answer_start_ix"].append(start_ix)
        #     parts_dict[obs["example_id"]]["answer_end_ix"].append(end_ix)

    # Downsampling
for key in parts_dict.keys():
    if len(np.argwhere(parts_dict[key]["answer_start_ix"])) > 0:
        valid_ix = np.argwhere(parts_dict[key]["answer_start_ix"])
        valid_ix = valid_ix.reshape(1, -1)[0].tolist()
        all_ix = [i for i in range(len(parts_dict[key]["answer_start_ix"]))]
        null_sample_ix = np.random.choice(all_ix, len(valid_ix)).tolist()
        sample_ix = valid_ix + null_sample_ix
        print(key)
        parts_dict[key]["answer_start_ix"] = np.array(parts_dict[key]["answer_start_ix"])[sample_ix]
        parts_dict[key]["answer_end_ix"] = np.array(parts_dict[key]["answer_end_ix"])[sample_ix]
        parts_dict[key]["passages"] = np.array(parts_dict[key]["passages"])[sample_ix]
        print(key)
        print(len(parts_dict[key]["answer_start_ix"]))
        print(len(parts_dict[key]["answer_end_ix"]))
        print(len(parts_dict[key]["passages"]))

# Sanity check
for key in parts_dict.keys():
    print('\n')
    print(f"Key is {key}")
    print(parts_dict[key]["question_text"])
    print(parts_dict[key]["long_answer"])
    for i in range(len(parts_dict[key]["answer_end_ix"])):
        start = parts_dict[key]["answer_start_ix"][i]
        end = parts_dict[key]["answer_end_ix"][i]
        passage = np.array(parts_dict[key]["passages"][i].split())
        print("\n")
        print(passage[start:end])

# parts_dict[8933277615800380148].keys()
# dict_keys(['question_text', 'passages', 'answer_start_ix', 'answer_end_ix', 'long_answer'])
# parts_dict[8933277615800380148]["question_text"]

for key in parts_dict.keys():
    for i in range(len(parts_dict[key]["passages"])):
        print(f"Question: {parts_dict[key]["question_text"]}")
        print(f"Passage: {parts_dict[key]["passages"][i]}")
        




input_ids = []
attention_masks = []
token_type_ids = []
start_positions = []
end_positions = []

for key in parts_dict.keys():
    # key = 5655493461695504401
    for i in range(len(parts_dict[key]["passages"])):
        encoded_dict = tokenizer.encode_plus(parts_dict[key]["question_text"],
                                             parts_dict[key]["passages"][i],                      # Sentence to encode.
                                             add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                             max_length = 512,           # Pad & truncate all sentences.
                                             pad_to_max_length = True,
                                             return_attention_mask = True,   # Construct attn. masks.
                                             return_tensors = 'pt'     # Return pytorch tensors.
                                        )
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        # Token type ids
        token_type_ids.append(encoded_dict['token_type_ids'])

        # Tokenize ans and inputs
        tokenized_input = np.array(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'].tolist()[0]))
        tokenized_long_answer = np.array(tokenizer.tokenize(" ".join(parts_dict[key]["long_answer"])))

        # Search answer in the input
        pos_answer = search_sequence_numpy(tokenized_input, tokenized_long_answer)

        # Start positions
        if len(pos_answer) > 0:
            start_positions.append(np.min(pos_answer))
        else:
            start_positions.append(0)

        # End positions
        if len(pos_answer) > 0:
            end_positions.append(np.max(pos_answer))
        else:
            end_positions.append(0)
            
ans = np.array(tokenizer.tokenize(" ".join(parts_dict[5655493461695504401]["long_answer"])))
ans_start = np.min(search_sequence_numpy(tokens, ans))
ans_end = np.max(search_sequence_numpy(tokens, ans))

np.array(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[0]))[ans_start:ans_end]




# print(np.array(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[0])))
# print(np.array(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[0]))[288:355])
print(" ".join(parts_dict[5655493461695504401]["long_answer"]))
# print(input_ids[0].tolist()[0])
# print(start_positions[0])
# print(end_positions[0])
# a = torch.tensor(np.array([1, 2]))

# encoded_dict["attention_mask"].size()

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)


# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# model = BertForQuestionAnswering.from_pretrained(
#     "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
# )
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')


# input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         start_positions=None,
#         end_positions=None,
#     ):

# CREATE start and end position tensors
# Token types are important: sentence one and sentence two

start_scores, end_scores = model(input_ids, 
                             token_type_ids=None, 
                             attention_mask=attention_masks, 
                             start_positions=)

# Tell pytorch to run this model on the GPU.
# model.cuda()

params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()


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
