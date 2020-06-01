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



a = [1, 2, 3, 0]                                                           
b= [2, 3, 3, 0]                                                           
c = [1, 1, 1, 0, 5, 0]                                                         
d= [1, 1, 1, 0, 6, 0]

a = np.array(a)                                                        
b = np.array(b) 
c = np.array(c)                                                        
d = np.array(d)  


np.sum(c==d)

np.sum(c[np.argwhere(d==0)] == 0)


np.sum(b> a)
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

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_accuracy_start_end(start_preds, end_preds, start_pos, end_pos):
    start_pred_flat = np.argmax(start_preds, axis=1).flatten()
    end_pred_flat = np.argmax(end_preds, axis=1).flatten()
    start_pos_flat = start_pos.flatten()
    end_pos_flat = end_pos.flatten()
    return np.sum((start_pred_flat == start_pos_flat) & (end_pred_flat == end_pos_flat)) / len(start_pos_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Set parameters and paths
pyData = "/Volumes/750GB-HDD/root/Question-Answering/pyData/"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')


# Import data
nq_train_txt_short = pyData + "tensorflow2-question-answering/simplified-nq-train-shorter.json"

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
        print(f"Question: {parts_dict[key]['question_text']}")
        print(f"Passage: {parts_dict[key]['passages'][i]}")
        




input_ids = []
attention_masks = []
token_type_ids = []
start_end_labels = []

for key in parts_dict.keys():
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
            start_end_labels.append(torch.tensor([[np.min(pos_answer), np.max(pos_answer)]]))
        else:
            start_end_labels.append(torch.zeros(1, 2).long())



# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
token_type_ids = torch.cat(token_type_ids, dim=0)
start_end_labels = torch.cat(start_end_labels, dim=0)

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, token_type_ids, start_end_labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.7 * len(dataset))
val_size =  int(0.2 * len(dataset))
test_size = len(dataset) - val_size - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))
print('{:>5,} test samples'.format(test_size))

batch_size = 10
epochs = 4

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

seed_val = 42
random.seed(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0]
        b_attention_masks = batch[1]
        b_token_type_ids = batch[2]
        b_start_positions = batch[3][:, 0]
        b_end_positions = batch[3][:, 1]

        model.zero_grad()

    # input_ids=None,
    #         attention_mask=None,
    #         token_type_ids=None,
    #         position_ids=None,
    #         head_mask=None,
    #         inputs_embeds=None,
    #         start_positions=None,
    #         end_positions=None,
    #     ):

        loss, start_logits, end_logits = model(input_ids=b_input_ids, 
                                            token_type_ids=b_token_type_ids, 
                                            attention_mask=b_attention_masks, 
                                            start_positions=b_start_positions,
                                            end_positions=b_end_positions
                                            )

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    # Tracking variables 
    total_start_accuracy = 0
    total_end_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        
        b_input_ids = batch[0]
        b_attention_masks = batch[1]
        b_token_type_ids = batch[2]
        b_start_positions = batch[3][:, 0]
        b_end_positions = batch[3][:, 1]
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            loss, start_logits, end_logits = model(input_ids=b_input_ids, 
                                                token_type_ids=b_token_type_ids, 
                                                attention_mask=b_attention_masks, 
                                                start_positions=b_start_positions,
                                                end_positions=b_end_positions
                                                )
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        # logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_start_accuracy += flat_accuracy(start_logits.detach().numpy(), b_start_positions.detach().numpy())
        total_end_accuracy += flat_accuracy(end_logits.detach().numpy(), b_end_positions.detach().numpy())

    # Report the final accuracy for this validation run.
    # avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

    avg_start_accuracy = total_start_accuracy / len(validation_dataloader)
    avg_end_accuracy = total_end_accuracy / len(validation_dataloader)
    print("  Start Accuracy: {0:.2f}".format(avg_start_accuracy))
    print("  End Accuracy: {0:.2f}".format(avg_end_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Start Accur.': avg_start_accuracy,
            'Valid. End Accur.': avg_end_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


import pandas as pd

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
df_stats

import matplotlib.pyplot as plt

import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.savefig(pyData+"plot_loss.png")