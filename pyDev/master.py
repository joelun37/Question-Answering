# This is the master file
# Libraries
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Mock test
# We are going to get 2 long answer candidates with the real long answer
# and feed it to the BERT model
long_answer_candidates = test_dict["long_answer_candidates"]
document_text = test_dict["document_text"]
question_text = test_dict["question_text"]

# [{'yes_no_answer': 'NONE',
#   'long_answer': {'start_token': 1952,
#    'candidate_index': 54,
#    'end_token': 2019},
#   'short_answers': [{'start_token': 1960, 'end_token': 1969}],
#   'annotation_id': 593165450220027640}]
annotations = test_dict["annotations"]

long_answer = ""
candidate_index = annotations[0]["long_answer"]["candidate_index"]

for i in range(long_answer_candidates[candidate_index]["start_token"], \
               long_answer_candidates[candidate_index]["end_token"]):
    long_answer += document_text.split()[i] + " "

short_answer = ""
for i in range(annotations[0]["short_answers"][0]["start_token"], \
               annotations[0]["short_answers"][0]["end_token"]):
    short_answer += document_text.split()[i] + " "

# First two candidates
candidate_dict = {}

i = 0
while len(candidate_dict.keys()) < 2:
    if long_answer_candidates[i]["top_level"] == True:
        txt = ""
        for j in range(long_answer_candidates[i]["start_token"], \
                       long_answer_candidates[i]["end_token"]):
            txt += document_text.split()[j] + " "
        candidate_dict[i] = txt
    i += 1

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = ""
for key in candidate_dict.keys():
    text += remove_html_tags(candidate_dict[key])

inputs = tokenizer.encode_plus(question_text, text, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer_start_scores, answer_end_scores = model(**inputs)
answer_start = torch.argmax(
    answer_start_scores
)  # Get the most likely beginning of answer with the argmax of the score
answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
print(f"Question: {question_text}")
print(f"Answer: {answer}\n")

# Bare BERT
from transformers import BertTokenizer
from transformers import BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer.encode_plus(question_text, remove_html_tags(text), add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

input_ids = torch.tensor(tokenizer.encode("golden")).unsqueeze(0)  # Batch size 1

text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

outputs = model(inputs)
last_hidden_states = outputs[0].detach().numpy()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

# Set text
my_text = "golden"
my_second_text = "sciences"
my_third_text = "concert"

# Tokenize text
inputs = tokenizer.encode_plus(my_text, my_second_text+" "+my_third_text, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Convert inputs to tensor
input_ids_tensor = torch.tensor(inputs).unsqueeze(0)

# Model
outputs = model(**inputs)

# Last hidden states
last_hidden_states = outputs[0]
last_hidden_states_np = last_hidden_states.detach().numpy().reshape(6, 768)

