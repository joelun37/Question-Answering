"""
In this file, we show how to use the simplify_nq_example function
created by Google AI team.

We import the json file as a dictionary using the JSON package

Input:

- example_txt: is the file that we create from the development set, it is the 
               first line of this set


Output:

- simplified_ex: it's a dictionary in the simplified data format.
"""

import json
example_txt = "/Volumes/750GB-HDD/root/Question-Answering/pyData/tensorflow2-question-answering/simplified-nq-train-for-content.json"
dev_example = "/Volumes/750GB-HDD/root/Question-Answering/pyData/tensorflow2-question-answering/v1.0-simplified_nq-dev-all-for-content.json"

def raw_NQ_data_dict(input_text_file):

    with open(input_text_file, 'r') as f:
        for line in f:
            example_dict = json.loads(line) 
            simplfied_ex =   (example_dict)

    return simplfied_ex

test_dict = raw_NQ_data_dict(input_text_file=example_txt)

dict = raw_NQ_data_dict(input_text_file=dev_example)

def test_simplify_nq_example(input_text_file):

    with open(input_text_file, 'r') as f:
        example_dict = json.load(f) 
    simplfied_ex = simplify_nq_example(example_dict)
    return simplfied_ex

simplified_dev_example = test_simplify_nq_example(dev_example)

my_list = txt.split(" ")

long_answer = []
for i in range(1952,2019):
    long_answer.append(my_list[i])

txt_long_answer = ""
for i in range(len(long_answer)):
    txt_long_answer += " " + long_answer[i]


short_answer = []
for i in range(1960,1969):
    short_answer.append(my_list[i])

txt_short_answer = ""
for i in range(len(short_answer)):
    txt_short_answer += " " + short_answer[i]


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = test_dict["document_text"]
question = test_dict["question_text"]

inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")

