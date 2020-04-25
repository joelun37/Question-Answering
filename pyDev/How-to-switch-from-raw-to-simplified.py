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

with open(example_txt, 'r') as f:
    example_dict = json.load(f)

simplfied_ex = simplify_nq_example(example_dict)

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
