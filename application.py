#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import jsonify
from flask import request
from flask import Flask, render_template

from datetime import datetime
import json

import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
#app.config['JSON_AS_ASCII'] = False

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.route('/')
def about():
    return "hi" #render_template('index.html')

@app.route('/tldr/', methods=["GET"])
def tldr():

    parameter = request.args.get('txt')
        
    LONG_TEXT = parameter.replace('\n','')

    tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')

    article_input_ids = tokenizer.batch_encode_plus([LONG_TEXT], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
    summary_ids = model.generate(input_ids=article_input_ids,
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=142,
                                 #min_len=56,
                                 no_repeat_ngram_size=3)

    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary_txt;

# main

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
