#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import jsonify
from flask import request
from flask import Flask, render_template

from threading import Thread
import time

from datetime import datetime
import json

import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
#app.config['JSON_AS_ASCII'] = False

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')

results = []
id = 0

def submit(LONG_TEXT, id, sent_no, max_words):
    article_input_ids = tokenizer.batch_encode_plus([LONG_TEXT], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
    summary_ids = model.generate(input_ids=article_input_ids,
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=max_words,
                                 #min_len=56,
                                 no_repeat_ngram_size=3)

    #summary_ids = model.generate(input_ids=article_input_ids, num_beams=5, num_return_sequences=1, temperature=1.5)
    #outputs = model.generate(input_ids=article_input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)
    #summary_txt =""
    #for i in range(3): #  3 output sequences were generated
    #    print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
    #    summary_txt += tokenizer.decode(outputs[i], skip_special_tokens=True)
    
    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    global results
    del results[id-1];
    results.insert(id-1, summary_txt);
          
@app.route('/')
def about():
    return render_template('index.html')

@app.route('/tldr/', methods=["POST"])
def tldr():
    parameter = request.form['txt']
    sent_no = int(request.form['sent_no'])
    max_words = int(request.form['max_words'])
    #parameter = request.args.get('txt')
    #sent_no = 3
    #max_words = 142
        
    LONG_TEXT = parameter.replace('\n','')
    global id
    global results
    id = id + 1
    results.append("processing...") # waiting for thread
    #submit(LONG_TEXT,id)
    
    thread = Thread(target=submit, args=(LONG_TEXT,id,sent_no,max_words,))
    thread.daemon = True
    thread.start()
    return "your token is " + str(id-1)

@app.route('/tkn', methods=["GET"])
def token():
    parameter = int(request.args['token'])
    if(parameter < 0 or parameter >= len(results)):
        return "Invalid Token"
    else:
        return results[int(parameter)]

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
