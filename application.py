#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import jsonify
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm

from threading import Thread
import time
import os
from os.path import join, dirname, realpath

from datetime import datetime
import json

import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

import pytesseract
import cv2
import numpy as np

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

HOSTNAME = '83.212.102.161'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
#app.config['JSON_AS_ASCII'] = False
bootstrap = Bootstrap(app)

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')

results = []
articles = []
id = 0

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def submit(LONG_TEXT, id, max_words):
    article_input_ids = tokenizer.batch_encode_plus([LONG_TEXT], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
    summary_ids = model.generate(input_ids=article_input_ids,
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=142,
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
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/uploader', methods=["POST"])
@app.route('/uploader/', methods=["POST"])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #imPath = str(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        config = ('-l eng --oem 1 --psm 3')

        #im = cv2.imread(imPath, cv2.IMREAD_C.OLOR)

        filestr = request.files['file'].read()
        npimg = np.fromstring(filestr, np.uint8)
        im = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        text = pytesseract.image_to_string(im, config=config)
        print(text)
        return render_template('index.html', txt1=text)
        #print(url_for('upload_file',filename=filename))

@app.route('/tldr/', methods=["POST"])
def tldr():
    parameter = str(request.form['txt'])
    #sent_no = int(request.form['sent_no'])
    #max_words = int(request.form['max_words'])
    #parameter = request.args.get('txt')

    #sent_no = 3
    max_words = 142
    LONG_TEXT = parameter#.replace('\n',' ') # replace('\n', ' ') not working correct
    LONG_TEXT = " ".join(parameter.split())

    global id
    global results
    id = id + 1
    results.append("processing...") # waiting for thread
    articles.append(LONG_TEXT)
    #submit(LONG_TEXT,id)

    thread = Thread(target=submit, args=(LONG_TEXT,id,max_words,))
    thread.daemon = True
    thread.start()
    return "your token is " + str(id-1) + "<br> use it <a href=\"http://"+HOSTNAME+"/result?token="+ str(id-1) +"\">here</a>"

@app.route('/howto', methods=["GET"])
def howto():
    return render_template('howto.html')

@app.route('/result', methods=["GET"])
def result():
    parameter = int(request.args['token'])
    if(parameter < 0 or parameter >= len(results)):
        return render_template('result.html', res="Invalid Token")
    else:
        return render_template('result.html', res=results[int(parameter)])

@app.route('/article',methods=["GET"])
def get_articles():
    parameter = int(request.args['token'])
    if(parameter < 0 or parameter >= len(articles)):
        return "Invalid Token"
    else:
        return articles[int(parameter)]

if __name__ == '__main__':
    app.run(threaded=True, host=HOSTNAME, port=5000)
