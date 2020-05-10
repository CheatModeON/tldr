# TLDR service
Too long didn't read is a micro service in flask that with the use of dnn (transformers, pytorch), that returns the summary of a text. 

# Implementation

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python

sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
pip install tesseract
pip install tesseract-ocr

git clone https://github.com/CheatModeON/tldr.git
cd tldr
pip install -r requirements.txt
python application.py

In case you don't have enough physical memory, add swap memory (https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-18-04)

# The Model

https://huggingface.co/transformers/main_classes/model.html
