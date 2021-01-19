# Resuma Extractor

## Introduction

This project soul purpose is to extract data from a pdf file and then save the results where ever you want.I have created the both the script and the django project as well so that you can choose either one off them for your project. Just as a desclamber this code i have used to extract data from a pdf file was created by 2 saparate people i cant their repositores but if i find them i will mention them. The extracted contents from the pdf file are

- Name
- Email
- Phone No
- Education
- Experience
- Qualification
- Skills

I have also added a buch of resumas for you test them out for yourself.

# Tech

The core libraries used in this project are

- nltk
- django
- numpy
- pandas
- pdfminer

The rest of the libries are in the requirments.txt file you can just easily create a virtual env and type the command

```python
$ pip install -r requirements.txt
$ pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz#egg=en_core_web_sm==2.1.0
$ python -m spacy download en_core_web_sm
```
