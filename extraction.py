import nltk
import os
import subprocess
import code
import glob
import re
import traceback
import sys
import inspect
from pprint import pprint
import json
from convertPDFToText import convertPDFToText
import re
from nltk.corpus import stopwords
import re
import spacy
from nltk.corpus import stopwords
import pandas as pd

information = []
inputString = ''
tokens = []
lines = []
sentences = []
info = []
education = []
skill = []
name = None
email = []
phone = []
filename = None

# load pre-trained model
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
# Grad all general stop words
STOPWORDS = set(stopwords.words('english'))

# Education Degrees
EDUCATION = [

    'SSC', 'HSSC', 'ICS', 'BSCS', 'BSEE', 'BSSE',
    'BBA', 'BIT', 'BSME', 'BSCE', 'PHARM-D', 'DPT',
    'B.TECH', 'BTECH', 'BS(CS)', 'BS(EE)', 'BS(SE)''BS(ME)',
    'BS(CE)', 'MSCS', 'MSEE', 'MSSE', 'MSCE', 'MBA',
    'MTECH', 'M.TECH', 'MSME', 'MS(CS)', 'MS(EE)', 'MS(SE)',
    'MS(CE)', 'MBA', 'MTECH', 'M.TECH', 'MS(ME)', 'Ph.D.'
]


def preprocess(document):
    '''
    Information Extraction: Preprocess a document with the necessary POS tagging.
    Returns three lists, one with tokens, one with POS tagged lines, one with POS tagged sentences.
    Modules required: nltk
    '''

    try:
        # Try to get rid of special characters

        try:

            document = document.decode('ascii', 'ignore')

        except:

            document = document.encode('ascii', 'ignore')

        # Newlines are one element of structure in the data
        # Helps limit the context and breaks up the data as is intended in resumes - i.e., into points

        lines = [el.strip() for el in document.split("\n") if len(el) > 0]
        # Splitting on the basis of newlines
        # Tokenize the individual lines
        lines = [nltk.word_tokenize(el) for el in lines]
        lines = [nltk.pos_tag(el) for el in lines]  # Tag them
        # Below approach is slightly different because it splits sentences not just on the basis of newlines, but also full stops
        # - (barring abbreviations etc.)
        # But it fails miserably at predicting names, so currently using it only for tokenization of the whole document
        # Split/Tokenize into sentences (List of strings)
        sentences = nltk.sent_tokenize(document)
        # Split/Tokenize sentences into words (List of lists of strings)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        tokens = sentences
        # Tag the tokens - list of lists of tuples - each tuple is (<word>, <tag>)
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        # Next 4 lines convert tokens from a list of list of strings to a list of strings; basically stitches them together
        dummy = []
        for el in tokens:
            dummy += el
        tokens = dummy
        # tokens - words extracted from the doc, lines - split only based on newlines (may have more than one sentence)
        # sentences - split on the basis of rules of grammar
        return tokens, lines, sentences
    except Exception as e:
        print(e)


def tokenize(inputString):
    try:
        tokens, lines, sentences = preprocess(inputString)
        return tokens, lines, sentences
    except Exception as e:
        print(e)


def getName(inputString, infoDict, debug=False):
    '''
    Given an input string, returns possible matches for names. Uses regular expression based matching.
    Needs an input string, a dictionary where values are being stored, and an optional parameter for debugging.
    Modules required: clock from time, code.
    '''

    # Reads Indian Names from the file, reduce all to lower case for easy comparision [Name lists]
    indianNames = open("allNames.txt", "r").read().lower()
    # Lookup in a set is much faster
    indianNames = set(indianNames.split())
    cunk = None

    otherNameHits = []
    nameHits = []
    name = None

    try:
        #tokens, lines, sentences = preprocess(inputString)
        lines = [el.strip() for el in inputString.split("\n") if len(el) > 0]
        # Splitting on the basis of newlines
        # Tokenize the individual lines
        lines = [nltk.word_tokenize(el) for el in lines]
        lines = [nltk.pos_tag(el) for el in lines]  # Tag them
        # Below approach is slightly different because it splits sentences not just on the basis of newlines, but also full stops
        # - (barring abbreviations etc.)
        # But it fails miserably at predicting names, so currently using it only for tokenization of the whole document
        # Split/Tokenize into sentences (List of strings)
        sentences = nltk.sent_tokenize(inputString)
        # Split/Tokenize sentences into words (List of lists of strings)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        tokens = sentences
        # Tag the tokens - list of lists of tuples - each tuple is (<word>, <tag>)
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        # Next 4 lines convert tokens from a list of list of strings to a list of strings; basically stitches them together
        dummy = []
        for el in tokens:
            dummy += el
        tokens = dummy
        # Try a regex chunk parser
        # grammar = r'NAME: {<NN.*><NN.*>|<NN.*><NN.*><NN.*>}'
        grammar = r'NAME: {<NN.*><NN.*><NN.*>*}'
        # Noun phrase chunk is made out of two or three tags of type NN. (ie NN, NNP etc.) - typical of a name. {2,3} won't work, hence the syntax
        # Note the correction to the rule. Change has been made later.
        chunkParser = nltk.RegexpParser(grammar)
        all_chunked_tokens = []
        for tagged_tokens in lines:
            # Creates a parse tree
            if len(tagged_tokens) == 0:
                continue  # Prevent it from printing warnings
            chunked_tokens = chunkParser.parse(tagged_tokens)
            all_chunked_tokens.append(chunked_tokens)
            cunk = all_chunked_tokens
            for subtree in chunked_tokens.subtrees():
                #  or subtree.label() == 'S' include in if condition if required
                if subtree.label() == 'NAME':
                    for ind, leaf in enumerate(subtree.leaves()):
                        if leaf[0].lower() in indianNames and 'NN' in leaf[1]:
                            # Case insensitive matching, as indianNames have names in lowercase
                            # Take only noun-tagged tokens
                            # Surname is not in the name list, hence if match is achieved add all noun-type tokens
                            # Pick upto 3 noun entities
                            hit = " ".join(
                                [el[0] for el in subtree.leaves()[ind:ind+3]])
                            # Check for the presence of commas, colons, digits - usually markers of non-named entities
                            if re.compile(r'[\d,:]').search(hit):
                                continue
                            nameHits.append(hit)
                            print(nameHits)
                            # Need to iterate through rest of the leaves because of possible mis-matches
        # Going for the first name hit
        if len(nameHits) > 0:
            nameHits = [re.sub(r'[^a-zA-Z \-]', '', el).strip()
                        for el in nameHits]
            name = " ".join([el[0].upper()+el[1:].lower()
                             for el in nameHits[0].split() if len(el) > 0])
            otherNameHits = nameHits[1:]

    except Exception as e:
        print(traceback.format_exc())
        print(e)

    infoDict['name'] = name

    return name, otherNameHits, cunk


def getExperience(inputString, infoDict, debug=False):
    experience = []
    try:
        #tokens, lines, sentences = preprocess(inputString)
        lines = [el.strip() for el in inputString.split("\n") if len(el) > 0]
        # Splitting on the basis of newlines
        # Tokenize the individual lines
        lines = [nltk.word_tokenize(el) for el in lines]
        lines = [nltk.pos_tag(el) for el in lines]  # Tag them
        # Below approach is slightly different because it splits sentences not just on the basis of newlines, but also full stops
        # - (barring abbreviations etc.)
        # But it fails miserably at predicting names, so currently using it only for tokenization of the whole document
        # Split/Tokenize into sentences (List of strings)
        sentences = nltk.sent_tokenize(inputString)
        # Split/Tokenize sentences into words (List of lists of strings)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        tokens = sentences
        sentences = [nltk.pos_tag(sent) for sent in sentences]

        for sentence in lines:  # find the index of the sentence where the degree is find and then analyse that sentence
            # string of words in sentence
            sen = " ".join([words[0].lower() for words in sentence])
            if re.search('experience', sen):
                sen_tokenised = nltk.word_tokenize(sen)
                tagged = nltk.pos_tag(sen_tokenised)
                entities = nltk.chunk.ne_chunk(tagged)
                for subtree in entities.subtrees():
                    for leaf in subtree.leaves():
                        if leaf[1] == 'CD':
                            experience = leaf[0]
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    if experience:
        infoDict['experience'] = experience
    else:
        infoDict['experience'] = 0
    if debug:
        print("\n", pprint(infoDict), "\n")
        code.interact(local=locals())
    return experience


def getQualification(inputString, infoDict, D1, D2):
    # key=list(qualification.keys())
    qualification = {'institute': '', 'year': ''}
    # open file which contains keywords like institutes,university usually  fond in institute names
    nameofinstitutes = open('nameofinstitutes.txt', 'r').read().lower()
    nameofinstitues = set(nameofinstitutes.split())
    instiregex = r'INSTI: {<DT.>?<NNP.*>+<IN.*>?<NNP.*>?}'
    chunkParser = nltk.RegexpParser(instiregex)

    try:
        #tokens, lines, sentences = preprocess(inputString)
        lines = [el.strip() for el in inputString.split("\n") if len(el) > 0]
        # Splitting on the basis of newlines
        # Tokenize the individual lines
        lines = [nltk.word_tokenize(el) for el in lines]
        lines = [nltk.pos_tag(el) for el in lines]  # Tag them
        # Below approach is slightly different because it splits sentences not just on the basis of newlines, but also full stops
        # - (barring abbreviations etc.)
        # But it fails miserably at predicting names, so currently using it only for tokenization of the whole document
        # Split/Tokenize into sentences (List of strings)
        sentences = nltk.sent_tokenize(inputString)
        # Split/Tokenize sentences into words (List of lists of strings)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        tokens = sentences
        sentences = [nltk.pos_tag(sent) for sent in sentences]

        index = []
        line = []  # saves all the lines where it finds the word of that education
        # find the index of the sentence where the degree is find and then analyse that sentence
        for ind, sentence in enumerate(lines):
            sen = " ".join([words[0].lower()
                            for words in sentence])  # string of words
            if re.search(D1, sen) or re.search(D2, sen):
                index.append(ind)  # list of all indexes where word Ca lies
        if index:  # only finds for Ca rank and CA year if it finds the word Ca in the document

            for indextocheck in index:  # checks all nearby lines where it founds the degree word.ex-'CA'
                # checks the line with the keyword and just the next line to it
                for i in [indextocheck, indextocheck+1]:
                    try:
                        try:
                            # string of that particular line
                            wordstr = " ".join(words[0] for words in lines[i])
                        except:
                            wordstr = ""
                        # if re.search(r'\D\d{1,3}\D',wordstr.lower()) and qualification['rank']=='':
                            # qualification['rank']=re.findall(r'\D\d{1,3}\D',wordstr.lower())
                            # line.append(wordstr)
                        if re.search(r'\b[21][09][8901][0-9]', wordstr.lower()) and qualification['year'] == '':
                            qualification['year'] = re.findall(
                                r'\b[21][09][8901][0-9]', wordstr.lower())
                            line.append(wordstr)
                        # regex chunk for searching univ name
                        chunked_line = chunkParser.parse(lines[i])
                        for subtree in chunked_line.subtrees():
                            if subtree.label() == 'INSTI':
                                for ind, leaves in enumerate(subtree):
                                    if leaves[0].lower() in nameofinstitutes and leaves[1] == 'NNP' and qualification['institute'] == '':
                                        qualification['institute'] = ' '.join(
                                            [words[0]for words in subtree.leaves()])
                                        line.append(wordstr)

                    except Exception as e:
                        print(traceback.format_exc())

        if D1 == 'c\.?a':
            infoDict['%sinstitute' % D1] = "I.C.A.I"
        else:
            if qualification['institute']:
                infoDict['%sinstitute' % D1] = str(qualification['institute'])
            else:
                infoDict['%sinstitute' % D1] = "NULL"
        if qualification['year']:
            infoDict['%syear' % D1] = int(qualification['year'][0])
        else:
            infoDict['%syear' % D1] = 0
        infoDict['%sline' % D1] = list(set(line))
    except Exception as e:
        print(traceback.format_exc())
        print(e)


def Qualification(inputString, infoDict, debug=False):
    degre = []
    getQualification(inputString, infoDict, 'c\.?a', 'chartered accountant')
    if infoDict['%sline' % 'c\.?a']:
        degre.append('ca')
    getQualification(inputString, infoDict, 'icwa', 'icwa')
    if infoDict['%sline' % 'icwa']:
        degre.append('icwa')
    getQualification(inputString, infoDict, 'b\.?com', 'bachelor of commerce')
    if infoDict['%sline' % 'b\.?com']:
        degre.append('b.com')
    getQualification(inputString, infoDict, 'm\.?com', 'masters of commerce')
    if infoDict['%sline' % 'm\.?com']:
        degre.append('m.com')
    getQualification(inputString, infoDict, 'mba', 'mba')
    if infoDict['%sline' % 'mba']:
        degre.append('mba')
    if degre:
        infoDict['degree'] = degre
    else:
        infoDict['degree'] = "NONE"
    if debug:
        print("\n", pprint(infoDict), "\n")
        code.interact(local=locals())
    return infoDict['degree']


def getEmail(inputString, infoDict, debug=False):
    '''
    Given an input string, returns possible matches for emails. Uses regular expression based matching.
    Needs an input string, a dictionary where values are being stored, and an optional parameter for debugging.
    Modules required: clock from time, code.
    '''

    email = None
    try:
        pattern = re.compile(r'\S*@\S*')
        # Gets all email addresses as a list
        matches = pattern.findall(inputString)
        email = matches
    except Exception as e:
        print(e)

    infoDict['email'] = email

    if debug:
        print("\n", pprint(infoDict), "\n")
        code.interact(local=locals())
    return email


def getPhone(inputString, infoDict, debug=False):
    '''
    Given an input string, returns possible matches for phone numbers. Uses regular expression based matching.
    Needs an input string, a dictionary where values are being stored, and an optional parameter for debugging.
    Modules required: clock from time, code.
    '''

    number = None
    try:
        pattern = re.compile(
            r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\-]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')
        # Understanding the above regex
        # +91 or (91) -> [+(]? \d+ -?
        # Metacharacters have to be escaped with \ outside of character classes; inside only hyphen has to be escaped
        # hyphen has to be escaped inside the character class if you're not incidication a range
        # General number formats are 123 456 7890 or 12345 67890 or 1234567890 or 123-456-7890, hence 3 or more digits
        # Amendment to above - some also have (0000) 00 00 00 kind of format
        # \s* is any whitespace character - careful, use [ \t\r\f\v]* instead since newlines are trouble
        match = pattern.findall(inputString)
        # match = [re.sub(r'\s', '', el) for el in match]
        # Get rid of random whitespaces - helps with getting rid of 6 digits or fewer (e.g. pin codes) strings
        # substitute the characters we don't want just for the purpose of checking
        match = [re.sub(r'[,.]', '', el)
                 for el in match if len(re.sub(r'[()\-.,\s+]', '', el)) > 6]
        # Taking care of years, eg. 2001-2004 etc.
        match = [re.sub(r'\D$', '', el).strip() for el in match]
        # $ matches end of string. This takes care of random trailing non-digit characters. \D is non-digit characters
        match = [el for el in match if len(re.sub(r'\D', '', el)) <= 15]
        # Remove number strings that are greater than 15 digits
        try:
            for el in list(match):
                # Create a copy of the list since you're iterating over it
                if len(el.split('-')) > 3:
                    continue  # Year format YYYY-MM-DD
                for x in el.split("-"):
                    try:
                        # Error catching is necessary because of possibility of stray non-number characters
                        # if int(re.sub(r'\D', '', x.strip())) in range(1900, 2100):
                        if x.strip()[-4:].isdigit():
                            if int(x.strip()[-4:]) in range(1900, 2100):
                                # Don't combine the two if statements to avoid a type conversion error
                                match.remove(el)
                    except:
                        pass
        except:
            pass
        number = match
    except:
        pass

    infoDict['phone'] = number

    if debug:
        print("\n", pprint(infoDict), "\n")
        code.interact(local=locals())
    return number


def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.string.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education


def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]

    # reading the csv file
    data = pd.read_csv("skills.csv")

    # extract values
    skills = list(data.columns.values)

    skillset = []

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def get_data(path):
    pdf_files = glob.glob("resuma/*.pdf")
    files = pdf_files
    files = list(files)
    #print("%d files identified" % len(files))
    info = {}
    getstring = 'resuma/'+str(path)
    f = str(getstring)
    print(f)
    # for f in sorted(files):
    #     print("Reading File %s" % f)# info is a dictionary that stores all the data obtained from parsing

    inputString = convertPDFToText(f)
    info['extension'] = 'pdf'
    info['fileName'] = f
    filename = info['fileName']

    tokenize(inputString)
    email = getEmail(inputString, info)
    phone = getPhone(inputString, info)
    getName(inputString, info)
    experience = getExperience(inputString, info)
    education = extract_education(inputString)
    skill = extract_skills(inputString)
    name = info['name']
    return name, email, phone, education, skill, experience