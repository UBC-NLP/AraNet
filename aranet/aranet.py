# encoding: utf-8
import optparse
import os, json
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification


class AraNet():
    '''
    '''

    def __init__(self, path):
        # check if gpu is available
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n___gpu = torch.cuda.device_count()

        # check if the path exists
        if path is None:
            raise Exception('Undefined path to model')
        if not os.path.exists(path):
            raise Exception("Couldn't find the path %s" %path)
        self.__path = path

        # check if the path contains model directory
        model_path = os.path.join(self.__path, 'model')
        if not os.path.exists(model_path):
            raise Exception("Couldn't find the path %s" %model_path)

        # load the model
        try:
            self.__tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
            self.__model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)
            
            self.__model = self.__model.to(self.__device)
            
        except Exception as e:
            raise Exception ("Couldn't load the model", e)

        # load the labels dictionary
        dict_path = os.path.join(self.__path, 'labels-dict.json')
        if not os.path.exists(dict_path):
            raise Exception("Couldn't find the path %s" %dict_path)
        with open(dict_path) as json_file:
            self.__lab2ind = json.load(json_file)
        self.__ind2lab = {}
        for label in self.__lab2ind.keys():
            self.__ind2lab[self.__lab2ind[label]] = label



    def predict(self, text=None, path=None, with_dist=False):
        # init normalizer
        language_normalizer = _araNorm()

        if text is not None:
            sentences = [text]
        elif path is not None:
            if not os.path.exists(path):
                raise Exception("File not found %s" % path)

            # read the batch file in tsv format
            df = pd.read_csv(path, delimiter='\t', header=None, names=['sentence'])

            # Create sentence lists
            sentences = df.sentence.values
        else:
            raise Exception('No text/path specified')

        # adding special tokens at the beginning and end of each sentence for BERT to work properly
        sentences = ["[CLS] " + language_normalizer.run(sentence) for sentence in sentences]

        # load test data
        test_inputs, test_masks = self.__data_prepare(sentences=sentences)
        test_data = TensorDataset(test_inputs, test_masks)
        test_dataloader = DataLoader(test_data, batch_size=100)

        # set model to evaluation mode
        self.__model.eval()

        results = []
        for batch in test_dataloader:
            batch = tuple(t.to(self.__device) for t in batch)

            # Unpack the inputs from dataloader
            b_input_ids, b_input_mask = batch
            del batch

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.__model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits[0].cpu()

            pred_probs = nn.functional.softmax(logits, dim=1)
            max_values, max_indices = torch.max(pred_probs, 1)
            max_values = max_values.numpy()
            max_indices = max_indices.numpy()
            pred_probs = pred_probs.numpy()

            for i in range(len(max_indices)):
                if with_dist:
                    results.extend([(self.__ind2lab[max_indices[i]], max_values[i],
                                     tuple(zip(self.__ind2lab.values(), pred_probs[i])))])
                else:
                    results.extend([(self.__ind2lab[max_indices[i]], max_values[i])])

        return results

    def __data_prepare(self, sentences, MAX_LEN=50):
        # Import the BERT tokenizer, used to convert the text into tokens that correspond to BERT's vocabulary.
        tokenized_texts = [self.__tokenizer.tokenize(sent) for sent in sentences]
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [self.__tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        # Pad the input tokens
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype=np.int_, truncating="post", padding="post")

        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        # Convert all of the data into torch tensors, the required datatype for the model
        inputs = torch.LongTensor(input_ids)
        masks = torch.tensor(attention_masks)

        return inputs, masks


'''--------------------------------------------------------------------------------
Script: Normalization class
Authors: Abdel-Rahim Elmadany and Muhammad Abdul-Mageed
Creation date: Novamber, 2018
Last update: Jan, 2019
input: text
output: normalized text
------------------------------------------------------------------------------------
Normalization functions:
- Check if text contains at least one Arabic Letter, run normalizer
- Normalize Alef and Yeh forms
- Remove Tashkeeel (diac) from Atabic text
- Reduce character repitation of > 2 characters at time
- repalce links with space
- Remove twitter username with the word USER
- replace number with NUM
- Remove non letters or digits characters such as emoticons
------------------------------------------------------------------------------------'''
import re

class _araNorm():
    '''
        araNorm is a normalizer class for n Arabic Text
    '''

    def __init__(self):
        '''
        List of normalized characters
        '''
        self.normalize_chars = {u"\u0622": u"\u0627", u"\u0623": u"\u0627", u"\u0625": u"\u0627",
                                # All Araf forms to Alaf without hamza
                                u"\u0649": u"\u064A",  # ALEF MAKSURA to YAH
                                u"\u0629": u"\u0647"  # TEH MARBUTA to  HAH
                                }
        '''
        list of diac unicode and underscore
        '''
        self.Tashkeel_underscore_chars = {u"\u0640": "_", u"\u064E": 'a', u"\u064F": 'u', u"\u0650": 'i',
                                          u"\u0651": '~', u"\u0652": 'o', u"\u064B": 'F', u"\u064C": 'N',
                                          u"\u064D": 'K'}

    def __normalizeChar(self, inputText):
        '''
        step #2: Normalize Alef and Yeh forms
        '''
        norm = ""
        for char in inputText:
            if char in self.normalize_chars:
                norm = norm + self.normalize_chars[char]
            else:
                norm = norm + char
        return norm

    def __remover_tashkeel(self, inputText):
        '''
        step #3: Remove Tashkeeel (diac) from Atabic text
        '''
        text_without_Tashkeel = ""
        for char in inputText:
            if char not in self.Tashkeel_underscore_chars:
                text_without_Tashkeel += char
        return text_without_Tashkeel

    def __reduce_characters(self, inputText):
        '''
        step #4: Reduce character repitation of > 2 characters at time
                 For example: the word 'cooooool' will convert to 'cool'
        '''
        # pattern to look for three or more repetitions of any character, including
        # newlines.
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        reduced_text = pattern.sub(r"\1\1", inputText)
        return reduced_text

    def __replace_links(self, inputText):
        '''
        step #5: repalce links to LINK
                 For example: http://too.gl/sadsad322 will replaced to LINK
        '''
        text = re.sub('(\w+:\/\/[ ]*\S+)', '+++++++++', inputText)  # LINK
        text = re.sub('\++', 'URL', text)
        return re.sub('(URL\s*)+', ' URL ', text)

    def __replace_username(self, inputText):
        '''
        step #5: Remove twitter username with the word USER
                 For example: @elmadany will replaced by space
        '''
        text = re.sub('(@[a-zA-Z0-9_]+)', 'USER', inputText)
        return re.sub('(USER\s*)+', ' USER ', text)

    def __replace_Number(self, inputText):
        '''
        step #7: replace number with NUM
                 For example: \d+ will replaced with NUM
        '''
        text = re.sub('[\d\.]+', 'NUM', inputText)
        return re.sub('(NUM\s*)+', ' NUM ', text)

    def __remove_nonLetters_Digits(self, inputText):
        '''
        step #8: Remove non letters or digits characters
                 For example: emoticons...etc
                 this step is very important for w2v  and similar models; and dictionary
        '''
        p1 = re.compile('[\W_\d\s]', re.IGNORECASE | re.UNICODE)  # re.compile('\p{Arabic}')
        sent = re.sub(p1, ' ', inputText)
        p1 = re.compile('\s+')
        sent = re.sub(p1, ' ', sent)
        return sent

    def run(self, text):
        normtext = ""
        text = self.__normalizeChar(text)
        text = self.__remover_tashkeel(text)
        text = self.__reduce_characters(text)
        text = self.__replace_links(text)
        text = self.__replace_username(text)
        text = self.__replace_Number(text)
        text = self.__remove_nonLetters_Digits(text)
        text = re.sub('\s+', ' ', text.strip())
        text = re.sub('\s+$', '', text.strip())
        normtext = re.sub('^\s+', '', text.strip())
        return normtext


def main():
    parser = optparse.OptionParser()
    parser.add_option('-p', '--path', action="store", default=None,
                      help='specify the model path on the command line')
    parser.add_option('-b', '--batch', action="store", default=None,
                      help='specify a file path on the command line')
    parser.add_option('-d', '--dist', action='store_true', default=False, help='show full distribution over languages')

    options, args = parser.parse_args()

    identifier = AraNet(options.path)
    if options.batch is not None:
        # "==== Batch Mode ===="
        predictions = identifier.predict(text=None, path=options.batch, with_dist=options.dist)
        print(predictions)
    else:
        import sys
        if sys.stdin.isatty():
            # "==== Interactive Mode ===="
            while True:
                try:
                    print(">>>", end=' ')
                    text = input()
                except Exception as e:
                    print(e)
                    break
                predictions = identifier.predict(text=text, path=None, with_dist=options.dist)
                print(predictions)
        else:
            # "==== Redirected Mode ===="
            lines = sys.stdin.read()
            predictions = []
            for text in lines:
                predictions.extend(identifier.predict(text=text, path=None, with_dist=options.dist))
            print(predictions)


if __name__ == "__main__":
    main()
