import os
import re
from transformers import AutoTokenizer
from utils import load_text, load_json, save_json
import sys

def char_to_token(encoding, char_pos):
    offsets = encoding.offset_mapping[:-1]
    pos = 0
    for i in range(len(offsets)):
        offset = offsets[i]
        if offset[0] <= char_pos and offset[1] > char_pos:
            pos = i
    return pos

def convert(tokenizer, triplet, encoding, words, tokens, sentence, re_sentence):
    aspect, opinion, sentiment = triplet
    sentiment_convert = {'NEG':'negative', 'NEU':'neutral', 'POS':'positive'}
    
    def entity_convert(entity):
        start = entity[0]
        end   = entity[-1]+1

        char_start = sum([len(word) for word in words[:start]])
        char_end   = sum([len(word) for word in words[:end]])-1

        token_start = char_to_token(encoding, char_start)
        token_end   = char_to_token(encoding, char_end-1)+1

        token_ids = encoding.input_ids[token_start:token_end]

        term = tokenizer.decode(token_ids, skip_special_tokens = True).strip()

        char_term = tokenizer.decode(tokenizer(' '.join(words[start:end])).input_ids,skip_special_tokens=True)
        assert char_term == term, char_term + '|||' + term
        assert (term in sentence) or (term in re_sentence), term + '|||' + sentence + '|||' + re_sentence+ '\n' + str(term in sentence)
        assert token_start is not None

        return [
            token_start,
            token_end,
            tokens[token_start:token_end],
            term
        ]

    aspect = entity_convert(aspect)
    opinion = entity_convert(opinion)
    return {'aspect':aspect, 'opinion':opinion, 'sentiment':sentiment_convert[sentiment]}


def process(example, tokenizer):
    sentence, triplets = example.split('####')
    triplets = eval(triplets)

    words = sentence.split()
    words = [' '+word for word in words]

    if words[0][0] != ' ':
        words[0] = ' ' +words[0]

    encoding = tokenizer(sentence, return_offsets_mapping=True)
    tokens   = tokenizer.tokenize(sentence)

    re_tokenized_sentence = tokenizer.decode(encoding.input_ids, skip_special_tokens=True)

    assert len(sentence) == sum([len(word) for word in words])-1, '\n' + sentence + '\n' + str(words)

    triplets_token = []
    for triplet in triplets:
        triplet_token = convert(tokenizer, triplet, encoding, words, tokens, sentence, re_tokenized_sentence)
        triplets_token.append(triplet_token)
    
    triplets_token = sorted(triplets_token, key=lambda x:(x['aspect'][0],x['aspect'][1],x['opinion'][0],x['opinion'][1]))

    return {
        'sentence': sentence,
        'triplets': triplets_token,
        'tokens'  : str(tokens)
    }


if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_data_dir',    type=str)
    parser.add_argument('--output_data_dir', type=str)
    parser.add_argument('--dataset',         type=str)

    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained('t5-base',use_fast=True)

    for mode in ('train', 'dev', 'test'):

        t5_examples = []
        file_name = os.path.join(args.raw_data_dir, args.dataset, f'{mode}_triplets.txt')
        raw_examples = load_text(file_name)

        for i, example in enumerate(raw_examples):
            if example:
                t5_example = process(example, tokenizer)
                t5_example['ID'] = i
                t5_examples.append(t5_example)

        save_file_name = os.path.join(args.output_data_dir, args.dataset, f'{mode}.json')
        print('save', len(t5_examples), 'to', save_file_name)
        save_json(t5_examples, save_file_name)
