import os
import re
from transformers import AutoTokenizer
from utils import load_text, load_json, save_json
import sys

categories = set()
def char_to_token(encoding, char_pos):
    offsets = encoding.offset_mapping[:-1]
    pos = 0
    for i in range(len(offsets)):
        offset = offsets[i]
        if offset[0] <= char_pos and offset[1] > char_pos:
            pos = i
    return pos

def convert(tokenizer, quad, encoding, words, tokens, sentence, re_sentence):
    aspect, category, sentiment, opinion = quad

    if category not in categories:
        categories.add(category)
    
    def entity_convert(entity, sentence,tokens):
        # fix some bugs, which could not be transformed in to index
        if entity == 'try it and believe it':
            entity = 'try it to believe it'
        elif entity == 'covered in a glaze of some kin':
            entity = 'covered in a glaze of some kind'
        elif entity == 'didn \\’ t really care':
            entity = 'didn ’ t really care'
        entity = entity.strip()
        char_end = 0
        if entity == '':
            entity = 'NULL'
        if entity == 'NULL':
            char_start = 0
            char_end = len(sentence)
            return [
                0,
                len(tokens),
                [],
                'NULL'
            ]
        else:
            while char_end == 0 or (char_end != len(sentence) and sentence[char_end].isalpha()) or (char_start != 0 and sentence[char_start-1].isalpha()):
                char_start = sentence.find(entity,char_end)
                if char_start == -1:
                    char_start = sentence.lower().find(entity.lower(),char_end)
                    if char_start == -1 and " n't" in sentence and "n't" in entity:
                        char_start = sentence.find(entity.replace("n't", " n't"), char_end)
                        char_end = char_start + len(entity) + 1
                    else:
                        char_end   = char_start + len(entity)
                else:
                    char_end   = char_start + len(entity)

        token_start = char_to_token(encoding, char_start)
        token_end   = char_to_token(encoding, char_end-1)+1

        token_ids = encoding.input_ids[token_start:token_end]

        term = tokenizer.decode(token_ids, skip_special_tokens = True).strip()

        char_term = tokenizer.decode(tokenizer(sentence[char_start:char_end]).input_ids,skip_special_tokens=True)
        if entity not in ['`` free ``', 'ssd drive ` `', "` ` salt encrusted shrimp ' ' appetizer"]:
            assert char_term.lower() == term.lower(), char_term + '|||' + term + '|||' + str(token_start) + '|||' +entity+'|||' +str(token_end)
            assert (entity.lower() in sentence.lower()) or (entity.lower() in re_sentence.lower()), term + '|||' + sentence + '|||' + re_sentence+ '\n' + str(term in sentence)
            assert token_start is not None

        return [
            token_start,
            token_end,
            tokens[token_start:token_end],
            entity
        ]

    aspect = entity_convert(aspect, sentence, tokens)
    opinion = entity_convert(opinion, sentence, tokens)
    return {'aspect':aspect, 'opinion':opinion, 'sentiment':sentiment, 'category':category}


def process(example, tokenizer):
    sentence, quads = example.split('####')
    quads = eval(quads)

    words = sentence.split()
    words = [' '+word for word in words]

    if words[0][0] != ' ':
        words[0] = ' ' +words[0]

    encoding = tokenizer(sentence, return_offsets_mapping=True)
    tokens   = tokenizer.tokenize(sentence)

    re_tokenized_sentence = tokenizer.decode(encoding.input_ids, skip_special_tokens=True)
    assert len(sentence) == sum([len(word) for word in words])-1, '\n' + sentence + '\n' + str(words)

    quads_token = []
    for quad in quads:
        quad_token = convert(tokenizer, quad, encoding, words, tokens, sentence, re_tokenized_sentence)
        quads_token.append(quad_token)
    
    quads_token = sorted(quads_token, key=lambda x:(x['aspect'][0],x['aspect'][1],x['opinion'][0],x['opinion'][1]))

    return {
        'sentence': sentence,
        'quads': quads_token,
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
        file_name = os.path.join(args.raw_data_dir, args.dataset, f'{mode}.txt')
        raw_examples = load_text(file_name)

        for i, example in enumerate(raw_examples):
            if example:
                t5_example = process(example, tokenizer)
                t5_example['ID'] = i
                t5_examples.append(t5_example)

        save_file_name = os.path.join(args.output_data_dir, args.dataset, f'{mode}.json')
        print('save', len(t5_examples), 'to', save_file_name)
        save_json(t5_examples, save_file_name)
        print(categories)
