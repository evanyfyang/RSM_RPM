import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from transformers import AutoTokenizer
import json
from . import load_json
from utils.f1_measure import F1_Measure

sep1 = ' | '
sep2 = ' ; '
lite_sep1 = '|'
lite_sep2 = ';'

rest_category_former = {'restaurant': 0, 'service': 1, 'food': 2, 'drinks': 3, 'ambience': 4, 'location': 5}
rest_category_laster = {'general': 0, 'quality': 1, 'style_options': 2, 'prices': 3, 'miscellaneous': 4}

laptop_category_former = {'support': 0, 'laptop': 1, 'shipping': 2, 'multimedia_devices': 3, 
                          'ports': 4, 'hardware': 5, 'display': 6, 'keyboard': 7, 'power_supply': 8, 
                          'battery': 9, 'software': 10, 'os': 11, 'fans&cooling': 12, 'cpu': 13, 
                          'out_of_scope': 14, 'memory': 15, 'company': 16, 'warranty': 17, 'graphics': 18, 
                          'optical_drives': 19, 'hard_disc': 20, 'mouse': 21, 'motherboard': 22}
laptop_category_laster = {'price': 0, 'general': 1, 'quality': 2, 'operation_performance': 3, 
                          'design_features': 4, 'portability': 5, 'connectivity': 6, 'usability': 7, 'miscellaneous': 8}

_sentiment_to_word = {
    'POS': 'positive',
    'NEU': 'neutral' ,
    'NEG': 'negative',
    'positive': 'POS',
    'neutral' : 'NEU',
    'negative': 'NEG',
}
def sentiment_to_word(key):
    if key not in _sentiment_to_word:
        return 'UNK'
    return _sentiment_to_word[key]



class DataCollator:
    def __init__(self, tokenizer, max_seq_length, mode):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

    def tok(self, text, max_seq_length):
        kwargs = {
            'text': text,
            'return_tensors': 'pt'
        }

        if max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True

        else:
            kwargs['max_length'] = self.max_seq_length
            kwargs['padding'] = 'max_length'
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        return batch_encodings    

    def __call__(self, examples):
        IDs  = [example['ID'] for example in examples]
        text = [example['sentence'] for example in examples]

        batch_encodings = self.tok(text, self.max_seq_length)
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']
        max_len = attention_mask.shape[1]

        labels = None
        labels = self.make_labels(examples)
        table_labels = self.make_table_supervising_labels(examples, max_len, attention_mask)
        table_labels_S, table_labels_E = self.make_table_se(examples, max_len, attention_mask)
        table_labels_new = self.make_table_new(examples, max_len, attention_mask)
        pairs_true = self.make_pairs_true(examples)
        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'ID'            : IDs,
            'labels'        : labels,
            'table_labels'  : table_labels,
            'table_labels_S': table_labels_S,
            'table_labels_E': table_labels_E,
            'pairs_true'    : pairs_true,
            'table_labels_new': table_labels_new
        }

    def make_pairs_true(self, examples):
        pairs_true = []
        pol_map = {'negative':1, 'neutral':2, 'positive':3}
        for i in range(len(examples)):
            pairs = []
            quads = examples[i]['quads']
            for quad in quads:
                aspect_s, aspect_e = quad['aspect'][:2]
                opinion_s, opinion_e = quad['opinion'][:2]
                pairs.append([aspect_s, aspect_e, opinion_s, opinion_e, pol_map[quad['sentiment']]])
            pairs_true.append(pairs)
        return pairs_true
    
    def make_table_supervising_labels(self, examples, max_len, attention_mask):
        batch_size = len(examples)
        table_labels = -(1-(attention_mask[:,None,:] & attention_mask[:,:,None]))*100
        pol_map = {'negative':1, 'neutral':2, 'positive':3}
        for i in range(len(examples)):
            quads = examples[i]['quads']
            for quad in quads:
                aspect_s, aspect_e = quad['aspect'][:2]
                opinion_s, opinion_e = quad['opinion'][:2]
                polarity = pol_map[quad['sentiment']]

                table_labels[i, aspect_s, aspect_e-1] = 1
                table_labels[i, aspect_e-1, aspect_s] = 1
                table_labels[i, aspect_s, opinion_e-1] = 2
                table_labels[i, opinion_e-1, aspect_s] = 2
                table_labels[i, opinion_s, aspect_e-1] = 3
                table_labels[i, aspect_e-1, opinion_s] = 3
                table_labels[i, opinion_s, opinion_e-1] = 4
                table_labels[i, opinion_e-1, opinion_s] = 4
                
        return table_labels
    
    def make_table_new(self,examples, max_len, attention_mask):
        batch_size = len(examples)
        table_labels_new = -(1-(attention_mask[:,None,:] & attention_mask[:,:,None]))*100
        table_labels_new = table_labels_new.unsqueeze(-1).repeat(1,1,1,72)
        end_pos = attention_mask.sum(dim=1)
        pol_map = {'negative':1, 'neutral':2, 'positive':3}
        for i in range(len(examples)):
            quads = examples[i]['quads']
            for quad in quads:
                aspect_s, aspect_e = quad['aspect'][:2]
                opinion_s, opinion_e = quad['opinion'][:2]
                aspect_term = quad['aspect'][-1]
                opinion_term = quad['opinion'][-1]
                c_f, c_l = quad['category'].split()
                if c_f in rest_category_former:
                    category_former = rest_category_former[c_f]
                    former_len = len(rest_category_former)
                    category_laster = rest_category_laster[c_l]
                else:
                    category_former = laptop_category_former[c_f]
                    former_len = len(laptop_category_former)
                    category_laster = laptop_category_laster[c_l]

                polarity = pol_map[quad['sentiment']]
                # A,O,S_neg, S-neu, S-pos, E-neg, E-neu, S-pos
                end_item = end_pos[i].item()-1
                if aspect_term == 'NULL':
                    aspect_s = end_item
                    aspect_e = end_item+1
                if opinion_term == 'NULL':
                    opinion_s = end_item
                    opinion_e = end_item+1
                table_labels_new[i, aspect_s, aspect_e-1, 0] = 1
                table_labels_new[i, aspect_e-1, aspect_s, 0] = 1
                table_labels_new[i, opinion_s, opinion_e-1, 1] = 1
                table_labels_new[i, opinion_e-1, opinion_s, 1] = 1

                table_labels_new[i, aspect_s, opinion_s, polarity*2] = 1
                table_labels_new[i, aspect_e-1, opinion_e-1, polarity*2+1] = 1
                table_labels_new[i, opinion_s, aspect_s, polarity*2] = 1
                table_labels_new[i, opinion_e-1, aspect_e-1, polarity*2+1] = 1

                table_labels_new[i, aspect_s, opinion_s, category_former*2+8] = 1
                table_labels_new[i, aspect_e-1, opinion_e-1, category_former*2+9] = 1
                table_labels_new[i, opinion_s, aspect_s, category_former*2+8] = 1
                table_labels_new[i, opinion_e-1, aspect_e-1, category_former*2+9] = 1

                table_labels_new[i, aspect_s, opinion_s, category_laster*2+former_len*2+8] = 1
                table_labels_new[i, aspect_e-1, opinion_e-1, category_laster*2+former_len*2+9] = 1
                table_labels_new[i, opinion_s, aspect_s, category_laster*2+former_len*2+8] = 1
                table_labels_new[i, opinion_e-1, aspect_e-1, category_laster*2+former_len*2+9] = 1
        return table_labels_new
    
    def make_table_se(self,examples, max_len, attention_mask):
        batch_size = len(examples)
        table_labels_S = -(1-(attention_mask[:,None,:] & attention_mask[:,:,None]))*100
        table_labels_E = -(1-(attention_mask[:,None,:] & attention_mask[:,:,None]))*100
        pol_map = {'negative':1, 'neutral':2, 'positive':3}
        for i in range(len(examples)):
            quads = examples[i]['quads']
            for quad in quads:
                aspect_s, aspect_e = quad['aspect'][:2]
                opinion_s, opinion_e = quad['opinion'][:2]
                polarity = pol_map[quad['sentiment']]
                table_labels_S[i, aspect_s, opinion_s] = polarity
                table_labels_S[i, opinion_s, aspect_s] = polarity
                table_labels_E[i, aspect_e-1, opinion_e-1] = polarity
                table_labels_E[i, opinion_e-1, aspect_e-1] = polarity
        return table_labels_S, table_labels_E
    
    def make_labels(self, examples):
        quads_seqs = []
        for i in range(len(examples)):
            quads_seq = self.make_quads_seq(examples[i])
            quads_seqs.append(quads_seq)

        batch_encodings = self.tok(quads_seqs, -1)
        labels = batch_encodings['input_ids']
        labels = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100)
             for l in label]
            for label in labels
        ])

        return labels

    def make_quads_seq(self, example):
        return make_quads_seq(example)

def make_quads_seq(example):
    quads_seq = []
    for quad in sorted(
        example['quads'],
        key=lambda t: (t['aspect'][0], t['opinion'][0])
    ):  
        quad_seq = (
            quad['aspect'][-1] + 
            sep1 + 
            quad['category'] +
            sep1 + 
            quad['opinion'][-1] + 
            sep1 +
            quad['sentiment']
        )
        quads_seq.append(quad_seq)

    return sep2.join(quads_seq)

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str='',
        max_seq_length: int = -1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        data_dir: str = '',
        dataset: str = '',
        seed: int = 42,
    ):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length     = max_seq_length
        self.train_batch_size   = train_batch_size
        self.eval_batch_size    = eval_batch_size
        self.seed               = seed

        if dataset != '':
            self.data_dir       = os.path.join(data_dir, dataset)
        else:
            self.data_dir       = data_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        self.raw_datasets = {
            'train': load_json(train_file_name),
            'dev'  : load_json(dev_file_name),
            'test' : load_json(test_file_name)
        }
        print('-----------data statistic-------------')
        print('Train', len(self.raw_datasets['train']))
        print('Dev',   len(self.raw_datasets['dev']))
        print('Test',  len(self.raw_datasets['test']))

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=8,
            num_workers=1,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer, 
                max_seq_length=self.max_seq_length,
                mode=mode
            )
        )

        print('dataloader-'+mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)

    def predict_dataloader(self):
        return self.get_dataloader("predict", self.eval_batch_size, shuffle=False)



def parse_quad(tokenizer, quad_seq, example):
    if quad_seq.count(lite_sep1) != 3:
        return False

    aspect, category, opinion, sentiment = quad_seq.split(lite_sep1)
    aspect  = aspect.strip()
    opinion = opinion.strip()
    category = category.strip()
    sentiment = sentiment.strip()

    sentence = example['sentence']
    re_sentence = tokenizer.decode(tokenizer(sentence).input_ids, skip_special_tokens=True).strip()
    
    if (aspect not in example['sentence']) and (aspect not in re_sentence) and aspect != 'NULL':
        return False

    if (opinion not in example['sentence']) and (opinion not in re_sentence) and opinion != 'NULL':
        return False

    if sentiment not in ('positive', 'neutral', 'negative'):
        return False

    category_divs = category.split()
    category_former = category_divs[0]
    category_laster = category_divs[-1]
    if category_former not in rest_category_former.keys() and category_former not in laptop_category_former.keys():
        return False

    if category_laster not in rest_category_laster.keys() and category_laster not in laptop_category_laster.keys():
        return False
    return aspect, category, opinion, sentiment



class Result:
    def __init__(self, data):
        self.data = data 

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor >  other.monitor

    @classmethod
    def parse_from(cls, tokenizer, outputs, examples):
        data = {}
        examples = {example['ID']: example for example in examples}

        for output in outputs:
            IDs = output['ID']
            predictions = output['predictions']

            for ID in IDs:
                if ID not in data:
                    example = examples[ID]
                    sentence = example['sentence']
                    data[ID] = {
                        'ID': ID,
                        'sentence': sentence,
                        'quads': example['quads'],
                        'quad_preds' : [],
                    }

            for ID, prediction in zip(IDs, predictions):
                example = data[ID]
                quad_seqs = prediction.split(lite_sep2)
                for quad_seq in quad_seqs:
                    quad = parse_quad(tokenizer, quad_seq, example)
                    if not quad:
                        continue

                    example['quad_preds'].append(quad)

        return cls(data)

    def cal_metric_and_save(self, hparams):
        f1 = F1_Measure()

        for ID in self.data:
            example = self.data[ID]
            g = [(t['aspect'][-1], t['category'],t['opinion'][-1], t['sentiment']) 
                  for t in example['quads']]
            p = example['quad_preds']
            f1.true_inc(ID, g)
            f1.pred_inc(ID, p)

        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['r'],
            'precision': f1['p'],
        }

        self.monitor = self.detailed_metrics['f1']
        
        try:
            fr = open('./result.json','r')
            js = json.load(fr)
        except:
            js = []
        js.append({'use_prompt':hparams.use_prompt, 'use_super':hparams.use_super, 'seed':hparams.seed, 'dataset':hparams.dataset, 'metric':self.detailed_metrics})
        fr.close()
        fw = open('./result.json','w')
        json.dump(js, fw, indent=2)
        fw.close()

    def cal_metric(self):
        f1 = F1_Measure()

        for ID in self.data:
            example = self.data[ID]
            g = [(t['aspect'][-1], t['category'], t['opinion'][-1], t['sentiment']) 
                  for t in example['quads']]
            p = example['quad_preds']
            f1.true_inc(ID, g)
            f1.pred_inc(ID, p)

        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['r'],
            'precision': f1['p'],
        }

        self.monitor = self.detailed_metrics['f1']

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()
