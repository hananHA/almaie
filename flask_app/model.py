from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining
from torch.utils.data import Dataset
import torch
import random

MODEL = 'akhooli/gpt2-small-arabic-poetry'
WEIGHT = 'pytorch_model.bin'
MAXLEN = 768  # {768, 1024, 1280, 1600}
SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}


class myDataset(Dataset):
    def __init__(self, data, tokenizer, randomize=True):
        rhyme, verse, keywords = [], [], []
        for k, v in data.items():
            rhyme.append(v[0])
            verse.append(v[1])
            keywords.append(v[2])

        self.randomize = randomize
        self.tokenizer = tokenizer
        self.rhyme = rhyme
        self.verse = verse
        self.keywords = keywords


    @staticmethod
    def join_keywords(keywords, randomize=True):
        N = len(keywords)

        # random sampling and shuffle
        if randomize: 
            M = random.choice(range(N+1))
            keywords = keywords[:M]
            random.shuffle(keywords)

        return ','.join(keywords)


    def __len__(self):
        return len(self.verse)
    
    def __getitem__(self, i):
        keywords = self.keywords[i].copy()
        kw = self.join_keywords(keywords, self.randomize)
        
        input = SPECIAL_TOKENS['bos_token'] + self.rhyme[i] + \
                SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token'] + \
                self.verse[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = tokenizer(input,                                   
                                   truncation=True, 
                                   max_length=MAXLEN, 
                                   padding="max_length")   
        
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}

def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #gpt2-small-arabic-poetry
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):
    # GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        # Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path: # use map_location=torch.device('cpu') if no cuda
        model.load_state_dict(torch.load(load_model_path))

    model.cuda() # comment if no cuda
    return model