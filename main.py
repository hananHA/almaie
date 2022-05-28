from flask import Flask, render_template, redirect, url_for, request
from model import myDataset, get_tokenier, get_model
import torch
app=Flask(__name__)

WEIGHT = 'pytorch_model.bin'
SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
MAXLEN = 768  # {768, 1024, 1280, 1600}
n_verse = 5

tokenizer=get_tokenier(special_tokens=SPECIAL_TOKENS)
model=get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path=WEIGHT)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    string = ''
    rhyme = request.form['rhyme']
    keywords = request.form['keywords']
    subject = request.form['subject']
    
    rhyme = keywords[0][-1]
    kw = myDataset.join_keywords(keywords, randomize=False)

    prompt = SPECIAL_TOKENS['bos_token'] + rhyme + \
        SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']
         
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    print('generate start')
    model.eval()
    # Top-p (nucleus) text generation (10 samples):
    sample_outputs = model.generate(generated, 
                                    do_sample=True,
                                    min_length=40, 
                                    max_length=MAXLEN,
                                    top_k=30,                                 
                                    top_p=0.7,
                                    temperature=0.9,
                                    repetition_penalty=2.0,
                                    num_return_sequences=n_verse
                                )

    print('generate end')
    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(rhyme) + len(','.join(keywords))
        string += str(i+1) + ' ' + text[a:] + '\n\n'
        
    return render_template('index.html', result=string)

if __name__=='__main__':
    app.run(debug=True)