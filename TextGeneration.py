import numpy as np
import torch.nn.functional as F

import torch
import os

from tqdm import trange
from transformers import (GPT2Tokenizer, GPT2LMHeadModel)


class TextGenerator:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        # Here will use the GPT-2 model ,which had the SOTA preformance for GPT-2 model
        # Recommand to download model and tokenize file into local folder first
        self.model_address = 'models/gpt2-large/'

        # Downlaod pytorch_model.bin : https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin
        # Downlaod config.json : https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-config.json
        self.model = GPT2LMHeadModel.from_pretrained(self.model_address)

        # Download vocab file : https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json
        # Download merges_file : https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-merges.txt
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_address)

        # Set model to  device support
        # self.model.to(self.device);

        # Add multi GPU support
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Set random seed and tempreture
        self.seed = 4
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # if self.n_gpu > 0:
        #     torch.cuda.manual_seed_all(self.seed)

        self.temperature = 1.0

        self.max_len = 30
        self.top_k = 100
        self.top_p = 0.8

    def evaluateModel(self):
        # Set model to evalue mode
        self.model.eval();

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(self, model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
        '''Method to generate text with GPT-2 '''
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context

        with torch.no_grad():
            for _ in trange(length):
                inputs = {'input_ids': generated}

                outputs = model(
                    **inputs) 
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    def generateText(self, raw_text):
        # Text to embedding
        # Tokenize the text then map token to id
        context_tokens = self.tokenizer.encode(raw_text)
        # Generate
        out = self.sample_sequence(model=self.model, length=self.max_len, context=context_tokens, num_samples=1,
                                   temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, device='cpu')

        # Paraser result
        out = out[0, len(context_tokens):].tolist()
        text = self.tokenizer.decode(out, clean_up_tokenization_spaces=True)
        return text
        #print("=" * 50)
        #print("Text generated by computer")
        #print(text)

        #print("=" * 50)
        #print("The context we written before")
        #print(raw_text)

        #print("=" * 50)
        #print("Full text combined the context and generated text")
        #print(raw_text + ' ' + text)

    def executeprocess(self, raw_text):
        return self.generateText(raw_text)

'''
if __name__ == "__main__":
    # Demo test text, the generation will base on this sentence
    raw_text = "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976."
    txtGenrtrObj = TextGenerator()
    txtGenrtrObj.executeprocess(raw_text)
'''