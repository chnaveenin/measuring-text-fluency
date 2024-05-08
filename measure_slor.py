import math
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def compute_slor(sentence):
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Pad token_ids to a maximum length
    max_length = 512  # Max length for BERT
    padded_token_ids = token_ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(token_ids))
    
    # Convert token_ids to tensor
    input_ids = torch.tensor([padded_token_ids])
    
    # Compute the log-probability of the sentence given by the language model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        sentence_log_prob = sum(logits[0, i, token_ids[i]].item() for i in range(len(tokens)))  # Log probability of the sentence

    # Compute the unigram probability
    unigram_log_prob = sum(math.log(tokenizer.get_vocab()[token] / tokenizer.vocab_size) for token in tokens)
    
    # Compute the length of the sentence in terms of tokens
    sentence_length = len(tokens)
    
    # Compute SLOR score
    slor_score = (sentence_log_prob - unigram_log_prob) / sentence_length
    
    return slor_score

# Example usage
# sentence = "Six after dead construction in collapse wall."
# slor = compute_slor(sentence)
# print("SLOR Score:", slor)

