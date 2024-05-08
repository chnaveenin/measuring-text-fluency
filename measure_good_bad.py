import math
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def calculate_fluency(text, val_for_good=0.4, val_for_bad=0.01):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Pad token_ids to a maximum length
    max_length = 512  # Max length for BERT
    padded_token_ids = token_ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(token_ids))
    
    # Convert token_ids to tensor
    input_ids = torch.tensor([padded_token_ids])
    
    
    # Mask one token at a time and calculate the log probabilities
    log_prob_sum = 0
    total_tokens = len(tokens)
    fluency = 1/total_tokens
    
    with torch.no_grad():
        for i in range(total_tokens):
            masked_input_ids = input_ids.clone()
            masked_input_ids[0][i] = tokenizer.mask_token_id
            outputs = model(masked_input_ids)
            logits = outputs.logits
            mask_token_logits = logits[0, i, :]
            mask_token_probs = torch.softmax(mask_token_logits, dim=-1)
            token_prob = mask_token_probs[token_ids[i]].item()  # Probability of the correct token
            
            print(f"Token: {tokens[i]}, Probability: {token_prob}")
            
            # Adjust perplexity based on token probability
            if token_prob >= val_for_good:
                fluency /= token_prob
            elif token_prob <= val_for_bad:
                fluency *= token_prob
    
    return math.log(fluency)/total_tokens

# Example usage
text = "Six after dead construction in collapse wall."
fluency = calculate_fluency(text)
print("Fluency:", fluency)