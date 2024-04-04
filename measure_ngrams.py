import math
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def calculate_fluency(text):
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
            log_prob_sum += math.log(token_prob)
    
    fluency = log_prob_sum / total_tokens
    return fluency

# Example usage
text = "What is the capital of a country?"
fluency = calculate_fluency(text)
print("fluency:", fluency)