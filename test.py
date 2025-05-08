from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Check reverse lookup
print("Token ID 4249 →", tokenizer.convert_ids_to_tokens(4249))
print("Token ID 2914 →", tokenizer.convert_ids_to_tokens(2914))

# Also forward lookup for safety
print("Token 'dangerous' →", tokenizer.convert_tokens_to_ids("."))
print("Token 'safe' →", tokenizer.convert_tokens_to_ids("##ly"))
