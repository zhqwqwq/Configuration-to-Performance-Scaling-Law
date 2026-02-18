def recover_text(input_ids, is_number_mask, number_values_filled, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    new_tokens = []
    for token, is_num, val in zip(tokens, is_number_mask, number_values_filled):
        if is_num:
            if hasattr(val, "item"):
                val = val.item()
            new_tokens.append(str(val))
        else:
            new_tokens.append(token)
    text = tokenizer.convert_tokens_to_string(new_tokens)
    return text