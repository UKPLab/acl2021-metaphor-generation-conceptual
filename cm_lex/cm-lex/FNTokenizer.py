from transformers import BertTokenizer

class FNTokenizer(BertTokenizer):
    """
    Special tokenizer that keeps frame names as their own tokens
    """
    def tokenize(self, text, **kwargs):
        tokens = super().tokenize(text, **kwargs)
        cor_tokens = []
        in_frame = False
        fn_token = ""
        for c, token in enumerate(tokens):
            if token == "_" or any((x.isupper() for x in token)):
                fn_token += token.strip("#")
                in_frame = True
            else:
                if in_frame:
                    cor_tokens.append(fn_token)
                    fn_token = ""
                cor_tokens.append(token)  
                in_frame = False

        if in_frame:
            cor_tokens.append(fn_token)
        return cor_tokens
