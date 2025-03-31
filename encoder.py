import tiktoken


class Encoder:
    def __init__(self):
        gpt2_base = tiktoken.get_encoding("gpt2")

        # In production, load the arguments directly instead of accessing private attributes
        # See openai_public.py for examples of arguments for specific encodings
        enc = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name="gpt_instruct",
            pat_str=gpt2_base._pat_str,
            mergeable_ranks=gpt2_base._mergeable_ranks,
            special_tokens={
                **gpt2_base._special_tokens,
                "<|pad_token|>": 50257,
                "<|eop_token|>": 50258,
            },
        )
        enc.pad_token = 50257
        enc.eop_token = 50258

        self.encoder = enc


if __name__ == "__main__":
    encoder = Encoder()
    instruction = "Write a summary of the given text."
    print("Instruction:", instruction)

    encoded_instr = encoder.encoder.encode_ordinary(instruction)
    print("Encoded instruction:", encoded_instr)

    decoded_instr = encoder.encoder.decode(encoded_instr)
    print("Decoded instruction:", decoded_instr)
