import re


def extract_args(text):
    support_list = []
    attack_list = []

    # Extract support statements using regular expressions
    support_matches = re.findall(
        r"Support\d+: (.+?)(?=(Support\d+: |Attack\d+: |$))", text, re.S
    )
    for match in support_matches:
        # The regular expression capture group returns a tuple; we take the first element
        support_list.append(match[0].strip())

    # Extract attack statements using regular expressions
    attack_matches = re.findall(
        r"Attack\d+: (.+?)(?=(Support\d+: |Attack\d+: |$))", text, re.S
    )
    for match in attack_matches:
        # The regular expression capture group returns a tuple; we take the first element
        attack_list.append(match[0].strip())

    return support_list, attack_list


def construct_constraint_fun(
    tokenizer, prompt, force_prefix=None, force_options=None, end_after_options=False
):
    # Note that we disregard the BOS token when using input IDs
    if force_prefix is not None:
        force_prefix = tokenizer(force_prefix).input_ids[1:]
    if force_options is not None:
        force_options = [tokenizer(op).input_ids[1:] for op in force_options]
    all_tokens = list(tokenizer.get_vocab().values())

    def constraint_fun(batch_id, input_ids):
        prompt_len = len(tokenizer(prompt).input_ids)
        generated_tokens = input_ids[prompt_len:].tolist()
        num_generated = len(generated_tokens)
        prefix_len = 0 if force_prefix is None else len(force_prefix)

        if force_prefix is not None and num_generated < prefix_len:
            # Force prefix to be generated first if provided
            return [force_prefix[num_generated]]
        elif num_generated >= prefix_len and force_options is not None:
            # Determine what option tokens have been generated
            op_tokens = generated_tokens[prefix_len:]
            num_op = len(op_tokens)

            # Calculate valid option continuations
            possible_continuations = [
                c[num_op]
                for c in force_options
                if num_op < len(c) and c[:num_op] == op_tokens
            ]

            if not possible_continuations and end_after_options:
                # No further continuations — terminate generation as requested
                return [tokenizer.eos_token_id]
            elif not possible_continuations:
                # No further continuations, but can continue free generation
                return all_tokens
            else:
                # Allow generation to terminate if desirable
                if op_tokens in force_options:
                    possible_continuations.append(tokenizer.eos_token_id)
                # Force generation according to options
                return possible_continuations
        else:
            return all_tokens

    return constraint_fun
