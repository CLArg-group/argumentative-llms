import re


def baseline_formatter(response):
    if "false" in response.lower() and "true" in response.lower():
        return response.lower().rindex("true") > response.lower().rindex("false")
    elif "true" in response.lower():
        return True
    elif "false" in response.lower():
        return False
    else:
        # Return false as a default
        return False


class BaselinePrompts:
    @staticmethod
    def analyst(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            instructions = (
                f"You are an analyst evaluating whether statements are true or false. "
                f'For the statement:\n\nStatement: "{statement}"\n\nplease determine '
                f"whether the statement is correct, accurate and truthful. "
                "Please think step by step before providing the final answer."
            )

            return instructions, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Result:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        instructions = (
            f"You are an analyst evaluating whether statements are true or false. "
            f'For the statement:\n\nStatement: "{statement}"\n\nplease determine '
            f"whether the statement is correct, accurate and truthful. "
            "Please give the result in the form 'Result: True' if the statement "
            "is true and 'Result: False' if the statement is false."
        )

        return instructions, constraints, baseline_formatter



class ArgumentMiningPrompts:
    @staticmethod
    def new_sup_att(statement, support=False, **_):
        """ New prompting for generating supporting and attacking arguments. Fixes the issue mentioned in the paper by conditioning the N/A statement on if it is supporting or attacking"""
        def formatter(argument, prompt):
            if "N/A" in argument or "n/a" in argument:
                return "N/A"
            return argument

        return (
            f"""Please provide a single short argument {"supporting" if support else "attacking"} the following claim. Construct the argument so it refers to the truthfulness of the claim. Only provide an argument if you think there is a valid and convincing {"support" if support else "attack"} for this claim (there is a non zero probability that this claim is {"true" if support else "false"}), otherwise return: N/A.
        Claim: {statement}
        Now take a deep breath and come up with an argument.
        Argument:""",
            {},
            formatter,
        )


class UncertaintyEvaluatorPrompts:
    @staticmethod
    def analyst(statement, claim=None, support=False, verbal=False, topic=False, **_):
        if not topic and claim is None:
            raise ValueError(
                "Claim is required for the analyst prompt without topic flag, but was None"
            )

        if verbal:

            def formatter(output):
                likelihood = output.replace("Confidence in argument:", "").strip()
                likelihood_dict = {
                    "fully confident": 0.95,
                    "highly confident": 0.8,
                    "quite confident": 0.65,
                    "moderately confident": 0.5,
                    "slightly confident": 0.35,
                    "not very confident": 0.2,
                    "not confident at all": 0.05,
                }
                return likelihood_dict[likelihood]

            relation = "supports" if support else "refutes"
            options = [
                "fully confident",
                "highly confident",
                "quite confident",
                "moderately confident",
                "slightly confident",
                "not very confident",
                "not confident at all",
            ]
            q = '"'
            constraints = {
                "constraint_prefix": "Confidence in argument:",
                "constraint_options": options,
                "constraint_end_after_options": True,
            }

            if topic:
                instructions = (
                    f"You are an analyst evaluating the validity of statements. "
                    f'For the statement:\n\nStatement: "{statement}"\n\nplease give your confidence '
                    f"that the statement is correct, accurate and truthful. "
                )
            else:
                instructions = (
                    f"You are an analyst evaluating the validity and relevance of arguments. "
                    f'For the argument:\n\nArgument: "{statement}"\n\nplease give your confidence '
                    f"that the argument presents a compelling case {'in favour of' if support else 'against'} "
                    f'the statement:\n\nStatement: "{claim}"\n\nYour assessment should be based '
                    f"on how well the argument {'supports' if support else 'refutes'} the considered "
                    "statement as well as the correctness, accuracy and truthfulness of the given argument. "
                )

            return (
                instructions
                + (
                    f"Your response should be chosen out of the options: "
                    f'{", ".join([q + o + q for o in options])}. '
                    "Please respond in the following form:"
                    f"\n\nConfidence in {'argument' if not topic else 'statement'}: "
                    f"Your confidence in the {'argument' if not topic else 'statement'} validity"
                ),
                constraints,
                formatter,
            )

        def formatter(output):
            try:
                likelihood = output.replace("Likelihood:", "").strip()
                likelihood = likelihood.replace("is", "").strip()
                likelihood = likelihood.replace("%", "").strip()
                likelihood = likelihood.replace(".", "").strip()
                likelihood = likelihood.split("\n")[0]
                return int(likelihood) / 100
            except ValueError:
                print(
                    "WARNING: Could not parse likelihood, returning 0.5. Offending output:",
                    output,
                )
                return 0.5

        constraints = {
            "constraint_prefix": "Likelihood:",
            "constraint_options": [f" {l}%" for l in range(0, 101)],
            "constraint_end_after_options": True,
        }

        if topic:
            instructions = (
                "You are an analyst evaluating the validity of statements. "
                f'For the statement:\n\nStatement: "{statement}"\n\nplease give your confidence '
                f"that the statement is correct, accurate and truthful. "
                f"Your response should be between 0% and 100% with 0% indicating that the "
                f"considered statement is definitely invalid, 100% indicating that the considered statement is "
            )
        else:
            instructions = (
                "You are an analyst evaluating the validity and relevance of arguments. "
                f'For the argument:\n\nArgument: "{statement}"\n\nplease give your confidence '
                f"that the argument presents a compelling case {'in favour of' if support else 'against'} "
                f'the statement:\n\nStatement: "{claim}"\n\nYour assessment should be based '
                f"on how well the argument {'supports' if support else 'refutes'} the considered "
                "statement as well as the correctness, accuracy and truthfulness of the given argument. "
                f"Your response should be between 0% and 100% with 0% indicating that the "
                f"considered argument is definitely invalid, 100% indicating that the considered argument is "
            )

        return (
            instructions
            + (
                "definitely valid and values in between indicating various levels of "
                "uncertainty. Your estimates should be well-calibrated, so feel free to "
                "err on the side of caution and output moderate probabilities if you are "
                "not completely sure in your assessment. "
                "Please respond in the following form:"
                "\n\nLikelihood: The predicted likelihood that the considered "
                f"{'argument' if not topic else 'statement'} is valid"
            ),
            constraints,
            formatter,
        )

    