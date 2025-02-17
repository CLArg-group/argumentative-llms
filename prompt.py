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
    def deniz(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            prompt = f"""
            Claim: {statement}

            Instructions:
                Consider the claim and determine whether it is true or false.
                Think step by step before providing the final answer.
                Utilize critical thinking and logical reasoning in your evaluation.
                Ensure clarity in your reasoning."""

            return prompt, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Answer:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        prompt = f"""
            Claim: {statement}

            Instructions:
                Consider the claim and determine whether it is true or false.
                Utilize critical thinking and logical reasoning in your evaluation.
                Ensure clarity in your reasoning."""

        return prompt, constraints, baseline_formatter

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

    @staticmethod
    def gabriel(statement, direct=False, **_):
        if not direct:
            constraints = {
                "constraint_prefix": "Let's think step by step.",
            }

            prompt = f"""Please provide an assessment based on the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, you should consider it false.
    Statement: {statement}
    Now take a deep breath, and determine whether the statement is true or false."""

            return prompt, constraints, baseline_formatter

        constraints = {
            "constraint_prefix": "Assessment:",
            "constraint_options": [" True", " False"],
            "constraint_end_after_options": True,
        }

        prompt = f"""Please provide an assessment based on the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, you should consider it false. Please think step by step before providing the final answer.
Statement: {statement}
Now take a deep breath, think step by step and determine whether the statement is true or false."""

        return prompt, constraints, baseline_formatter


class ArgumentMiningPrompts:
    @staticmethod
    def deniz(statement, support=False, **_):
        def formatter(argument, prompt):
            if "N/A" in argument or "n/a" in argument:
                return "N/A"
            return re.sub(
                " +",
                " ",
                "".join(re.findall(r"(.*[.?!])", argument.split("\n\n")[0])).strip(
                    '"\n'
                ),
            )

        return (
            f"""
        Claim: {statement}

        Instructions:
            Provide a concise argument {"supporting" if support else "opposing"} the claim in less than 2 sentences.
            Utilize critical thinking and logical reasoning in your argument.
            Ensure clarity in your reasoning.
            Avoid circular reasoning or fallacious arguments.
            If you cannot return a valid and convincing argument for this claim, reply N/A.

        {"Supporting" if support else "Opposing"} Argument for '{statement}':""",
            {},
            formatter,
        )

    @staticmethod
    def debater(statement, support=False, **_):
        def formatter(argument, prompt):
            if "N/A" in argument or "n/a" in argument:
                return "N/A"
            return argument

        return (
            (
                "You are a professional debater who will try to provide arguments on a topic even if "
                "they go against your personal believes. Please give a brief, one-sentence argument "
                f"{'in favour of' if support else 'against'} the statement:\n\nStatement: \"{statement}\"\n\n"
                "Note that the provided argument should provide a clear justification why the considered "
                f"statement is {'true and accurate' if support else 'untrue or inaccurate'}. "
                "The argument should also be as self-contained as possible. "
                "Please reply only with the argument sentence without any further commentary. "
                "If you are truly unable to provide such an argument, reply N/A."
            ),
            {},
            formatter,
        )

    @staticmethod
    def gabriel(statement, support=False, **_):
        def formatter(argument, prompt):
            if "N/A" in argument or "n/a" in argument:
                return "N/A"
            return argument

        return (
            f"""Please provide a single short argument {"supporting" if support else "attacking"} the following claim. Construct the argument so it refers to the truthfulness of the claim. Only provide an argument if you think there is a valid and convincing {"support" if support else "attack"} for this claim (there is a non-zero probability that the claim is true), otherwise return: N/A.
        Claim: {statement}
        Now take a deep breath and come up with an argument.
        Argument:""",
            {},
            formatter,
        )
    @staticmethod
    def new_sup_att(statement, support=False, **_):
        """ ISO Contribution: New prompting for generating supporting and attacking arguments. Fixes the
        issue mentioned in the report by conditioning the N/A statement on if it is supporting or attacking"""
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
    def deniz(statement, verbal=False, **_):
        if verbal:
            certainty_dict = {
                "certain": 0.95,
                "almost certain": 0.8,
                "quite certain": 0.65,
                "moderately certain": 0.5,
                "slightly certain": 0.35,
                "almost uncertain": 0.2,
                "uncertain": 0.05,
            }

            def formatter(certainty):
                return certainty_dict[certainty.replace("Certainty:", "").strip()]

            constraints = {
                "constraint_prefix": "Certainty:",
                "constraint_options": certainty_dict.keys(),
                "constraint_end_after_options": True,
            }

            prompt = f"""Task Description:
        Provide your certainty for a given argument. The certainty should reflect the level of confidence in the validity or strength of the argument.

        Instructions:
        1. Consider the argument provided and evaluate its clarity, logical coherence, and the amount of evidence presented.
        2. Provide your certainty using the following numerical mapping: {" ".join([k + ": " + str(certainty_dict[k]) for k in certainty_dict.keys()])} 
        3. Explain the reasoning behind your certainty, taking into account factors such as ambiguity in language, lack of evidence, and logical fallacies.

        Argument: {statement}

        Additional Guidance:
        Consider the context in which the argument is presented and any potential biases or assumptions that may affect its credibility.
        Evaluate the strength of the evidence provided and the coherence of the argument's structure.
        Take into account any potential counterarguments or alternative interpretations that could impact the certainty of the argument.

        Please respond in the following form:

        Certainty: Your certainty on the argument"""

        else:

            def formatter(certainty):
                return float(certainty.replace("Certainty Score:", "").strip())

            constraints = {
                "constraint_prefix": "Certainty Score:",
                "constraint_options": [f" {x / 100}" for x in range(0, 101)],
                "constraint_end_after_options": True,
            }

            prompt = f"""Task Description:
        Generate a certainty score for a given argument. The certainty score should reflect the level of confidence in the validity or strength of the argument.
        
        Instructions:
        1. Consider the argument provided and evaluate its clarity, logical coherence, and the amount of evidence presented.
        2. Assign a numerical certainty score on a scale from 0 to 1, where 0 indicates maximum uncertainty, 1 indicates maximum certainty, and values in between represent varying degrees of certainty.
        3. Explain the reasoning behind the certainty score, taking into account factors such as ambiguity in language, lack of evidence, and logical fallacies.
        
        Argument: {statement}
        
        Additional Guidance:
        Consider the context in which the argument is presented and any potential biases or assumptions that may affect its credibility.
        Evaluate the strength of the evidence provided and the coherence of the argument's structure.
        Take into account any potential counterarguments or alternative interpretations that could impact the certainty of the argument.
        
        Please respond in the following form:
        
        Certainty Score: Your certainty score on the argument"""

        return prompt, constraints, formatter

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

    @staticmethod
    def gabriel(statement, verbal=False, support=False, claim=None, topic=False):
        if verbal:

            def formatter(output):
                likelihood = output.replace("Quality score:", "").strip()
                likelihood_dict = {
                    "fully confident": 0.95,
                    "highly confident": 0.8,
                    "quite confident": 0.65,
                    "chances about even": 0.5,
                    "quite unconfident": 0.35,
                    "highly unconfident": 0.2,
                    "fully unconfident": 0.05,
                }
                return likelihood_dict[likelihood]

            options = [
                "fully confident",
                "highly confident",
                "quite confident",
                "chances about even",
                "quite unconfident",
                "highly unconfident",
                "fully unconfident",
            ]
            q = '"'
            constraints = {
                "constraint_prefix": "Quality score:",
                "constraint_options": options,
                "constraint_end_after_options": True,
            }

            prompt = f"""Please provide a quality score (as a verbal confidence taken from the list that follows) based on the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, it should receive a low score. Statement: {statement} Now take a deep breath and give a quality score, out of the options {", ".join([q + o + q for o in options])}."""
        else:

            def formatter(certainty):
                return int(certainty.replace("Quality score:", "").strip()) / 100

            constraints = {
                "constraint_prefix": "Quality score:",
                "constraint_options": [f" {x}" for x in range(0, 101)],
                "constraint_end_after_options": True,
            }

            if topic:
                prompt = f"""Please provide a quality score (as a single numerical value between 0 and 100) based the factuality and truthfulness of the following statement. If the statement is partially false or has to be interpreted in a very specific way to be considered true, it should receive a low score.
Statement: {statement}
Now take a deep breath and give a quality score."""
            else:
                prompt = f"""Please provide a quality score (as a single numerical value between 0 and 100) based on factuality, relevance and effectiveness, for how well the following argument {"supports" if support else "attacks"} the claim. If the argument suggests that the claim is partially false or must be interpreted in a specific way to be considered true, it should receive a low score.
Claim: {claim}
{"Supporting" if support else "Attacking"} argument: {statement}
Now take a deep breath and give a quality score."""

        return prompt, constraints, formatter
