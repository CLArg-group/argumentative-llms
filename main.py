import argparse
import json
import os

from collections import defaultdict
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss, roc_auc_score

import Uncertainpy.src.uncertainpy.gradual as grad
from argument_miner import ArgumentMiner
from uncertainty_estimator import UncertaintyEstimator
from llm_managers import HuggingFaceLlmManager, OpenAiLlmManager

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.model import BlackboxModel
from lm_polygraph.estimators import *
from lm_polygraph.estimators import SemanticEntropy, PTrue, ClaimConditionedProbability, Eccentricity, MeanConditionalPointwiseMutualInformation, SAR
import prompt
import pickle
import torch

if __name__ == "__main__":
    baseline_prompt_class = prompt.BaselinePrompts()
    am_prompt_class = prompt.ArgumentMiningPrompts()
    ue_prompt_class = prompt.UncertaintyEvaluatorPrompts()
    baseline_prompts = [func for func in dir(baseline_prompt_class) if "__" not in func]
    am_prompts = [func for func in dir(am_prompt_class) if "__" not in func]
    ue_prompts = [func for func in dir(ue_prompt_class) if "__" not in func]

    parser = argparse.ArgumentParser(description="ArgumentativeLLM")
    # model related args
    parser.add_argument(
        "--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="Datasets/TruthfulQA/Prompt"
    )
    parser.add_argument("--save-loc", type=str, default="results/")
    parser.add_argument(
        "--cache-dir", type=str, default="/vol/bitbucket/clarg/argumentative-llm/cache"
    )
    parser.add_argument(
        "--baselines", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--direct", action=argparse.BooleanOptionalAction, default=False
    )
    # model parameter args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--quantization", type=str, default="8bit", choices=["4bit", "8bit", "none"]
    )
    # generation related args
    parser.add_argument(
        "--baseline-prompt", type=str, choices=baseline_prompts, default="all"
    )
    parser.add_argument(
        "--am-prompt", type=str, choices=am_prompts + ["all"], default="all"
    )
    parser.add_argument(
        "--ue-prompt", type=str, choices=ue_prompts + ["all"], default="all"
    )
    parser.add_argument(
        "--verbal", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--breadth", type=int, default=1)
    parser.add_argument(
        "--semantics", type=str, choices=["dfquad", "qe", "eb"], default="dfquad"
    )
    # ISO Contribution: pass in the ue_method through the command line
    parser.add_argument(
        "--ue-method", type=str, choices=["semantic_entropy", "eccentricity"], default=None)
    args = parser.parse_args()

    print("Loading model...")
    if "openai" in args.model_name:
        llm_manager = OpenAiLlmManager(
            model_name=args.model_name,
        )
    else:
        llm_manager = HuggingFaceLlmManager(
            model_name=args.model_name,
            quantization=args.quantization,
            cache_dir=args.cache_dir
        )
    generation_args = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
    }

    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_name)
    
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    model_name = args.model_name
    cache_dir = args.cache_dir
 
    transformers.logging.set_verbosity_error()   
    depth = args.depth 
    dataset_name = args.dataset_name
    ue_method = args.ue_method
    
    # ISO Contribution: Loading in the model using the LM-Polygraph library
    if "openai" in model_name:
        model_name = model_name.split("openai/")[1]
        openai_key = os.environ["OPENAI_KEY"]
        model = BlackboxModel.from_openai(openai_key, model_name, device_map="cuda:1", quantization_config = quantization_config)
    else:
        model = WhiteboxModel.from_pretrained(model_name, device_map="cuda:1", quantization_config = quantization_config)

    # ISO Contribution: Selecting and initializing the UQ method
    if ue_method:
        if ue_method == "semantic_entropy":
            ue_method = SemanticEntropy()
        elif ue_method == "eccentricity":
            ue_method = Eccentricity()
        else:
            raise ValueError("The UQ method you passed in is not yet supported.")
    
    if not args.baselines:
        if args.semantics == "qe":
            agg_f = grad.semantics.modular.SumAggregation()
            inf_f = grad.semantics.modular.QuadraticMaximumInfluence(conservativeness=1)
        elif args.semantics == "dfquad":
            agg_f = grad.semantics.modular.ProductAggregation()
            inf_f = grad.semantics.modular.LinearInfluence(conservativeness=1)
        elif args.semantics == "eb":
            agg_f = grad.semantics.modular.SumAggregation()
            inf_f = grad.semantics.modular.EulerBasedInfluence()

        if args.am_prompt != "all":
            am_prompts = [args.am_prompt]
        if args.ue_prompt != "all":
            ue_prompts = [args.ue_prompt]
        
        for am_prompt in am_prompts:

            for ue_prompt in ue_prompts:
                print()
                print()
                print(
                    f"Prompt experiment with AM: {am_prompt}, UE: {ue_prompt}, and verbal: {args.verbal}"
                )
                generate_prompt_am = getattr(am_prompt_class, am_prompt)
                generate_prompt_ue = getattr(ue_prompt_class, ue_prompt)

                base_score_gen = UncertaintyEstimator(
                    llm_manager=llm_manager,
                    generate_prompt=generate_prompt_ue,
                    verbal=args.verbal,
                    generation_args=generation_args,
                )
                
                am = ArgumentMiner(
                    generate_prompt_am=generate_prompt_am,
                    generate_prompt_ue=generate_prompt_ue,
                    llm_manager=llm_manager,
                    depth=args.depth,
                    breadth=args.breadth,
                    generation_args=generation_args,
                )
                
                results = []
                # ISO Contribution: For min-max linear normalization, retrieve the min and max of the 
                # normalization pass, which are stored in pkl files
                if ue_method == "semantic_entropy":
                    with open("min_max_vals_se.pkl", "rb") as f:
                        data = pickle.load(f)
                        min_val = data["min_val"]
                        max_val = data["max_val"]
                elif ue_method == "eccentricity":
                    with open("min_max_vals_ecc.pkl", "rb") as f:
                        data = pickle.load(f)
                        min_val = data["min_val"]
                        max_val = data["max_val"]
                else:
                    min_val = None 
                    max_val = None
                    
                    
                for data in dataset:
                    # Call the LM-Polygraph version of the function if there is a ue_method passed into the command line
                    if ue_method:
                        t_base, t_estimated = am.generate_arguments_lm_polygraph(data["claim"], model, ue_method, base_score_gen, min_val, max_val)
                    else:
                        t_base, t_estimated = am.generate_arguments(data["claim"], base_score_gen)
                    grad.algorithms.computeStrengthValues(t_base, agg_f, inf_f)
                    grad.algorithms.computeStrengthValues(t_estimated, agg_f, inf_f)
                    results.append(
                        {
                            "base": {
                                "bag": t_base.to_dict(),
                                "prediction": t_base.arguments[f"db0"].strength,
                            },
                            "estimated": {
                                "bag": t_estimated.to_dict(),
                                "prediction": t_estimated.arguments[f"db0"].strength,
                            },
                            "valid": data["valid"],
                        }
                    )
             
                    torch.cuda.empty_cache()
                    
                    

                print("Evaluating...")
                bag_types = ["base", "estimated"]
                # Entry format: (metric implementation, metric takes probabilities flag)
                metrics = {
                    "accuracy": (accuracy_score, False),
                    "f1": (f1_score, False),
                    "brier": (brier_score_loss, True),
                    "auc": (roc_auc_score, True),
                }

                probabilities = defaultdict(list)
                predictions = defaultdict(list)
                labels = []
                for result in results:
                    for t in bag_types:
                        probability = result[t]["prediction"]
                        probabilities[t].append(probability)
                        predictions[t].append(probability > 0.5)
                    labels.append(bool(result["valid"]))

                eval_results = defaultdict(dict)
                for t in bag_types:
                    eval_results[t] = {}
                    for metric, (metric_fun, takes_probabilities) in metrics.items():
                        if takes_probabilities:
                            result = metric_fun(labels, probabilities[t])
                        else:
                            result = metric_fun(labels, predictions[t])
                        eval_results[t][metric] = result

                experiment_summary = {
                    "arguments": vars(args),
                    "eval_results": eval_results,
                    "data": results,
                }

                if not os.path.exists(args.save_loc):
                    os.makedirs(args.save_loc)
                with open(
                    os.path.join(
                        args.save_loc,
                        f"AM-{am_prompt}_UE-{ue_prompt}_V-{args.verbal}_D-{args.depth}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(experiment_summary, f)
    else:
        # Boost the max token limit for CoT for fairness
        generation_args["max_new_tokens"] *= 6

        if args.baseline_prompt != "all":
            baseline_prompts = [args.baseline_prompt]

        for baseline_prompt in baseline_prompts:
            print()
            print()
            print("Baseline experiment with prompt: ", baseline_prompt)
            generate_prompt = getattr(baseline_prompt_class, baseline_prompt)

            predictions = []
            labels = []
            for data in dataset:
                generated_prompt, constraints, format_fun = generate_prompt(
                    data["claim"], direct=args.direct
                )
                if not args.direct:
                    if "meta-llama" in args.model_name:
                        # Ignore constraints for meta-llama, as it breaks generation
                        constraints = {}

                    reasoning = llm_manager.chat_completion(
                        generated_prompt,
                        print_result=True,
                        trim_response=False,
                        **constraints,
                        **generation_args,
                    )
                    prediction = format_fun(
                        llm_manager.chat_completion(
                            reasoning,
                            constraint_prefix="Therefore, the final answer (true or false) is:",
                            print_result=True,
                            trim_response=True,
                            apply_template=False,
                            **generation_args,
                        )
                    )
                else:
                    prediction = format_fun(
                        llm_manager.chat_completion(
                            generated_prompt,
                            print_result=True,
                            trim_response=True,
                            **constraints,
                            **generation_args,
                        )
                    )
                predictions.append(prediction)
                labels.append(data["valid"])
                print(f"Prediction: {prediction}, Label: {data['valid']}", flush=True)

            print("Evaluating...")
            metrics = {
                "accuracy": accuracy_score,
                "f1": f1_score,
            }

            eval_results = {}
            for metric, metric_fun in metrics.items():
                result = metric_fun(labels, predictions)
                eval_results[metric] = result

            experiment_summary = {
                "arguments": vars(args),
                "predictions": predictions,
                "labels": labels,
                "eval_results": eval_results,
            }

            if not os.path.exists(args.save_loc):
                os.makedirs(args.save_loc)
            with open(
                os.path.join(
                    args.save_loc,
                    f"B-{baseline_prompt}_D-{args.direct}.json",
                ),
                "w",
            ) as f:
                json.dump(experiment_summary, f)
