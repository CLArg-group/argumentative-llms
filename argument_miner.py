from copy import deepcopy

import Uncertainpy.src.uncertainpy.gradual as grad
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import *
from lm_polygraph.estimators import SemanticEntropy
from lm_polygraph.utils import estimate_uncertainty
import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig

from utils import construct_constraint_fun

from openai import OpenAI
import time
from csv import writer
class ArgumentMiner:
    def __init__(
        self, generate_prompt_am, generate_prompt_ue, llm_manager, depth=1, breadth=1, generation_args={}, ue_method="semantic_entropy"
    ):
        self.depth = depth
        self.breadth = breadth
        self.llm_manager = llm_manager
        self.generate_prompt = generate_prompt_am
        self.generate_prompt_ue = generate_prompt_ue
        self.generation_args = generation_args
        self.ue_method = ue_method

    def generate_args_for_parent_lm_polygraph(self, parent, name, model, ue_method, min_val, max_val):
        """ ISO Contribution: New function for ISO Uncertainty Quantification Experiments
        Generates supporting and attacking arguments and computes the desired uncertainty measures """
        torch.cuda.empty_cache()
        s_prompt, s_constraints, s_format_args = self.generate_prompt(
            parent.get_arg(), support=True
        )
      
        print()
        sup_ue = estimate_uncertainty(model, ue_method, input_text=s_prompt)
        sup_ue_score = sup_ue.uncertainty
        sup_generation = sup_ue.generation_text
        sup = s_format_args(
            sup_generation,
            s_prompt,
        )
        print(s_prompt)
        print(sup_generation)
        print("pre normalized UE: ", sup_ue_score)
        if "N/A" in sup_generation:
            sup_ue_score = 0.0
        else:
            sup_ue_score = 1.0 - (sup_ue_score - min_val) / (max_val - min_val)

        print(sup_ue_score)
        
        a_prompt, a_constraints, a_format_args = self.generate_prompt(parent.get_arg())
        print()
        att_ue = estimate_uncertainty(model, ue_method, input_text=a_prompt)
        att_ue_score = att_ue.uncertainty
        att_generation = att_ue.generation_text
        att = a_format_args(
            att_generation,
            a_prompt,
        )
        print(a_prompt)
        print(att_generation)
        print("pre normalized UE: ", att_ue_score)
        if "N/A" in str(att_generation):
            att_ue_score = 0.0
        else:
            att_ue_score = 1.0 - (att_ue_score - min_val) / (max_val - min_val)
    
        print(att_ue_score)
           
        s = grad.Argument(f"S{name}", sup, float(sup_ue_score))
        a = grad.Argument(f"A{name}", att, float(att_ue_score))
        
        self.argument_tree.add_support(s, parent)
        self.argument_tree.add_attack(a, parent)

        return s, a
        
    def generate_args_for_parent(self, parent, name, base_score_generator):
        s_prompt, s_constraints, s_format_args = self.generate_prompt(
            parent.get_arg(), support=True
        )
        sup = s_format_args(
            self.llm_manager.chat_completion(
                s_prompt,
                print_result=True,
                trim_response=True,
                **s_constraints,
                **self.generation_args,
            ),
            s_prompt,
        )
 
        a_prompt, a_constraints, a_format_args = self.generate_prompt(parent.get_arg())
        att = a_format_args(
            self.llm_manager.chat_completion(
                a_prompt,
                print_result=True,
                trim_response=True,
                **a_constraints,
                **self.generation_args,
            ),
            a_prompt,
        )
   
        sup_base_score = base_score_generator(sup, claim=parent.get_arg(), support=True)
        att_base_score = base_score_generator(
            att, claim=parent.get_arg(), support=False
        )
        
        s = grad.Argument(f"S{name}", sup, float(sup_base_score))
        a = grad.Argument(f"A{name}", att, float(att_base_score))
        self.argument_tree.add_support(s, parent)
        self.argument_tree.add_attack(a, parent)
       
        return s, a
    
    
    def generate_arguments_lm_polygraph(self, statement, model, ue_method, base_score_generator, min_val, max_val):
        """ ISO Contribution: New function for ISO Uncertainty Quantification Experiments
        Sets the topic confidence score in the same way as the original ArgLLM experiments, then calls the new 
        generate_args_for_parent function with specific UQ (ue) method passed in """
        self.argument_tree = grad.BAG()
        topic = grad.Argument(f"db0", statement, 0.5)

        topic_base_score = base_score_generator(statement, topic=True)
        previous_layer = []
        for d in range(1, self.depth + 1):
            for b in range(1, self.breadth + 1):
                if d == 1:
                    s, a = self.generate_args_for_parent_lm_polygraph(
                        topic, f"db0←d{d}b{b}", model, ue_method, min_val, max_val
                    )
                    previous_layer.append(s) if s.arg != "N/A" else ""
                    previous_layer.append(a) if a.arg != "N/A" else ""
                else:
                    temp = []
                    for p in previous_layer:
                        s, a = self.generate_args_for_parent_lm_polygraph(
                            p, f"{p.name}←d{d}b{b}", model, ue_method
                        )
                        temp.append(s)
                        temp.append(a)
                    previous_layer = temp

        topic_base_score_bag = deepcopy(self.argument_tree)
        # topic base score is from the generator
        topic_base_score_bag.arguments[topic.name].reset_initial_weight(
            topic_base_score
        )


        return self.argument_tree, topic_base_score_bag
    
    """Generates arguments for and against a statement, up to the given breadth and depth."""

    def generate_arguments(self, statement, base_score_generator):
        self.argument_tree = grad.BAG()
        topic = grad.Argument(f"db0", statement, 0.5)

        topic_base_score = base_score_generator(statement, topic=True)
        previous_layer = []
        for d in range(1, self.depth + 1):
            for b in range(1, self.breadth + 1):
                if d == 1:
                    s, a = self.generate_args_for_parent(
                        topic, f"db0←d{d}b{b}", base_score_generator
                    )
                    previous_layer.append(s) if s.arg != "N/A" else ""
                    previous_layer.append(a) if a.arg != "N/A" else ""
                else:
                    temp = []
                    for p in previous_layer:
                        s, a = self.generate_args_for_parent(
                            p, f"{p.name}←d{d}b{b}", base_score_generator
                        )
                        temp.append(s)
                        temp.append(a)
                    previous_layer = temp

        topic_base_score_bag = deepcopy(self.argument_tree)
        # topic base score is from the generator
        topic_base_score_bag.arguments[topic.name].reset_initial_weight(
            topic_base_score
        )

        return self.argument_tree, topic_base_score_bag

    """If argument is similar to other arguments in same branch then we cut of that argument."""

    def cut_arguments(self, arguments):
        pass
