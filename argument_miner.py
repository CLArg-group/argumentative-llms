from copy import deepcopy

import Uncertainpy.src.uncertainpy.gradual as grad


class ArgumentMiner:
    def __init__(
        self, llm_manager, generate_prompt, depth=1, breadth=1, generation_args={}
    ):
        self.depth = depth
        self.breadth = breadth
        self.llm_manager = llm_manager
        self.generate_prompt = generate_prompt
        self.generation_args = generation_args

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

    """Generates arguments for and against a statement, up to the given breadth and depth."""

    def generate_arguments(self, statement, base_score_generator):
        """Generates arguments for and against a statement, up to the given breadth and depth."""
        self.argument_tree = grad.BAG()
        topic = grad.Argument(f"db0", statement, 0.5)
        topic_base_score = base_score_generator(statement, topic=True)
        
        previous_layer = []
        
        for d in range(1, self.depth + 1):
            new_layer = []
            
            if d == 1:
                for b in range(1, self.breadth + 1):
                    s, a = self.generate_args_for_parent(
                        parent=topic,
                        name=f"db0←d{d}b{b}",
                        base_score_generator=base_score_generator
                    )
                    
                    if s.arg != "N/A":
                        new_layer.append(s)
                    if a.arg != "N/A":
                        new_layer.append(a)
            else:
                for p in previous_layer:
                    for b in range(1, self.breadth + 1):
                        try:
                            s, a = self.generate_args_for_parent(
                                parent=p,
                                name=f"{p.name}←d{d}b{b}",
                                base_score_generator=base_score_generator
                            )
                            print(f"Support for {p.name}: {s.arg}")
                            print(f"Attack for {p.name}: {a.arg}")
                            
                            if s.arg != "N/A":
                                new_layer.append(s)
                            if a.arg != "N/A":
                                new_layer.append(a)
                        except Exception as e:
                            print(f"Error generating arguments for {p.name}: {e}")
                            continue

            previous_layer = new_layer

        topic_base_score_bag = deepcopy(self.argument_tree)
        
        if topic.name in topic_base_score_bag.arguments:
            topic_base_score_bag.arguments[topic.name].reset_initial_weight(
                topic_base_score
            )

        return self.argument_tree, topic_base_score_bag

    """If argument is similar to other arguments in same branch then we cut of that argument."""

    def cut_arguments(self, arguments):
        pass
