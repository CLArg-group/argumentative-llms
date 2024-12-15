class Approximator:
    graph_data = {}

    def __init__(self, ads, time=0, arguments=[], argument_strength=[], attacker=[], supporter=[], name="") -> None:
        self.ads = ads
        self.time = time
        self.arguments = arguments
        self.argument_strength = argument_strength
        self.attacker = attacker
        self.supporter = supporter
        self.name = name

    def initialise_arrays(self):
        argument_set = list(self.ads.BAG.arguments.values())
        arg_to_index = {}

        arguments = []
        argument_strength = []

        for z, a in enumerate(argument_set):
            arguments.append(a)
            argument_strength.append(a.strength)
            arg_to_index[a] = z

        attacker = {}
        supporter = {}

        for z, a in enumerate(argument_set):

            attacker_child = []
            for b in a.attackers:
                attacker_child.append(arg_to_index[b])

            attacker[z] = attacker_child

            supporter_child = []
            for b in a.supporters:
                supporter_child.append(arg_to_index[b])

            supporter[z] = supporter_child

        self.ads.arguments = arguments
        self.ads.argument_strength = argument_strength
        self.ads.attacker = attacker
        self.ads.supporter = supporter

    def rewrite_arrays(self):
        for i in range(len(self.ads.arguments)):
            self.ads.arguments[i].strength = self.ads.argument_strength[i]

    def perform_iteration(delta, epsilon):
        return None

    def initialise_graph_data(self):
        for argument in self.ads.arguments:
            self.graph_data[argument.name] = [(0, argument.initial_weight)]

    def update_graph_data(self, time):
        for z, argument in enumerate(self.ads.arguments):
            self.graph_data[argument.name].append((time, self.ads.argument_strength[z]))

    def approximate_solution(self, delta, epsilon, verbose=False, generate_plot=False):
        self.initialise_arrays()

        if generate_plot:
            self.initialise_graph_data()

        time = 0
        time_limit = 99999999999
        max_derivative = 0

        while True:
            max_derivative = self.perform_iteration(delta, epsilon)
            time += delta

            if generate_plot:
                self.update_graph_data(time)

            if(max_derivative < epsilon or time >= time_limit):
                break

        self.rewrite_arrays()

        if (verbose):
            print_args = '\n'.join([str(x) for x in self.ads.arguments])
            print(f"{self.ads.name}, {self.ads.approximator.name}\nTime: {time}\n{print_args}\n")

        return max_derivative
