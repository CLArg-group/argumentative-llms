from numpy import array, sum
from random import randrange
from sklearn.preprocessing import normalize
from time import time

class MonteCarloSampler:
    """ MonteCarloSampler for approximating sufficient and necessary explanations for random forests
    and the percentage of non-ambiguous equivalence classes. The sampler has to be initialized with lists
    of feature names, their corresponding types, and the random forest that is to be explained.
    The type can be categorical (including boolean) or continuous and should be declared using the class 
    attributes type_cat or type_cont. 
    """
    
    type_cat = 0
    type_cont = 1
    
    def __init__(self, feature_names, feature_types, class_names, rf):
        
        self.n_features = len(feature_names)
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.feature_partitions = [[]] * self.n_features
        self.computePartitions(rf)
        
        self.n_classes = len(class_names)
        self.class_names = class_names
        
        self.rf = rf
        
    def sample(self, no_samples, min_no_estimates=20):
        """
        Approximate default queries for random forest. 
        
        min_no_estimates: minimal number of samples required to estimate a probability (not implemented yet)
        """
        self.sampleAtomicAndAmbiguous(no_samples, min_no_estimates)
        
    def evaluateSample(self, x):
        """
        Compute prediction and return pair (ambiguous, max_class), where
            ambiguous is True if the sample is ambiguous and False otherwise, and
            max_class is -1 if the input is ambiguous and the class index of the predicted class otherwise.
        """
        
        #get class probabilities (relative frequency of votes) 
        class_probs = self.rf.predict_proba([x])[0]
        
        #determine if input is non-ambiguous and determine class with max number of votes
        max_p = -1
        max_class = -1
        for c, p in enumerate(class_probs):
        
            if p >= max_p:
                if(max_p > 0 and max_p == p):
                    return (True, -1)
                max_p = p
                max_class = c
        
        return False, max_class
    
    def sampleAtomicAndAmbiguous(self, no_samples, min_no_estimates):
        """
        Approximate atomic necessary and sufficient queries for random forest and the percentage of nonambiguous inputs. 
        """
        
        #statistics
        stat_inp = MonteCarloSampler.InputStatistic()
        stat_feat_class = {}
        
        for i in range(self.n_features):
            stat_feat_class[i] = MonteCarloSampler.FeatureClassStatistic(self.feature_partitions[i].N, self.n_classes)
        
        #lists for samples of equivalence class representatives and equivalence class indices
        x = list(range(self.n_features))
        eq = list(range(self.n_features))       
        #list for class frequencies
        class_counts = [0]*self.n_classes
        
        print("Start Approximating Percentage of Nonambiguous Inputs and Atomic Queries")
        start_time = time()
        time_since_update = start_time
        
        for j in range(no_samples):
        
            #create sample
            for i in range(self.n_features):
                x[i], eq[i] = self.feature_partitions[i].createSample()


            ambiguous, max_class = self.evaluateSample(x)
            
            #store statistics
            if(ambiguous):
                stat_inp.countAmbiguous()
                
            else:
                stat_inp.countNonambiguous()
                class_counts[max_class] = class_counts[max_class] + 1
                
                for i in range(self.n_features):
                    stat_feat_class[i].count(eq[i], max_class)
                    
            #print information regularly
            cur_time = time()
            if (cur_time - time_since_update > 5):
                print(f"  ... sampling in progress ... completed {j}/{no_samples} samples ... estimated time remaining: {int((cur_time - start_time) * (no_samples-j)/j)} seconds ...")
                time_since_update =  cur_time               
                
                
                
        end_time = time()
        print(f"Sampling finished after {end_time - start_time} seconds.\n")
        
        
        
        print("\nEstimates:\n")   
        
        #print input ambiguity statistics
        stat_inp.printStatistics()
        
        #print atomic sufficient and necessary statistics 
        for i in range(self.n_features):
            print(f"\nFeature {self.feature_names[i]}:")
            stat_feat_class[i].printStatistics(self.feature_names[i], self.feature_partitions[i].getSampleDomain(), self.class_names, class_counts)
            
        
        print("\n\n\n")
            
        #pairwise sampling
        self.samplePairwise()
        
    def samplePairwise(self, delta = 0.9, no_pairwise_samples=100):
        """
        Approximate pairwise sufficient queries for random forest. 
        """
        
        print("\n\n\nTry to find pairwise delta-sufficient reasons.\n")
        
        #lists for samples of equivalence class representatives and equivalence class indices
        x = list(range(self.n_features)) 
        eq = list(range(self.n_features))   
        
        #Sampling: for every feature
        for f1 in range(self.n_features):
        
            print(f"\n Scanning Feature {self.feature_names[f1]}\n")  
            
            #for every value that the feature can take
            for f1val in range(self.feature_partitions[f1].N):
                
                #fix value of f1 in sample
                x[f1], eq[f1] = self.feature_partitions[f1].sampleDomain[f1val], f1val
                
                #for every other feature
                for f2 in range(f1+1, self.n_features):
                
                    print(f"\n  vs Feature {self.feature_names[f2]}\n")  
            
                    
                    #for every value that the other feature can take
                    for f2val in range(self.feature_partitions[f2].N):
                    
                        #approximate marginal probability of corresponding sufficient query
                        
                        #fix value of f2 in sample
                        x[f2], eq[f2] = self.feature_partitions[f2].sampleDomain[f2val], f2val
                        
                        #create samples
                        class_counts = [0] * self.n_classes
                        no_samples = 0
                        for i in range(no_pairwise_samples):
                        
                            #sample remaining values and store statistics
                            
                            for f3 in range(self.n_features):
                                if f1!=f3 and f2!=f3:
                                    x[f3], eq[f3] = self.feature_partitions[f3].createSample()
                                    
                            ambiguous, max_class = self.evaluateSample(x)
                                    
                            if not ambiguous:
                                class_counts[max_class] = class_counts[max_class] + 1
                                no_samples = no_samples + 1
                            
                        #print delta-sufficient reason if sufficient sample size and delta sufficiently large
                        if no_samples < 0.5 * no_pairwise_samples:
                            print(f"   {100*(1 - no_samples/no_pairwise_samples)}% of samples for (\'{self.feature_names[f1]}\'={self.feature_partitions[f1].getSampleDomain()[f1val]}, "
                               + f" \'{self.feature_names[f2]}\'={self.feature_partitions[f2].getSampleDomain()[f2val]}) are ambiguous. Reject estimates because nonambiguous sample size is too small.\n")
                        else:
                            class_probs = normalize([class_counts], axis=1, norm='l1')[0]
                            for i in range(len(class_probs)):
                                if class_probs[i] >= delta:
                                    print(f"   P( {self.class_names[i]} | \'{self.feature_names[f1]}\'={self.feature_partitions[f1].getSampleDomain()[f1val]}, "
                                       + f" \'{self.feature_names[f2]}\'={self.feature_partitions[f2].getSampleDomain()[f2val]})={class_probs[i]} based on {no_samples} samples.\n")
                        
        
        
    def computePartitions(self, rf):
        """
        Compute the domain partitioning 
        """

        #dictionary that stores key-value pairs of the form (feature_id, threshold_set)
        thresholds = list(range(self.n_features))
        for i in range(self.n_features):
            thresholds[i] = set()

        #iterate over all trees
        for dt in rf.estimators_:

            #take tree and traverse starting from root (id 0)
            tree = dt.tree_
            stack = [0]

            # while there are still node ids on the stack, pick the next node
            while stack != []:
                id = stack.pop()

                #if node is an inner node, extract the treshold (ignore leafs)
                if tree.children_left[id] != tree.children_right[id]:

                    thresholds[tree.feature[id]].add(tree.threshold[id])

                    stack.append(tree.children_left[id])
                    stack.append(tree.children_right[id])

        #after extracting all thresholds, sort them        
        for i in range(self.n_features):
            tList = list(thresholds[i])
            tList.sort()
            thresholds[i] = tList 
            partition = MonteCarloSampler.FeaturePartition(self.feature_names[i], i, self.feature_types[i])
            partition.initialize(tList)
            
            self.feature_partitions[i] = partition
    
            
        
    class FeaturePartition:
        """FeaturePartition class: stores the name of a feature, its id and its type. 

        The FeaturePartition has to be initialized with an ordered list of the threshold values that define the partition. 
        The actual initialization depends on whether the feature is categorical or continuous. 
        """   


        def __init__(self, feature_name, feature_id, feature_type):

            self.feature_name = feature_name
            self.feature_id = feature_id
            self.feature_type = feature_type
            #Initialized indicates if the partition has be initialized with the partitioning thresholds
            self.sampleDomain = None
            self.initialized = False

        def initialize(self, thresholds):
            
            if len(thresholds) == 0:
                #happens only if feature was not used in any tree. In this case, we can add an arbitrary value (that will always be sampled) 
                #since it is ignored by the forest. We add None to make clear that the value does not matter 
                self.sampleDomain = [float(-999)]
            elif self.feature_type == MonteCarloSampler.type_cat:
                #if feature is boolean, the only threshold should be 0.5 and we replace it with the truth values 0 and 1
                if len(thresholds) == 1: 
                    self.sampleDomain = [0,1]
                else:
                    raise NotImplementedError("Currently non-boolean categorical features are not encoded. Please one-hot-encode these features.")
            elif self.feature_type == MonteCarloSampler.type_cont:
                #for continuous feature, every threshold is a representative of one equivalence class (the upper bound of the class)
                #we have to add one additional representative for the last equivalence class (max, +infinity), where max is the maximum 
                #that occured in the forest. We just choose max+1 as the representative
                self.sampleDomain = thresholds
                self.sampleDomain.append(thresholds[-1]+1)
                
            self.initialized = True
            self.N = len(self.sampleDomain)
            
        #get representation of sample domain: domains of categorical features consist of original feature domain,
        # domains of continuous features correspond to subintervals of the original feature domain
        def getSampleDomain(self):
            if self.feature_type == MonteCarloSampler.type_cat:
                return self.sampleDomain
            elif self.feature_type == MonteCarloSampler.type_cont:
                dom = list(range(self.N))
                dom[0] = (float('-inf'), self.sampleDomain[0])
                for i in range(1, self.N-1):
                    dom[i] = (self.sampleDomain[i-1],self.sampleDomain[i])
                dom[self.N-1] = (self.sampleDomain[self.N-1], float('inf'))
                return dom
                
            
        #sample uniformly from equivalence classes and return representative and index of the equivalence class
        def createSample(self):
            
            r = randrange(self.N)
            return self.sampleDomain[r], r
            
    class InputStatistic:   
        """
        Manages statistics about ambiguous and non-ambiguous inputs.
        """
    
        def __init__(self):
            self.N_ambiguous = 0
            self.N_nonambiguous = 0
            
        def countAmbiguous(self):
            self.N_ambiguous = self.N_ambiguous + 1
            
        def countNonambiguous(self):
            self.N_nonambiguous = self.N_nonambiguous + 1
            
        def printStatistics(self):
            print(f"  Percentage of nonambiguous input equivalence classes: {self.N_nonambiguous/(self.N_nonambiguous + self.N_ambiguous)}")
            print(f"  Number of ambiguous input equivalence classess found: {self.N_ambiguous}")
           
           
    class FeatureClassStatistic:   
        """
        Manages statistics about co-occurence of feature value classes and class labels to find
        simple delta-sufficient and delta-necessary reasons.
        
        Counts are stored in a table, where rows correspond to feature value classes
        and columns to class labels.
        """
    
        def __init__(self, N_feature_classes, N_classes):
            self.feature_class_table = [[0 for j in range(N_classes)] for i in range(N_feature_classes)] 
            
        def count(self, feature_class_ix, class_ix):
            self.feature_class_table[feature_class_ix][class_ix] = self.feature_class_table[feature_class_ix][class_ix] + 1
            
        def printStatistics(self, feature_name, domain_names, class_names, class_counts, sufficient_threshold=0.9, necessary_threshold=0.6):
        
            self.feature_class_table = array(self.feature_class_table)
            
            #normalize class counts
            class_probs = normalize([class_counts], axis=1, norm='l1')[0]
        
            #get estimates P(class|feature) for delta-sufficient candidates
            print("\n  Delta-sufficient Candidates")
            sufficient_table = normalize(self.feature_class_table, axis=1, norm='l1')
            for i in range(sufficient_table.shape[0]):
                for j in range(sufficient_table.shape[1]):
                    #print sufficient candidate if probability > threshold and probability at least 10% larger (relative) than prior probability of class
                    if sufficient_table[i,j] > sufficient_threshold and sufficient_table[i,j] > class_probs[j]*1.1:
                        print(f"  P( {class_names[j]} | \'{feature_name}\'={domain_names[i]})={sufficient_table[i,j]} based on {sum(self.feature_class_table[i,:])} samples")
            
            #get P(feature|class) for delta-necessary candidates
            print("\n  Delta-necessary Candidates")
            necessary_table = normalize(self.feature_class_table, axis=0, norm='l1')
            for i in range(necessary_table.shape[0]):
                for j in range(necessary_table.shape[1]):
                    if necessary_table[i,j] > necessary_threshold:
                        print(f"  P( \'{feature_name}\'={domain_names[i]} | {class_names[j]})={necessary_table[i,j]} based on {sum(self.feature_class_table[:, j])} samples")
            
            
            