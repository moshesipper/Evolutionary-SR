# Binary and Multinomial Classification through Evolutionary Symbolic Regression
# copyright 2022 moshe sipper
# www.moshesipper.com

from random import random, randint, choice, choices #uniform, gauss, 
from copy import deepcopy 
from sys import exit
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from inspect import getfullargspec
# from IPython.display import Image, display
# from graphviz import Digraph, Source 

MIN_INIT_DEPTH = 2 # minimal initial random tree depth
MAX_INIT_DEPTH = 4 # maximal initial random tree depth
XO_RATE = 0.8 # crossover rate
P_MUT = 0.3 # mutation probability
ELITISM = 2 # how many top individuals to copy to next generation, must be even

def arity(func): return len(getfullargspec(func)[0])

# function set
def f_add(x, y): return np.add(x, y)
def f_sub(x, y): return np.subtract(x, y)
def f_mul(x, y): return np.multiply(x, y)
def f_div(x, y): return x/np.sqrt(1+y*y)
def f_sqrt(x): return np.sqrt(np.absolute(x))
def f_log(x): 
    x = np.abs(x)
    x = np.where(np.less(x, 0.0001), 1, x)
    return np.log(x)
def f_abs(x): return np.abs(x)
def f_neg(x): return np.negative(x)
def f_min(x, y): return np.minimum(x, y)
def f_max(x, y): return np.maximum(x, y)
def f_ifgte(x, y, z): return np.where(np.greater_equal(x, np.zeros(x.shape)), y, z)
def f_iflt(x, y, z): return np.where(np.less(x, np.zeros(x.shape)), y, z)
FUNCTIONS = [f_add, f_sub, f_mul, f_div, f_sqrt, f_log, f_abs, f_neg, f_min, f_max, f_ifgte, f_iflt]

# terminal set
TERMINALS = [] # set below in GPClassify init

class GPTree:
    
    def __init__(self, data = None, left = None, middle = None, right = None):
        self.data = data
        self.left = left
        self.middle = middle
        self.right = right

    def node_label(self): # string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def print_tree(self, prefix = ""): # textual printout
        print("%s%s" % (prefix, self.node_label()))        
        if self.left: self.left.print_tree(prefix + "   ")
        if self.middle: self.middle.print_tree(prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    '''
    def node_label(self): # return string label
        if self.data == majority:
            return 'majority\n[' + str(round(self.thresholds[0],3)) +\
                              ',' + str(round(self.thresholds[1],3)) +\
                              ',' + str(round(self.thresholds[2],3)) + ']'
        if self.data in FUNCTIONS:
            return self.data.__name__
        else: 
            return str(self.data)
    
    def draw(self, dot, count): # dot & count are lists in order to pass "by reference" 
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())
        if self.left:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        if self.middle:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.middle.draw(dot, count)
        if self.right:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.right.draw(dot, count)
        
    def draw_tree(self, fname, footer):
        dot = [Digraph()]
        dot[0].attr(kw='graph', label = footer)
        count = [0]
        self.draw(dot, count)
        Source(dot[0], filename = fname, format="pdf").render()
        if gl.running_in_spyder():
            Source(dot[0], filename = fname, format="png").render()
            display(Image(filename = fname + ".png"))
    '''

    def compute_tree(self, X): # X: ndarray of shape (n_samples, n_features)
        if (self.data in FUNCTIONS): 
            a = arity(self.data)
            if a == 1:
                return self.data(self.left.compute_tree(X))
            elif a == 2:                                 
                return self.data(self.left.compute_tree(X),\
                                 self.right.compute_tree(X))
            elif a == 3:                                 
                return self.data(self.left.compute_tree(X),\
                                 self.middle.compute_tree(X),\
                                 self.right.compute_tree(X))
        elif (self.data in TERMINALS): return X[:,self.data]
        else: exit("compute_tree: unknown tree node")

    '''        
    def nodes_in_tree(self, nodes): # check that all nodes in 'nodes' are in the tree
        if self.data in nodes: del nodes[nodes.index(self.data)]
        if len(nodes) == 0: return True
        if self.data in FUNCTIONS: 
            if self.left.nodes_in_tree(nodes): return True
            if self.middle: 
                if self.middle.nodes_in_tree(nodes): return True
            if self.right: return self.right.nodes_in_tree(nodes)
        else: return False
    '''
            
    def rand_tree(self, depth = 0): # create random tree using either grow or full method
        if depth < MIN_INIT_DEPTH: 
            self.data = choice(FUNCTIONS)
        elif depth >= MAX_INIT_DEPTH:   
            self.data = choice(TERMINALS)
        elif random () > 0.5: 
            self.data = choice(FUNCTIONS)
        else: 
            self.data = choice(TERMINALS)
        
        if self.data in FUNCTIONS:
            a = arity(self.data)
            self.left = GPTree()
            self.left.rand_tree(depth = depth + 1) 
            if a >= 2: 
                self.right = GPTree()
                self.right.rand_tree(depth = depth + 1)
            if a >= 3: 
                self.middle = GPTree()
                self.middle.rand_tree(depth = depth + 1)
            if (a > 3): exit(f'random_tree: error in arity ({a})')
        # print('>>',self.data, self.left, self.middle, self.right)

    '''
    def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if self.root: # do nothing, root node already has majority function from constructor
            None
        elif depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        if self.data in FUNCTIONS:
            a = arity(self.data)
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1) 
            if a >= 2:                
                self.right = GPTree()
                self.right.random_tree(grow, max_depth, depth = depth + 1)
            if a >= 3:                
                self.middle = GPTree()
                self.middle.random_tree(grow, max_depth, depth = depth + 1)
            if (a > 3): exit("random_tree: error in arity")    
    '''

    def mutation(self):
        if random() < P_MUT: # mutate at this node
            self.rand_tree()
        elif self.left: self.left.mutation()
        elif self.middle: self.middle.mutation()
        elif self.right: self.right.mutation() 
        
    def size(self): # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size() if self.left else 0
        m = self.middle.size() if self.middle else 0
        r = self.right.size() if self.right else 0
        return 1 + l + m + r

    def build_subtree(self): 
        t = GPTree()
        t.data = self.data
        if self.left: t.left = self.left.build_subtree()
        if self.middle: t.middle = self.middle.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list in order to pass "by reference"
        count[0] -= 1            
        if count[0] <= 1: 
            if second==None: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data = second.data
                self.left = second.left
                self.middle = second.middle
                self.right = second.right
        else:  
            ret = None              
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)  
            if self.middle and count[0] > 1: ret = self.middle.scan_tree(count, second)  
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree

    '''
    def prune_tree(self, size): # prune tree if > size
        queue = [self]
        s = 1
        while len(queue) != 0:
            q_head = queue[0]
            del queue[0]
            if s >= size:
                q_head.left   = None
                q_head.middle = None
                q_head.right  = None    
                if q_head.data in FUNCTIONS: q_head.data = TERMINALS[randint(0, len(TERMINALS)-1)]                                    
            if q_head.left != None: 
                queue.append(q_head.left)
                s += 1
            if q_head.middle != None: 
                queue.append(q_head.middle)
                s += 1
            if q_head.right != None: 
                queue.append(q_head.right)
                s += 1
    '''
# end class GPTree

def softmax(x):
	e = np.exp(x)
	return e / e.sum()

def cross_entropy_loss(y_pred, y):
    # y is one-hot encoded
    y_pred = np.where(np.equal(y_pred, np.zeros(y_pred.shape)), 1e-10, y_pred) # avoid log(0)
    y_pred = np.where(np.equal(y_pred, np.ones(y_pred.shape)), (1-1e-10), y_pred) # avoid log(1-1)
    return np.where(np.equal(y, np.ones(y.shape)), -np.log(y_pred), -np.log(np.ones(y.shape) - y_pred))

class GPClassify:
    
    def __init__(self, n_pop, n_gens, n_pars, n_tour):
        self.n_pop = n_pop
        self.n_gens = n_gens
        self.n_pars = n_pars
        self.n_tour = n_tour
        self.best_of_run = None
    
    def init_pop(self, n_classes):
        subpops = []
        for cl in range(n_classes):
            subpop = []
            for i in range(self.n_pop):
                t = GPTree()
                t.rand_tree()
                subpop.append((t,1e3)) # each individual is a tuple (tree,fitness)
            subpops.append(subpop)
        return subpops
                
    def selection(self, pop): # select one individual using tournament selection
        tournament = choices(range(len(pop)), k=self.n_tour)
        fits = [pop[t][1] for t in tournament]
        winner_i = tournament[fits.index(min(fits))] 
        return deepcopy(pop[winner_i])

    def fitness(self, ind, y_pred, y): # minimize loss + size
        return np.mean(cross_entropy_loss(y_pred, y)) + self.n_pars*ind.size()

    def fit(self, X, y):
        # print(f'>>> self.n_pop {self.n_pop} self.n_gens {self.n_gens} self.n_pars {self.n_pars} self.n_tour {self.n_tour}')
        n_feat = X.shape[1]
        n_classes = len(np.unique(y))
        global TERMINALS
        TERMINALS = list(range(n_feat))
        yhot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1))
        
        subpops = prev_subpops = self.init_pop(n_classes)
            
        best_f = 1e3 # arbitrary large value, since we're minimizing fitness
        for gen in range(self.n_gens):
            current_subpops = []
            next_subpops = []
                        
            prev = [np.argmin([ind[1] for ind in subpop]) for subpop in prev_subpops] # index of prev-gen best individual per subpop
            prev_ind = [subpop[i][0] for (subpop,i) in zip(prev_subpops,prev)] # save best ind from prev gen of each subpop
            prev_out = [ind.compute_tree(X) for ind in prev_ind] # compute outputs of each ind in prev_ind
            
            for cl in range(n_classes): # evolve each of the n_classes subpops
                
                current_subpop = []
                for tupl in subpops[cl]: # compute fitness of each individual tree in subpop cl
                    tree = tupl[0] # tupl is (tree,fitness)
                    outputs = []
                    for i in range(n_classes):
                        if i!=cl: outputs.append(prev_out[i]) # best prev-gen output from the other classes
                        else: outputs.append(tree.compute_tree(X)) # current tree's output for this class
                    y_pred = np.array([softmax(row) for row in np.array(outputs).T]).T # do softmax on columns by using T
                    fitness = self.fitness(tree, y_pred[cl,:], yhot[:,cl])
                    current_subpop.append((tree, fitness))

                    if fitness < best_f:
                        best_f = fitness
                        self.best_of_run = []
                        for i in range(n_classes):
                            if i!=cl: self.best_of_run.append(prev_ind[i])
                            else: self.best_of_run.append(tree)
            
                current_subpops.append(current_subpop)

                current_subpop = sorted(current_subpop, key = lambda x: x[1])                
                next_subpop = []
                for i in range(ELITISM):        
                    next_subpop.append(deepcopy(current_subpop[i]))
                
                for i in range(int(self.n_pop/2) - int(ELITISM/2)):
                    parent1 = self.selection(current_subpop)[0]
                    parent2 = self.selection(current_subpop)[0]
                    save_p1 = deepcopy(parent1)
                    parent1.crossover(parent2)
                    parent1.mutation()
                    next_subpop.append((parent1,None))
                    parent2.crossover(save_p1)
                    parent2.mutation()
                    next_subpop.append((parent2,None))
                
                next_subpops.append(next_subpop)
            
            prev_subpops = current_subpops
            subpops = next_subpops

    def predict(self, X):
        preds = [ind.compute_tree(X) for ind in self.best_of_run]
        preds = np.array(preds)
        return np.argmax(preds, axis=0)

# end class GPClassify


                    