from types import new_class
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Uniform, Normal

from daphne import daphne
import numpy as np
import time

from primitives import PRIMITIVES
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
# env = {'normal': dist.Normal,
#        'sqrt': torch.sqrt}
env = PRIMITIVES

def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) in [int, float]:
        # We use torch for all numerical objects in our evaluator
        return torch.Tensor([float(exp)]).squeeze()
    elif type(exp) is torch.Tensor:
        return exp
    elif type(exp) is bool:
        return torch.tensor(exp)
    else:
        print("expression is:", exp)
        print(type(exp))
        raise Exception("Expression type unknown.", exp)

def topological_sort(nodes, edges):
    result = []
    visited = {}
    def helper(node):
        if node not in visited:
            visited[node] = True
            if node in edges:
                for child in edges[node]:
                    helper(child)
            result.append(node)
    for node in nodes:
        helper(node)
    return result[::-1]

def plugin_parent_values(expr, trace):
    if type(expr) == str and expr in trace:
        return trace[expr]
    elif type(expr) == list:
        return [plugin_parent_values(child_expr, trace) for child_expr in expr]
    else:
        return expr

def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    """
    1. Run topological sort on V using V and A, resulting in an array of v's
    2. Iterate through sample sites of the sorted array, and save sampled results on trace dictionary using P and Y
    - If keyword is sample*, first recursively replace sample site names with trace values in the expression from P. Then, run deterministic_eval.
    - If keyword is observe*, put the observation value in the trace dictionary
    3. Filter the trace dictionary for things sample sites you should return
    """
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    sigma = {}
    trace = {}
    for node in sorted_nodes:
        keyword = links[node][0]
        if keyword == "sample*":
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            dist_obj  = deterministic_eval(link_expr)
            trace[node] = dist_obj.sample()
        elif keyword == "observe*":
            trace[node] = obs[node]

    expr = plugin_parent_values(expr, trace)
    return deterministic_eval(expr), sigma, trace


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../HW3/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')



def run_probabilistic_tests():
    
    #TODO: 
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../HW3/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        print(p_val > max_p_value)
    
    print('All probabilistic tests passed')    




def MH_Gibbs(graph, numsamples):
    model = graph[1]
    vertices = model['V']
    arcs = model['A']
    links = model['P'] # link functions aka P

    # sort vertices for ancestral sampling
    V_sorted = topological_sort(vertices,arcs)

    def accept(x, cX, cXnew, Q):
        # compute acceptance ratio to decide whether
        # we keep cX or accept a new sample/trace cXnew
        # cX and cXnew are the proposal mappings (dictionaries)
        # which assign values to latent variables

        # cXnew corresponds to the values for the new samples

        # take the proposal distribution for the current vertex
        # this is Q(x)
        Qx = Q[x][1]

        # we will sample from this with respect to cX and cXnew

        # the difference comes from how we evaluate parents 
        # plugging into eval
        p = plugin_parent_values(Qx,cX)
        pnew = plugin_parent_values(Qx,cXnew)

        # p = Q(x)[X := \mathcal X]
        # p' = Q(x)[X := \mathcal X']
        # note that in this case we only need to worry about
        # the parents of x to sample from the proposal


        # evaluate 
        d = deterministic_eval(p) # d = EVAL(p)
        dnew = deterministic_eval(pnew) #d' = EVAL(p')

        ### compute acceptance ratio ###

        # initialize log alpha
        logAlpha = dnew.log_prob(cXnew[x]) - d.log_prob(cX[x])

        ### V_x = {x} \cup {v:x \in PA(v)} ###
        startindex = V_sorted.index(x)
        Vx = V_sorted[startindex:]

        # compute alpha
        for v in Vx:
            Pv = links[v] # getting a bug here
            v_exp = plugin_parent_values(Pv,cX) #same as we did for p and pnew
            v_exp_new = plugin_parent_values(Pv,cXnew)
            dv_new = deterministic_eval(v_exp_new[1])
            dv = deterministic_eval(v_exp[1])

            
            ## change below
            logAlpha = logAlpha + dv_new.log_prob(cXnew[v])
            logAlpha = logAlpha - dv.log_prob(cX[v])
        return torch.exp(logAlpha)


    
    def Gibbs_step(cX,Q):
        # here we need a list of the latent (unobserved) variables
        Xobsv = list(filter(lambda v: links[v][0] == "sample*", V_sorted))

        for u in Xobsv:
            # here we are doing the step
            # d <- EVAL(Q(u) [X := \cX]) 
            # note it suffices to consider only the non-observed variables
            Qu = Q[u][1]
            u_exp = plugin_parent_values(Qu,cX)
            dist_u = deterministic_eval(u_exp).sample()
            cXnew = {**cX}
            cX[u] = dist_u

            #compute acceptance ratio
            alpha = accept(u,cX,cXnew,Q)
            val = Uniform(0,1).sample()

            if val < alpha:
                cX = cXnew
        return cX


    Q = links # initialize the proposal with P (i.e. using the prior)
    cX_list = [ sample_from_joint(graph)[2] ] # initialize the state/trace

    for i in range(1,numsamples):
        cX_0 = {**cX_list[i-1]} #make a copy of the trace
        cX = Gibbs_step(cX_0,Q)
        cX_list.append(cX)
    
    samples = list(map(lambda cX: deterministic_eval(plugin_parent_values(graph[2], cX)), cX_list))
    #samples = [ deterministic_eval(plugin_parent_values(graph[2],X)) for X in cX_list ]

    return samples



'''
def compute_log_joint(sorted_nodes, links, trace_dict):
    joint_log_prob = 0.
    for node in sorted_nodes:
        link_expr = links[node][1]
        dist      = deterministic_eval(plugin_parent_values(link_expr, trace_dict))
        joint_log_prob += dist.log_prob(trace_dict[node])
    return joint_log_prob
'''

if __name__ == '__main__':
    

    print("==== running program 1 =====")
    graph = daphne(['graph','-i','../HW3/programs/1.daphne'])
    start = time.time()
    samples = MH_Gibbs(graph, 20000)
    end = time.time()
    print("Program 1 mean: ", np.mean(samples))
    print("Program 1 variance: ", np.var(samples))
    print("Total run time: ", end-start)
    fig,ax = plt.subplots()
    ax.plot(samples)
    fig.savefig('../HW3/p1traceplot',dpi = 150)

    figH,axH = plt.subplots()
    axH.hist(samples)
    figH.savefig('../HW3/p1histogram',dpi = 150)
    #plt.plot(list(range(0,len(samples))), samples)
    #plt.show()



    print(" \n \n ==== running program 2 =====")
    graph = daphne(['graph','-i','../HW3/programs/2.daphne'])
    #print(graph)
    start = time.time()
    samples = MH_Gibbs(graph, 10000)
    end = time.time()
    samples1 = [ s[0] for s in samples]
    samples2 = [s[1] for s in samples]
    print("Program 2 slope mean is: ",np.mean(samples1))
    print("Program 2 bias mean is: ", np.mean(samples2))
    print("Program 2 slope variance is: ",np.var(samples1))
    print("Program 2 bias variance is: ", np.var(samples2))
    print("Total run time: ", end-start)
    fig1,ax1 = plt.subplots()
    ax1.plot(samples1)
    fig1.savefig('../HW3/p2traceplot_slope',dpi = 150)

    fig2,ax2 = plt.subplots()
    ax2.plot(samples1)
    fig2.savefig('../HW3/p2traceplot_bias',dpi = 150)

    figH,axH = plt.subplots()
    axH.hist(samples1)
    figH.savefig('../HW3/p2histogram_slope',dpi = 150)

    figH,axH = plt.subplots()
    axH.hist(samples2)
    figH.savefig('../HW3/p1histogram_bias',dpi = 150)

    print(" \n \n ==== running program 3 ====")
    graph = daphne(['graph','-i','../HW3/programs/3.daphne'])
    start = time.time()
    samples = MH_Gibbs(graph, 9000)
    end = time.time()
    print("Program 3 mean is: ", np.mean(samples))
    print("Program 3 variance is: ", np.var(samples))
    print("total run time: ", end-start)
    figH,axH = plt.subplots()
    axH.hist([float(s) for s in samples])
    figH.savefig('../HW3/p3histogram',dpi = 150)




    print(" \n \n ==== running program 4 ====")
    graph = daphne(['graph','-i','../HW3/programs/4.daphne'])
    start = time.time()
    samples = MH_Gibbs(graph, 80000)
    end = time.time()
    print("Program 4 mean is: ", np.mean(samples))
    print("Program 4 variance is: ", np.var(samples))
    print("total run time: ", end-start)
    figH,axH = plt.subplots()
    axH.hist([float(s) for s in samples])
    figH.savefig('../HW3/p4histogram',dpi = 150)

    '''

    for i in range(3,5):
        graph = daphne(['graph','-i','../HW3/programs/{}.daphne'.format(i)])
        samples = MH_Gibbs(graph, 10000)
        #print(samples)
        print("the mean for program ", i, " is: ", np.mean(samples))
        print("the variance for program ", i, " is: ", np.var(samples))


    
    for i in range(1,5):
        graph = daphne(['graph','-i','../HW3/programs/{}.daphne'.format(i)])
        samples, n = [], 1000
        for j in range(n):
            sample = sample_from_joint(graph)[0]
            samples.append(sample)

        print(f'\nExpectation of return values for program {i}:')
        if type(samples[0]) is list:
            expectation = [None]*len(samples[0])
            for j in range(n):
                for k in range(len(expectation)):
                    if expectation[k] is None:
                        expectation[k] = [samples[j][k]]
                    else:
                        expectation[k].append(samples[j][k])
            for k in range(len(expectation)):
                print_tensor(sum(expectation[k])/n)
        else:
            expectation = sum(samples)/n
            print_tensor(expectation)
    '''