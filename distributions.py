from token_tree import TokenTree
import numpy as np

LN2 = np.log(2)

def get_tree_distribution(tree: TokenTree, max_depth = 9999):
    distribution = {}
    get_subtree_distribution(tree, distribution, max_depth)
    return distribution
    
def get_subtree_distribution(tree: TokenTree, distribution, max_depth):
    if(tree.depth > max_depth):
        return
    id = f"{tree.token}[{tree.depth}]"
    if not id in distribution:
        distribution[id] = tree.total_logprob
    else:
        distribution[id] = np.logaddexp(distribution[id], tree.total_logprob)

    for child in tree.children.values():
        get_subtree_distribution(child, distribution, max_depth)

def get_mixture_distribution(P, Q):
    M = {}
    for obs in P:
        p = P[obs]
        q = Q.get(obs, -9999.0)
        M[obs] = np.logaddexp(p, q) - LN2

    for obs in Q:
        if obs in P:
            continue
        p = -9999.0
        q = Q[obs]
        M[obs] = np.logaddexp(p, q) - LN2
    
    return M

def KL_divergence(P, Q):
    sum = 0
    for obs in P:
        log_p = P[obs]
        log_q = Q.get(obs, -9999.0)
        sum += kl_inner(log_p, log_q)

    for obs in Q:
        if obs in P:
            continue
        log_p = -9999.0
        log_q = Q[obs]
        sum += kl_inner(log_p, log_q)

    return sum

def kl_inner(log_q, log_p):
    log_odds = log_p - log_q
    p = np.exp(log_p)
    return p*log_odds

def JS_divergence(P, Q):
    M = get_mixture_distribution(P, Q)
    return 0.5*(KL_divergence(P, M) + KL_divergence(Q, M))