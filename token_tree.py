import numpy as np

class TokenTree:
    def __init__(self, text: str):
        self.token = ""
        self.text = text
        self.logprob = 0
        self.total_logprob = 0
        self.depth = 0
        self.children = {}
        self.root = self
        self.label = text
        self.total_node_count = 0
        self.max_depth = 0

    def add_child(self, token, logprob):
        child = TokenTree(self.text + token)

        child.token = token
        child.depth = self.depth + 1
        child.logprob = logprob
        child.total_logprob = self.total_logprob + logprob
        child.label = token
        self.children[child.token] = child

        child.root = self.root
        self.root.total_node_count += 1
        self.root.max_depth = max(self.root.max_depth, child.depth)

        print(f"{self.token} -> {token}[{np.exp(logprob):.2f}]")

        return child
    
    def __lt__(self, o):
        return self.total_logprob > o.total_logprob

