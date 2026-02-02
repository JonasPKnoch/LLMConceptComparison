import heapq
from openai import OpenAI
from token_tree import TokenTree

class TokenTreeGenerator:
    def __init__(self, client: OpenAI, prompt: str, temperature = 1.0):
        self.client = client
        self.prompt = prompt
        self.temperature = temperature

        self.root = TokenTree(prompt)
        self.nodes = [self.root]
        self.unexpanded_nodes_heap = [self.root]

    def get_completion(self, text):
        response = self.client.completions.create(
            model = "deepseek-chat",
            prompt = text,
            temperature = self.temperature,
            max_tokens = 1,
            logprobs=20)
        return response
    
    def expand_best_node(self):
        node = heapq.heappop(self.unexpanded_nodes_heap)
        self.expand_node(node)
        return node

    def expand_node(self, node):
        response = self.get_completion(node.text)
        top_logprobs = response.choices[0].logprobs.top_logprobs[0]

        for token in top_logprobs.keys():
            logprob = top_logprobs[token]

            if logprob > -9999.0:
                child = node.add_child(token, logprob)

                heapq.heappush(self.unexpanded_nodes_heap, child)
                self.nodes.append(child)
