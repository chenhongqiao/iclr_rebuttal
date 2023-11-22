class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False
        self.mask = None
        

class FunctionNameFSM():
    def __init__(self, functions, tokenizer, end_tokens):
        self.tokenizer = tokenizer
        self.end_tokens = end_tokens
        self._build_trie(functions)
        self._compute_mask()
        
    
    def _build_trie(self, functions):
        self.root = TrieNode()
        tokenized_func = [tokenizer.encode(s, bos=False, eos=False) for s in functions]
        for tokens in tokenized_func:
            node = self.root
            for token in tokens:
                if token not in node.children:
                    node.children[token] = TrieNode()
                node = node.children[token]
            node.end_of_word = True
    
    def _compute_mask(self):
        queue = deque()
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.popleft()
            node.mask = torch.zeros((32000), dtype=bool)
            node.mask[list(node.children.keys())] = 1
            if node.end_of_word:
                node.mask[self.end_tokens] = 1
            queue.extend(node.children.values())
        
        
    def __call__(self, logits):
        logits[-1, ~self.root.mask] = -1e5
        return logits


    def push(self, token):
        token = int(token)
        self.root = self.root.children[token]
