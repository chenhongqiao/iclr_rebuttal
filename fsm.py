class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class FunctionNameFSM():
    def __init__(self, functions, tokenizer, end_tokens):
        self.functions = functions
        self.tokenizer = tokenizer
        self.cur_str = ""
        self.end_tokens = end_tokens
        self.trie = self.build_trie(self.functions)
        self.cand_ids = np.arange(32000)
        self.cand_tokens = []
        for id in self.cand_ids.tolist():
            self.cand_tokens.append(self.tokenizer.decode(id))
    
    def build_trie(self, words):
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True
        return root

    def search_trie(self, prefix):
        node = self.trie
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return False
        return True
    
    def __call__(self, logits):
        cand_prefixes = np.char.add(np.array(self.cur_str), np.array(self.cand_tokens))

        if self.cur_str in self.functions:
            mask = np.isin(self.cand_ids, self.end_tokens) | np.array([self.search_trie(prefix) for prefix in cand_prefixes])
        else:
            mask = np.array([self.search_trie(prefix) for prefix in cand_prefixes]) & np.char.not_equal(cand_prefixes,np.array(self.cur_str))

        logits[-1, ~mask] = -1e5
        return logits

    def push(self, token):
        self.cur_str += self.tokenizer.decode([int(token)])