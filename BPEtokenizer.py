from collections import Counter

class BPE_tk:
    def __init__(self, data):
        self.data = data
        self.symbols = list(data)
        self.vocab = sorted(set(data))
        self.count = self.get_pair_counts(self.symbols)
        self.itos = {i:s for i,s in enumerate(self.vocab)}
        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.merges = []

    def get_pair_counts(self, symbols):
        count = Counter()
        for i in range(len(symbols)-1):
            count[(symbols[i], symbols[i+1])] += 1
        return count

    def merge_pairs(self, pair):
        i=0
        new_symbols = []
        merged_symb = pair[0]+pair[1]
        self.merges.append(pair)
        while i<len(self.symbols):
            if i<len(self.symbols)-1 and self.symbols[i] == pair[0] and self.symbols[i+1] == pair[1]:
                new_symbols.append(merged_symb)
                i +=2
            else:
                new_symbols.append(self.symbols[i])
                i+=1
                
        self.symbols = new_symbols
        assert "".join(self.symbols) == self.data
        self.count = self.get_pair_counts(self.symbols)
        if merged_symb not in self.stoi:
            idx = len(self.vocab)
            self.vocab += [merged_symb]
            self.stoi[merged_symb] = idx
            self.itos[idx] = merged_symb
        

    def train(self, final_size):
        
        while len(self.vocab) < final_size:
            if not self.count:
                break
            pair, num = self.count.most_common(1)[0]
            
            if num<2:
                break
            self.merge_pairs(pair)



    def merge_the_pair(self, syms, pair):

        i = 0
        mersym = pair[0]+pair[1]
        out = []
        while i<len(syms):
            if i<len(syms)-1 and syms[i]==pair[0] and syms[i+1]==pair[1]:
                out.append(mersym)
                i += 2
            else:
                out.append(syms[i])
                i += 1
            
        return out

        
    def encode(self, string):

        syms = list(string)
        for pair in self.merges:
            syms = self.merge_the_pair(syms, pair)
            
        out = [self.stoi[s] for s in syms]
        return out

    def decode(self, nums):
        
        out = [self.itos[idx] for idx in nums]
        return ''.join(out)








