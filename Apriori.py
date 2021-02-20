import numpy as np
import pandas as pd
from itertools import combinations
from itertools import permutations

class Apriori:
    
    def find_support(self, data, min_support=0.04,  max_length = 5):
        support = {} 
        items = list(data.columns)

        for number_of_attributes in range(1, max_length+1):
            comb = list(set(combinations(items,number_of_attributes)))
        
            items =set()         
            for j in (comb):
                sup = data.loc[:,j].product(axis=1).sum()/len(data.index)
                if sup > min_support:
                    support[j] = sup
                    items = list(set(items) | set(j))

        result = pd.DataFrame(list(support.items()), columns = ["Items", "Support"])
        
        return(result)



    def find_rules(self,dataframe, min_threshold=0.5):
        support = pd.Series(dataframe.Support.values, index=dataframe.Items).to_dict()
        data = []
        items= dataframe.Items.values
        perm = list(permutations(items, 2))

        for i in perm:
            if set(i[0]).issubset(i[1]):
                conf = support[i[1]]/support[i[0]]
                if conf > min_threshold:
                    j = i[1][not i[1].index(i[0][0])]
                    lift = support[i[1]]/(support[i[0]]* support[(j,)])
                    data.append([i[0], (j,), support[i[0]], support[(j,)], support[i[1]], conf, lift])

        result = pd.DataFrame(data, columns = ["ANT", "CONSEQ", "ANT support", "CONSEQ support","support", "confidence", "lift"])
        return(result)
