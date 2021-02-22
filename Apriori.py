import numpy as np
import pandas as pd
from itertools import combinations
from itertools import permutations

class AprioriCluster:
    
    def find_support(self, data, cluster_coverage=0.5,  max_length = 5):
        support = {}

        clusters=list(filter(lambda x: 'cluster' in x, data.columns))
        for cluster in clusters:
            cluster_supp=data.loc[:,cluster].sum()/len(data.index)
            support[cluster]=cluster_supp
            attributes = list(filter(lambda x: 'cluster' not in x, data.columns))
            for number_of_attributes in range(1, max_length):
             
                comb=[list(elem)+[cluster] for elem in (combinations(attributes,number_of_attributes))]

                attributes =[]         
                for j in (comb):
                   
                    sup = data.loc[:,j].product(axis=1).sum()/len(data.index)
                    if sup > (cluster_supp*cluster_coverage):
                        support[tuple(j)] = sup
                        j.pop()
                        support[tuple(j)]=data.loc[:,j].product(axis=1).sum()/len(data.index)
                        attributes = list(set(attributes) | set(j))
                
     
        support_result = pd.DataFrame(list(support.items()), columns = ["Items", "Support"])
        
        return(support_result)



    def find_rules(self,support_dataframe, min_threshold=0.5):

        support = pd.Series(support_dataframe.Support.values, index=support_dataframe.Items).to_dict()
        data = []
        items= support_dataframe.Items.values
        for item in items:
            if(type(item)!=str and ('cluster' in item[-1])):
                a=item[:-1]
                b=item[-1]
                supp_a=support[a]
                supp_b=support[b]
                conf=support[item]/supp_a
                if(conf>min_threshold):
                    lift=conf/support[b]
                    data.append([a, b, supp_a, supp_b, support[item], conf, lift])

        result = pd.DataFrame(data, columns = ["ANT", "CONSEQ", "ANT support", "CONSEQ support","support", "confidence", "lift"])
        return(result)
