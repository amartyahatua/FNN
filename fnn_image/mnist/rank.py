import torch
import random
import pandas as pd

"""
This class helps to find the order of the neurons based on the activation.
Four different types of orders are generated:

1. Node order: It returns the position of the node in the neueal network. Suppose, one layer of neural network consists of five nodes. The first node in the
n/w assigned as 1 and so on.
2. Random order: A random order is assigned with every node. 
3. Node rank: Ranking nodes based on the activation
4. Top K: Ranking nodes based on the activation and returning top K ranks and set zero to rest of the nodes
"""
class Rank:

    def get_index(self, df_avg):
        """
        It sorts the value of the dataframe and returns the sorted index

        param: df_avg Dataframe having activation value of each neuron
        return: list of sorted index
        """
        L = []
        for val in df_avg.values.tolist():
            L.extend(val)
        x = tuple(k[1] for k in sorted((x[1], j) for j, x in enumerate(
            sorted((x, i) for i, x in enumerate(L)))))
        ord_index = [max(x) - i for i in list(x)]
        return ord_index

    def node_order(self, weights):
        """
        Get sorted index based on the values of the weights

        param: weights Dataframe having activation value of each neuron
        return: list of sorted index
        """
        average = torch.mean(weights, axis=0)
        new_average = pd.DataFrame(average.detach().numpy())
        ord_index = self.get_index(new_average)
        return ord_index

    def random(self, weights):
        """
        Get random index based on the values of the weights

        param: weights Dataframe having activation value of each neuron
        return: list of sorted index
        """

        random_numbers = random.sample(range(0, weights.shape[1] - 1), random.randint(0, weights.shape[1] - 1))
        ord_index = self.node_order(weights)
        for rn in random_numbers:
            try:
                ord_index[rn] = 1
            except:
                continue
        ord_index = torch.tensor(ord_index)
        return ord_index

    def top_K(self, weights, K):
        """
        Get top K index based on the values of the weights

        param: weights Dataframe having activation value of each neuron
        return: list of sorted index
        """
        ord_index = self.node_order(weights)
        ord_index = [ind if K <= ind else 0 for ind in ord_index]
        ord_index = torch.tensor(ord_index)
        return ord_index

    def get_ranks(self, weights):
        ord_index = self.node_order(weights)
        ord_index = torch.tensor(ord_index)
        return ord_index
