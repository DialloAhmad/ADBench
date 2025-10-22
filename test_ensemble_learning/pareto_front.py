
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ParetoAnalysis:
    def __init__(self, df, criteria, maximize=True):
        """
        Initializes the Pareto analysis.

        Args:
            df (DataFrame): The DataFrame containing the data.
            criteria (list): List of column names used as criteria.
            maximize (bool or list): Indicates whether to maximize (True) or minimize (False) each criterion.
        """
        self.df = df.copy()
        self.criteria = criteria
        self.maximize = maximize
        self.fronts = []

    def identify_pareto_fronts(self):
        """
        Identifies the successive Pareto fronts.
        """
        costs = self.df[self.criteria].values
        n_points = costs.shape[0]
        unassigned = set(range(n_points))
        fronts = []

        # Convert 'maximize' to a list if necessary
        if isinstance(self.maximize, bool):
            maximize = [self.maximize] * costs.shape[1]
        else:
            maximize = self.maximize

        while unassigned:
            current_front = []
            is_efficient = np.ones(len(unassigned), dtype=bool)
            unassigned_list = list(unassigned)
            for i, idx_i in enumerate(unassigned_list):
                c = costs[idx_i]
                for j, idx_j in enumerate(unassigned_list):
                    if idx_i != idx_j:
                        better_or_equal = []
                        strictly_better = []
                        for k in range(costs.shape[1]):
                            if maximize[k]:
                                better_or_equal.append(costs[idx_j, k] >= c[k])
                                strictly_better.append(costs[idx_j, k] > c[k])
                            else:
                                better_or_equal.append(costs[idx_j, k] <= c[k])
                                strictly_better.append(costs[idx_j, k] < c[k])
                        if all(better_or_equal) and any(strictly_better):
                            is_efficient[i] = False
                            break
            for i, idx in enumerate(unassigned_list):
                if is_efficient[i]:
                    current_front.append(idx)
            fronts.append(current_front)
            unassigned = unassigned - set(current_front)
        self.fronts = fronts

    def plot_pareto_fronts(self, labels, title="Pareto Front", annotate=True, fronts_to_plot=[0, 1]):
        """
        Plots the Pareto front on a 2D graph, connecting only the specified fronts.

        Args:
            labels (list): List of labels for the axes [xlabel, ylabel].
            title (str): The title of the plot.
            annotate (bool): If True, adds the model names on the plot.
            fronts_to_plot (list): List of front indices to include in the Pareto front line.

        """
        plt.figure(figsize=(10,7))
        n_fronts = len(self.fronts)
        
        # Plot all models
        plt.scatter(self.df[self.criteria[0]], self.df[self.criteria[1]], color='blue', label='Models')
        
        # Combine specified fronts
        combined_front_indices = []
        for idx, i in enumerate(fronts_to_plot):
            if i < n_fronts:
                indices = self.fronts[i]
                combined_front_indices.extend(indices)
                front_df = self.df.iloc[indices]
        
            if idx == 0:
                plt.scatter(front_df[self.criteria[0]], front_df[self.criteria[1]],
                            color='red', marker='D', s=100, label='Front')
            else:
                plt.scatter(front_df[self.criteria[0]], front_df[self.criteria[1]],
                            color='red', marker='D', s=100)
            
        # Plot the connecting line for the combined fronts
        if combined_front_indices:
            combined_front_df = self.df.iloc[combined_front_indices]
            combined_front_df_sorted = combined_front_df.sort_values(by=self.criteria[0])
            plt.plot(combined_front_df_sorted[self.criteria[0]], combined_front_df_sorted[self.criteria[1]],
                     color='red', linestyle='--', label='Pareto Front Line')
        
        # Annotate each point with the model name
        if annotate:
            for idx, row in self.df.iterrows():
                plt.annotate(row['Model'], (row[self.criteria[0]], row[self.criteria[1]]),
                             textcoords="offset points", xytext=(5,5), ha='left', fontsize=9)
        
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(title)
        plt.legend()
        plt.grid(True)