import random


# This is a helper function
def custom_groupby(sequence, key):
    """
    A custom implementation of the itertool's "groupby" method that groups consecutive elements in a list that
    have the same value for a specified key.
    @param sequence: The sequence of elements to be grouped.
    @param key: A function that computes a key value for each element in the sequence.
    @return: A list of lists, where each sublist contains consecutive elements from the
    original list that have the same key value.
    """
    if not sequence:
        return []

    grouped = []  # Initialize an empty list to hold groups of elements.
    current_group = [sequence[0]]  # Start the first group with the first element of the sequence.
    current_key = key(sequence[0])  # Apply the key function to the first element to determine its grouping key.

    for element in sequence[1:]:  # Iterate over the rest of the sequence starting from the second element.
        k = key(element)  # Apply the key function to the current element.
        if k == current_key:  # Check if the key of the current element matches the key of the current group.
            current_group.append(element)  # If yes, add the current element to the current group.
        else:   # If the key does not match, it means the current group is complete.
            grouped.append(current_group)  # Add the current group to the list of grouped elements.
            current_group = [element]   # Start a new group with the current element.
            current_key = k  # Update the current key to the key of the new element.

    grouped.append(current_group)  # After the loop, add the last group to the list of grouped elements.
    return grouped   # Return the list of grouped elements.


class DecisionTree:
    """A Descision Tree Classifier."""

    def __init__(self):
        """
        Attributes:
        @param children (dict): A dictionary mapping feature values to child DecisionTree nodes.
        @param split_feature (int): The index of the feature this node splits on.
        @param leaf_label (int/float/str): The label of this node if it is a leaf.
        @param most_common_action (int/float/str): The most common label among the training data reaching this node.
        @return: None.
        """
        self.children = {}
        self.leaf_label = None
        self.split_feature = None
        self.most_common_action = None

    def _gini_impurity(self, examples, feature=None):
        """
        Calculates the Gini impurity for a set of examples, optionally split by a feature.
        @param examples (list): The dataset to calculate impurity for.
        @param feature (int, optional): The feature index to split the data on before calculating impurity.
        @return The Gini impurity, a measure of label variability among the examples.
        """
        if feature is None:   # If no feature is specified, calculate the Gini impurity for the current set of examples.
            # Extract the labels (assumed to be the last element of each example) from the examples.
            labels = [example[-1] for example in examples]
            # Calculate the Gini impurity for the labels. This is done by subtracting the sum of the squared
            # proportions of each label from 1.
            return 1 - sum([(labels.count(label) / len(labels)) ** 2 for label in set(labels)])
        else:  # If a feature is specified, first partition the dataset based on that feature.
            partitions = self._partition(examples, feature)
            # Calculate the weighted Gini impurity for the partitions.
            # This involves computing the Gini impurity for each partition and then weighting it
            # by the proportion of examples in that partition relative to the whole dataset.
            weighted_impurity = sum(
                [self._gini_impurity(partition) * (len(partition) / len(examples)) for partition in partitions])
            return weighted_impurity

    # Determine the most common outcome (class label) among a set of examples
    def _plurality_value(self, examples):
        """
        Finds the most common label (class) among the given examples.
        @param examples (list of lists): The dataset to examine, where each element is a list representing a data point.
        @return The most common label (class) among the examples.
        """
        # Extract the outcomes (assumed to be the last element of each example) from the dataset.
        outcomes = [example[-1] for example in examples]
        # Find and return the outcome that appears most frequently in the dataset.
        # This is done by converting the list of outcomes into a set to remove duplicates,
        # then using the max function with a key that counts the occurrences of each outcome in the original list.
        return max(set(outcomes), key=outcomes.count)

    def _best_feature_to_split(self, examples, available_features):
        """
        Identifies the best feature to split on, based on Gini impurity.
        @param examples (list): The dataset to examine.
        @param available_features (set): A set of indices representing features that can be split on.
        @return The index of the best feature to split on.
        """
        # Calculate the Gini impurity for splitting on each available feature.
        # This involves creating a dictionary where each key is a feature and its value is the
        # Gini impurity of the dataset when it is split based on that feature.
        impurities = {feature: self._gini_impurity(examples, feature) for feature in available_features}
        # Find and return the feature with the minimum Gini impurity.
        # This is achieved by using the `min` function on the impurities dictionary, with the key
        # argument set to retrieve the value (Gini impurity) for each feature. The feature with the
        # lowest Gini impurity is considered the best for splitting the dataset, as it leads to
        # the most homogeneous subsets (or the least impure splits).
        return min(impurities, key=impurities.get)

    # Partition the dataset based on the value of a specific feature
    def _partition(self, examples, feature):
        """
        Splits examples based on the values of a specified feature.
        @param examples (list of lists): The dataset to split, where each element is a list representing a data point.
        @param feature (int): The index of the feature to split on.
        @return list of lists: A list where each element is a group of examples that share the same value for the specified feature.
        """
        # Sort the examples based on the value of the specified feature.
        # This is done by using the `sorted` function with a lambda function as the key.
        # The lambda function takes an example `x` and returns the value of the example at the index `feature`.
        # This ensures that the examples are ordered by the values of the feature in question.
        sorted_by_feature = sorted(examples, key=lambda x: x[feature])
        # Group the sorted examples by the value of the feature.
        # The `custom_groupby` function is called with the sorted examples and a lambda function similar to the
        # one used for sorting. This lambda function again specifies that the grouping should be based on the value of
        # the feature. The `custom_groupby` function will return a list of groups, where each group is a list of
        # examples that have the same value for the specified feature.
        return custom_groupby(sorted_by_feature, key=lambda x: x[feature])

    def _learn(self, examples, available_features):
        """
        Recursively learns the decision tree structure from training data.
        The implementation adheres to the abstract code provided in week 2 lecture slides page 25.
        @param examples (list): The dataset to build the tree from.
        @param available_features (set): Indices of features available for splitting.
        @return: None.
        """
        # Store the most common action in the root during learning
        if not hasattr(self, 'most_common_action') or self.most_common_action is None:
            self.most_common_action = self._plurality_value(examples)

        if not examples:  # If the examples list is empty, stop the learning process for this node.
            return
        # If all examples have the same label, this branch of the tree represents a leaf node with that label.
        if all(example[-1] == examples[0][-1] for example in examples):
            self.leaf_label = examples[0][-1]
            return
        if not available_features:  # If there are no available features left to split on, this branch becomes a leaf.
            self.leaf_label = self._plurality_value(examples)  # The label would be the most common one.
            return

        # Determine the best feature to split on from the remaining available features.
        self.split_feature = self._best_feature_to_split(examples, available_features)
        partitions = self._partition(examples, self.split_feature)  # Partition the dataset based on best split feature.
        # For each partition (subset of examples with the same value for the split feature),
        # create a child decision tree node.
        for partition in partitions:
            # The value of the split feature for this partition identifies the child node.
            child_feature_value = partition[0][self.split_feature]
            child_tree = DecisionTree()
            # Store a reference to the child tree in the current node's children dictionary.
            self.children[child_feature_value] = child_tree
            # Recursively learn from the partition, excluding the feature that was just used for splitting.
            child_tree._learn(partition, available_features - {self.split_feature})

    def prune(self, validation_data):
        """
        Prune the decision tree based on validation data. The method is not finished yet.
        @param validation_data (list): The dataset to validate the decision tree against.
        @return: None.
        """
        # Check if the current node is a leaf node
        if self.leaf_label is not None:
            return

        # Recursively prune child nodes
        for child in self.children.values():
            child.prune(validation_data)

        # Evaluate error before pruning this node
        before_pruning_errors = sum(self.classify(x[:-1]) != x[-1] for x in validation_data)

        # Convert this node to a leaf node by choosing the most common label
        original_children = self.children
        original_split_feature = self.split_feature
        self.children = {}  # Remove children to make this a leaf node
        self.split_feature = None
        self.leaf_label = self._plurality_value(
            [x for x in validation_data if x[original_split_feature] in original_children])

        # Evaluate error after pruning this node
        after_pruning_errors = sum(self.classify(x[:-1]) != x[-1] for x in validation_data)

        # If errors increased after pruning, revert the changes
        if after_pruning_errors > before_pruning_errors:
            self.children = original_children
            self.split_feature = original_split_feature
            self.leaf_label = None

    def classify(self, features):
        """
        Classifies an example by traversing the decision tree.
        @param features (list): The feature values of the example to classify.
        @return: The predicted lael for the example.
        """
        if self.leaf_label is not None: # If the current node is a leaf node, return its label.
            return self.leaf_label
        # Attempt to find the child node corresponding to the value of the split feature in the input features.
        child = self.children.get(features[self.split_feature])
        # If a child node is found, recursively classify using that child node.
        # If no child node matches, return the most common action (label) observed in the training data.
        return child.classify(features) if child else self.most_common_action

    def fit(self, examples):
        """
        Initiates the learning process for the decision tree.
        @param examples (list): The training dataset, where each example includes feature values and a label.
        @return: None.
        """
        self._learn(examples, set(range(len(examples[0]) - 1)))


class RandomForest:
    """A Random Forest Classifier."""

    def __init__(self, number_of_trees=5):
        """
        Attributes:
        @param number_of_trees (int): Number of trees in the forest.
        @param trees (list): List of individual tree classifiers.
        @return: None.
        """
        self.number_of_trees = number_of_trees  # Number of trees in the forest
        self.trees = []  # List of individual tree classifiers

    def _bootstrap_sample(self, data):
        """
        Creates a bootstrap sample for each tree.
        @param data: List of data points.
        @return: A bootstrap sample.
        """
        # This is a random sample with replacement, where the sample size is equal to the size of the input data.
        return [data[i] for i in random.sample(range(len(data)), int(len(data)))]  # data / sample size

    def _fit_single_tree(self, examples, max_features=None):
        """
        Fits a tree to the bootstrap sample.
        @param examples: List of data points.
        @return: A trained decision tree classifier.
        """
        # Assuming examples is a list of lists where the last element is the target
        if max_features is not None:
            # Randomly select feature indices
            features_indices = random.sample(range(len(examples[0]) - 1), max_features)
            # Create a new dataset with only the selected features + target
            reduced_examples = [[example[i] for i in features_indices] + [example[-1]] for example in examples]
        else:
            reduced_examples = examples

        tree = DecisionTree()
        tree.fit(reduced_examples)
        return tree

    def fit(self, data, target):
        """
        Trains the random forest by fitting a tree to each bootstrap sample.
        @param data: List of data points.
        @param target: List of target labels.
        @return: None.
        """
        for _ in range(self.number_of_trees):
            # Create bootstrap sample for each tree.
            examples = self._bootstrap_sample([data[i] + [target[i]] for i in range(len(target))])
            tree = self._fit_single_tree(examples)
            self.trees.append(tree)

    def predict(self, data, legal=None):
        """
        Predict the class label for a data point. The class label is determined by majority vote.
        @param data: List of feature values.
        @param legal: List of legal actions.
        @return: The predicted class label.
        """
        # Aggregate predictions from all trees and take majority vote.
        predictions = [tree.classify(data) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    def reset(self):
        """
        Reset the random forest.
        @return: None.
        """
        self.trees = []


class Classifier:
    """
    Skeleton for a classifier necessary for similar execution
    """

    def __init__(self):
        """
        Attributes:
        @param: None.
        @return: None.
        """
        self.decision_tree = DecisionTree()
        self.random_forest = RandomForest()

    def reset(self):
        """
        Resets the classifier.
        @param: None.
        @return: None.
        """
        self.random_forest.reset()

    def fit(self, data, target):
        """
        Fits the classifier to the data
        @param data: List of data points
        @param target: List of target labels
        """
        # examples = [data[i] + [target[i]] for i in range(len(target))]
        # self.decision_tree.fit(examples)
        self.random_forest.fit(data, target)

    def predict(self, data, legal=None):
        """
        Predicts the class label for a data point
        @param data: List of feature values
        @param legal: List of legal actions
        """
        # self.decision_tree.classify(data)
        return self.random_forest.predict(data, legal)

    def split_data(self, examples, validation_ratio=0.1):
        """
        Splits the data into training and validation sets.
        @param examples (list): The dataset, where each example includes feature values and a label.
        @param validation_ratio (float): The fraction of data to use for validation.
        @return: Two lists, one for training and one for validation.
        """
        # Shuffle the dataset to ensure random distribution
        random.shuffle(examples)

        # Calculate the split index
        split_index = int(len(examples) * (1 - validation_ratio))

        # Split the data
        training_data = examples[:split_index]
        validation_data = examples[split_index:]

        return training_data, validation_data
