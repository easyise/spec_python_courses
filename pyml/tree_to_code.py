# copyright C. Lungu
# origin: http://www.clungu.com/Converting_a_DecisionTree_into_python_code/
#
import numpy as np
from sklearn.tree import _tree
PYTHON_INDENT_STEP = "    "

def pythonize(feature_name):
    """
    Since we will be likely using the columns names of some datasets, and will wish to 
    have some python parmeters for referencing them we need to make sure that these 
    names abide by the python varible nameing convention.

    This function is a really quick and dirty way of achieveing this, in through some quick replace rules.
    """
    return (
        feature_name
            .replace(" ", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("__", "_")
    )

def get_node_feature_names(tree_, feature_names):
    """
    Whenever possible, return the feature names (as in strings)
    """
    try:
        return [
            pythonize(feature_names[i]) if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
    except:
        # when something goes wrong with the above, we will have numbers in the `tree_.feature` list 
        # which we want to convert to actual python variable names (i.e. by converting 5 to "_5")

        # TODO: maybe add this rule to the `pythonize` function and use here instead
        return [f"_{i}" for i in tree_.feature]
    
def stringify_list(_list):
    return f"[{', '.join(str(i) for i in _list)}]"

def probabilities(node_counts):
    """
    By default, the tree stores the number of datapoints from each class in a leaf node (as the node values)
    but we want to convert this into probabilities so the generated code acts like a propper model.

    We can use `softmax` of other squish-list-to-probabilities formulas (in this case `a / sum(A)`)
    """
    return node_counts / np.sum(node_counts)

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_names = list(map(pythonize, feature_names))
    node_feature_name = get_node_feature_names(tree_, feature_names)
    print(f"def tree_model({', '.join(feature_names)}):")

    def __recurse(node, depth):
        indent = PYTHON_INDENT_STEP * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = node_feature_name[node]
            threshold = tree_.threshold[node]
            
            print(f"{indent}if ({name} <= {threshold}):")
            __recurse(tree_.children_left[node], depth + 1)

            print(f"{indent}else:  # if ({name} > {threshold})")
            __recurse(tree_.children_right[node], depth + 1)
        else:
            print(f"{indent}return {stringify_list(probabilities(tree_.value[node][0]))}")

    __recurse(0, 1)