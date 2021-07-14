def filter_df(df, question):
    """Filter dataframe into positive and negative classes.
    Parameters
    ----------
        df: pd.DataFrame
            Features
        question: dict
            Decision tree which is to be splitted
    Returns
    -------
        df_positive, df_negative
    """
    feature, comparison_operator, value = question.split()
    
    # continuous feature
    if comparison_operator == "<=":
        df_positive = df[df[feature] <= float(value)]
        df_negative =  df[df[feature] >  float(value)]
        
    # categorical feature
    else:
        df_positive = df[df[feature].astype(str) == value]
        df_negative  = df[df[feature].astype(str) != value]
    
    return df_positive, df_negative


def determine_leaf(df_train):
    """Determine if the dataframe is the leaf with only one label
    Parameters
    ----------
        df_train: pd.DataFrame
            dataframe to be tested
    """
    return df_train.label.value_counts().index[0]

def determine_errors(clf, df_val, tree):
    """Calculate error in prediction from validation dataset
    Parameters
    ----------
        clf: dict
            Initialized decision Tree
        df_val: pd.DataFrame
            validation set
        tree: dict
            Decision Tree
    Returns
    -------
        sum of total error
    """
    predictions = clf.predictions(df_val, tree)
    actual_values = df_val.label
    
    if len(predictions) == 0:
        return 0
    return sum(predictions != actual_values)
    
def pruning_result(clf, tree, df_train, df_val):
    """Determine if its a leaf or needs to be further splitted
    Parameters
    ----------
        clf: dict
            Initialized decision tree
        tree: dict
            Constructed decision tree
        df_train: pd.DataFrame
            Train set
        df_val: pd.DataFrame
            Validation set
    Returns
    -------
        Leaf if the labels are same else split into tree
    """
    
    leaf = determine_leaf(df_train)
    errors_leaf = determine_errors(clf, df_val, leaf)
    errors_decision_node = determine_errors(clf, df_val, tree)

    if errors_leaf <= errors_decision_node:
        return leaf
    else:
        return tree
    
def post_pruning(clf, tree, df_train, df_val):
    """Prune the tree if error while being leaf node is less than error 
    while splitting a tree
    Parameters
    ----------
        clf: dict
            Initialized decision tree
        tree: dict
            Constructed decision tree
        df_train: pd.DataFrame
            Train set
        df_val: pd.DataFrame
            Validation set
    Returns
    -------
        Pruned tree
    """
    
    question = list(tree.keys())[0]
    positive, negative = tree[question]

    if not isinstance(positive, dict) and not isinstance(negative, dict):
        return pruning_result(clf, tree, df_train, df_val)
        
    else:
        df_train_positive, df_train_negative = filter_df(df_train, question)
        df_val_positive, df_val_negative = filter_df(df_val, question)
        
        if isinstance(positive, dict):
            positive = post_pruning(clf, positive, df_train_positive, df_val_positive)
            
        if isinstance(negative, dict):
            negative = post_pruning(clf, negative, df_train_negative, df_val_negative)
            
        tree = {question: [positive, negative]}
    
        return pruning_result(clf, tree, df_train, df_val)