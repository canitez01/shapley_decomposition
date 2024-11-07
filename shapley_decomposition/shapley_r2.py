import pandas
import numpy
import warnings
from shapley_decomposition.shared_tools import flatten, frame_maker, powerset, weighter_r2, rsquared

def samples(dataframe):
    """
    Create powerset of variable sets with omitted variables.

    Parameters:
    ----------
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

        force (bool, optional): Force to calculate for more than 10 variables

    Returns:
    ----------
        (dict) : A dictionary of variable and powerset without it
    """

    if type(dataframe) != pandas.core.frame.DataFrame:
        dataframe = frame_maker(dataframe, mode=2)

    main_variables = dataframe.columns.tolist()[:-1]

    change_pairs_dict = {}
    for var in main_variables:
        main_set_wo=[m for m in main_variables if m != var]
        comb_var=[list(element) for element in list(powerset(main_set_wo))]
        change_pairs_dict[var]=comb_var
    return change_pairs_dict

def shapley_decomposition(dataframe, force=False):
    """
    Calculates shapley values and decomposes model r2 to individual contributions.

    Using shapley_owen_samples(), calculates differences between combinations
    with and without variables. Weighted differences give shapley values for all
    variables.

    Parameters:
    ----------
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

    Returns:
    ----------
        df_fin (pandas.core.frame.DataFrame) : Decomposition according to
        shapley values
    """

    if type(dataframe) != pandas.core.frame.DataFrame:
        dataframe = frame_maker(dataframe, mode=2)

    main_variables = dataframe.columns.tolist()[:-1]

    if len(main_variables) > 10 and force == False:
        raise ValueError('Number of variables exceeds the limit. In your own discretion you can force more than 10 variables by inputting - force=True - to the function. However, beware computation may take time')
    elif len(main_variables) > 10 and force == True:
        warnings.warn("As the number of variables increase, computation time and cost increase exponentially")
    else:
        pass

    sample=samples(dataframe)
    shapley_results=[]
    for variables in sample.keys():
        b_with = [rsquared(dataframe.loc[:,elements+[variables]], dataframe.iloc[:,-1]) for elements in sample[variables]]
        b_wo = [rsquared(dataframe.loc[:,elements], dataframe.iloc[:,-1]) for elements in sample[variables]]
        diff = [x-t for x,t in zip(b_with,b_wo)]
        weights=[weighter_r2(len(m),len(sample)) for m in sample[variables]]
        shapley_value = numpy.array(diff)*numpy.array(weights)
        shapley_results.append(shapley_value)

    df_fin = pandas.DataFrame(index = dataframe.columns.tolist()[:-1], columns = ["shapley_values","contribution"])
    df_fin["shapley_values"] = [res.sum() for res in shapley_results]
    df_fin["contribution"] = df_fin["shapley_values"].values/rsquared(dataframe.iloc[:,:-1],dataframe.iloc[:,-1])
    return df_fin

def owen_decomposition(dataframe, partitions, force=False):
    """
    Calculates owen values and decomposes model r2 to individual and
    coalitional/group contributions.

    Using input paritition structure, calculates individual owen values of
    variables and aggragated contribution of coalitions/groups.

    Parameters:
    ----------
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

        partitions (list): List of paritions with variable names(in str format)
        showing group structure

    Returns:
    ----------
        (list) : Final output for owen values and contribution of individuals [0]
        and owen values and contribution of groups [1]
    """

    if type(dataframe) != pandas.core.frame.DataFrame:
        dataframe = frame_maker(dataframe, mode=2)

    if len(flatten(partitions)) != len(dataframe.columns.tolist()[:-1]):
        raise ValueError('Number of individual variables in input partition does not match the dataframe. Make sure the partition and data matches.')

    all_partitions = {}
    for i,partition in enumerate(partitions):
        all_partitions["b"+str(i+1)] = partition
        for subs in partition:
            if isinstance(subs, list):
                raise TypeError('One of your partitions list has a sublist. Provide variables like [["x1","x2"],["x3","x4","x5"],...,["xn"]]')

    m = list(all_partitions.keys())

    if len(m) > 10 and force == False:
        raise ValueError('Number of groups exceeds the limit. In your own discretion you can force more than 10 variables by inputting - force=True - to the function. However, beware computation may take time')
    elif len(m) > 10 and force == True:
        warnings.warn("As the number of groups increase, computation time and cost increase exponentially")
    else:
        pass

    pset_m = [list(element) for element in list(powerset(m))]

    contributions={}
    for coalition in m:
        pset_wo_coalition = [m for m in pset_m if coalition not in m]
        for individuals in all_partitions[coalition]:
            mainset_wo_indi = [m for m in all_partitions[coalition] if m != individuals]
            pset_wo_indi = [list(element) for element in list(powerset(mainset_wo_indi))]
            pset_w_indi = [m + [individuals] for m in pset_wo_indi]

            simple_union = []
            union_w_indi = []

            first_weights = []
            second_weights = []
            for coal in pset_wo_coalition:
                for wo_indi, w_indi in zip(pset_wo_indi, pset_w_indi):
                    if len(coal) == 0:
                        simple_union.append([wo_indi])
                        union_w_indi.append([w_indi])
                    else:
                        simple_union.append([wo_indi, [all_partitions[n] for n in coal]])
                        union_w_indi.append([w_indi, [all_partitions[n] for n in coal]])
                    first_weights.append(weighter_r2(len(coal),len(m)))
                    second_weights.append(weighter_r2(len(wo_indi), len(all_partitions[coalition])))

            diff = []
            for x,y in zip(union_w_indi, simple_union):
                diff.append(rsquared(dataframe[flatten(x)].values, dataframe.iloc[:,-1]) - rsquared(dataframe[flatten(y)].values, dataframe.iloc[:,-1]))
            result=numpy.array(diff)*numpy.array(first_weights)*numpy.array(second_weights)
            contributions[individuals] = sum(result)
    df_results=pandas.DataFrame(index = list(contributions.keys()), columns = ["owen_values","contribution","group_owen"])
    df_results["owen_values"] = list(contributions.values())
    df_results["contribution"] = numpy.array(list(contributions.values()))/rsquared(dataframe.iloc[:,:-1],dataframe.iloc[:,-1])
    group_owen = []
    for n in df_results.index.tolist():
        for v in all_partitions.keys():
            if n in all_partitions[v]:
                group_owen.append(v)

    df_results["group_owen"]=group_owen

    df_results.groupby(["group_owen"]).sum()
    return [df_results, df_results.groupby(["group_owen"]).sum()]
