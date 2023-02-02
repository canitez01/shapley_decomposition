import pandas
import numpy
from itertools import combinations
from copy import deepcopy
import warnings
from shapley_decomposition.shared_tools import weighter, flatten, shunting_yard, RPN_calc, frame_maker, s_compute, cagr_calc

def samples(dataframe):
    """
    Create unique combinations of variables and time/instances.

    main_var lists all x-t combinations. var_group lists
    the list of x-t combinations seperately ([x1t1,x1t2],[...]). shapley_list's
    loop selects combinations only with xs in their original position. Finally,
    segments help create t2-t1 pairs for chosen independent x + other unique x
    combinations

    Parameters:
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

    Returns:
        change_pairs_dict (dict) : A dictionary of variable instance change pairs
    """

    dataframe = frame_maker(dataframe)
    dep_y = dataframe.index[0]

    change_pairs_dict={}
    for index,ind_x in enumerate(dataframe.index.tolist()[1:]):
        pruned_dataframe = dataframe[(dataframe.index != dep_y)&(dataframe.index != ind_x)]

        main_var = []
        var_group = []
        for xs in pruned_dataframe.index.tolist():
            var_with_instance = []
            for instance in pruned_dataframe.columns.tolist():
                var_with_instance.append(xs+"-"+str(instance))
                main_var.append(xs+"-"+str(instance))
            var_group.append(var_with_instance)
        comb_list = list(combinations(main_var,len(pruned_dataframe.index)))

        shapley_list = []
        for combos in comb_list:
            count = 0
            for i, varis in enumerate(combos):
                if varis in var_group[i]:
                    count += 1
            if count == len(pruned_dataframe.index):
                shapley_list.append(combos)

        shapley_list = [list(i) for i in shapley_list]
        name = "x"+str(index+1)
        segments = []
        for segm in shapley_list:
            segm1 = [ind_x+"-"+pruned_dataframe.columns.tolist()[0]] + segm
            segm2 = [ind_x+"-"+pruned_dataframe.columns.tolist()[1]] + segm
            segments.append([segm2,segm1])
        change_pairs_dict[name] = segments

    return change_pairs_dict

def shapley_values (dataframe, function):
    """
    Calculates shapley values for all variables/independent xs

    segm_difference() function converts input variables into a dictionary to attain
    correct values to the input function. finish_func is the shunting_yard assessed
    raw_func with values placed according to variable dictionary (a map between
    dataframe and input function)

    Parameters:
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

    Returns:
        calculated_shapley_for_samples (array) : Array with shapley value arrays
        of variables
    """
    dataframe = frame_maker(dataframe)
    sample = samples(dataframe)
    calculated_shapley_for_samples = []
    weights = []

    def segm_difference(dataframe, sample_segment, function):
        variable_dict = {}
        # a variable dictionary to attain positions for xs and input variables
        for i, name in enumerate(dataframe.index.tolist()):
            if i != 0:
                #disregard y which should be the first input variable in input dataframe
                variable_dict[name] = ["x"+str(i)]

        for variable_instance_couple in sample_segment:
            variable_dict[variable_instance_couple.split("-")[0]].append(dataframe.loc[variable_instance_couple.split("-")[0],variable_instance_couple.split("-")[1]])

        inv_map = {value[0]: [key,value[1]] for key, value in variable_dict.items()}

        if shunting_yard(function)[1] == len(inv_map.keys()):
            raw_func = shunting_yard(function)[0]
            finish_func = []
            var_checker = 0
            #put the values of variables in function according to the inverse of variable dictionary we created
            for variable in raw_func:
                if variable in inv_map.keys():
                    var_checker += 1
                    finish_func.append(inv_map[variable][1])
                else:
                    #constants and operators which are not in variable_dict
                    finish_func.append(variable)
            if var_checker != shunting_yard(function)[1]:
                raise ValueError('Input and generated variables are not matched. Make sure the variable names in input function is "x + some integer".')
            else:
                return RPN_calc(finish_func)
        else:
            raise ValueError('Number of variables in function and data are not equal. Check both the input function and data.')

    for variables in sample.keys():
        raw_shapley = []
        samples_to_weight=[pairs[0][1:] for pairs in sample[variables]]
        # first of pairs for every pair [0] without the varible we calculate the contr. for [0][1:]
        weights=weighter(len(dataframe.index.tolist()[1:]), samples_to_weight ,dataframe.columns.tolist()[1], owen=False)

        for combs in sample[variables]:
            raw_shapley.append(segm_difference(dataframe, combs[0], function)-segm_difference(dataframe, combs[1], function))

        calculated_shapley_for_samples.append(raw_shapley*weights)

    return calculated_shapley_for_samples

def decomposition(dataframe, function, cagr=False):
    """
    Creates final output for shapley_change decomposition.

    frame_maker(), shapley_cahnge_calc() and cagr_calc() functions interact under shapley_change() function.

    Parameters:
        dataframe (pandas.core.frame.DataFrame) : Inital dataframe
        function (str) : Input function in text format (right hand side of equation)
        cagr (bool, optional) : Calculate cagr results, default false.

    Returns:
        df_fin (pandas.core.frame.DataFrame) : Final output for shapley_change
    """

    dataframe = frame_maker(dataframe)
    dep_y = dataframe.index[0]
    warnings.warn("Check the dataframe as the dependent variable(y) should be the first in position i.e at index 0")

    df_fin=dataframe.copy()
    results = shapley_values(dataframe, function)

    df_fin["dif"] = [x-y for x,y in zip(dataframe.loc[:,dataframe.columns[1]].tolist(),dataframe.loc[:,dataframe.columns[0]].tolist())]
    df_fin["shapley"] = [df_fin.iloc[0,2]]+[result.sum() for result in results]
    df_fin["contribution"] = [m/df_fin.loc[dep_y,"shapley"] for m in df_fin["shapley"].tolist()]

    if 0.9999 < df_fin["contribution"].sum()-1 < 1.0001:
        pass
    else:
        raise ValueError('Contribution of variables either exceeds or fail to reach 1.0 within +-0.0001 precision. Check both the input function and data.')

    if cagr == True:
        df_fin["yearly_growth"]=[cagr_calc(dataframe.loc[dep_y, dataframe.columns[0]], dataframe.loc[dep_y,dataframe.columns[1]], (float(dataframe.columns[1])-float(dataframe.columns[0])))*n
        for n in df_fin.contribution.tolist()]
    return df_fin
