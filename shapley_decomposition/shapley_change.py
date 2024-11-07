from itertools import product
from math import factorial
import warnings
from shapley_decomposition.shared_tools import shunting_yard, rpn_calc, frame_maker, cagr_calc, s_sequence

def samples(dataframe):
    """
    Create cartesian products of n ordered pairs of the variable-instance
    couples (without omitted-variables). Ordered pairs are the values of
    [xn-t1, xn-t2], i.e. first and second instances of a variable. As cartesian
    product is equivalent to nested for-loop of n degrees, it creates a
    consistent order which is utilized with weight_computes and s_sequence
    function. For each element of cartesian product, add the first and second
    instances of the omitted variable to its original index in two seperate
    lists.

    Create ordered weights list for an element of the cartesian product of a
    variable (omitted) which applies same for the rest.

    Parameters:
    ----------
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

    Returns:
    ----------
        [change_pairs_dict, weight_computes] (list) :

            change_pairs_dict (dictionary): keys for variables, values for nested
            list of each cartesian product

            .. versionchanged:: 0.0.2

            weight_computes (list): List of computed weights of samples

            .. versionadded:: 0.0.2

    Notes:
    ----------
        .. versionchanged:: 0.0.2
    """
    dataframe = frame_maker(dataframe)
    dep_y = dataframe.index[0]

    instance0 = dataframe.columns.tolist()[0]
    instance1 = dataframe.columns.tolist()[1]
    change_pairs_dict = {}
    weight_computes = [(factorial(s_count)*factorial(len(dataframe[1:])-s_count-1))/factorial(len(dataframe[1:])) for s_count in s_sequence(len(dataframe[1:])-1)]

    for pos, ind_x in enumerate(dataframe.index.tolist()[1:]):
        pruned_dataframe = dataframe[(dataframe.index != dep_y)&(dataframe.index != ind_x)]
        second_instance_list = pruned_dataframe.iloc[:,1].tolist()
        ommitted_ins0 = dataframe.loc[ind_x, instance0]
        ommitted_ins1 = dataframe.loc[ind_x, instance1]

        comb_list = product(*pruned_dataframe.values)

        name = "x"+str(pos+1)
        segments = []
        for segm in comb_list:
            base_segm1=list(segm)
            base_segm2=list(segm)
            base_segm1.insert(pos, ommitted_ins0)
            base_segm2.insert(pos, ommitted_ins1)
            segments.append([base_segm2,base_segm1])

        change_pairs_dict[name] = segments
    return [change_pairs_dict, weight_computes]

def shapley_values(dataframe, function, progress_report=False):
    """
    Calculates shapley values for all variables/independent xs

    segm_difference() function combines the input function and values.

    Parameters:
    ----------
        dataframe (pandas.core.frame.DataFrame) : Input dataframe
        function (str) : Input function in text format (right hand side of equation)
        progress_report (bool, optional) : If the number of variables are more
        than or equal to 20 provide progress report, otherwise (default) false.

    Returns:
    ----------
        calculated_shapley_for_samples (array) : Array with shapley value arrays
        of variables

    Notes:
    ----------
        .. versionchanged:: 0.0.2
    """

    dataframe = frame_maker(dataframe)
    samples_and_weight = samples(dataframe)
    sample_return = samples_and_weight[0]
    weight_return = samples_and_weight[1]
    calculated_shapley_for_samples = []

    shunting_res = shunting_yard(function)
    function_transformed = shunting_res[0]
    var_transformed = shunting_res[1]
    real_variable_pos = shunting_res[2]

    iterable_length=2**(var_transformed-1)

    def segm_difference(dataframe, sample_segment, function_transformed, var_transformed, real_variable_pos):

        if var_transformed == len(sample_return):
            raw_func = function_transformed
            var_checker = 0
            for pos,variable in enumerate(real_variable_pos):
                raw_func[variable] = sample_segment[pos]
                var_checker += 1

            if var_checker != var_transformed:
                raise ValueError('Input and generated variables are not matched. Make sure the variable names in input function is "x+some integer".')
            else:
                return rpn_calc(raw_func)
        else:
            raise ValueError('Number of variables in function and data are not equal. Check both the input function and data.')

    if progress_report == True or iterable_length >= 1048576:
        for variables in sample_return.keys():
            iterable_process=0
            raw_shapley = []
            for combs,weights in zip(sample_return[variables],weight_return):
                raw_shapley.append((segm_difference(dataframe, combs[0], function_transformed, var_transformed,real_variable_pos)-
                                    segm_difference(dataframe, combs[1], function_transformed, var_transformed,real_variable_pos))*weights)
                iterable_process += 1
                if iterable_process % (iterable_length/8) == 0:
                    print("\r", "processing " + variables+ ": " +str(round(iterable_process*100/iterable_length,1)) + '% completed', end="     ")

            calculated_shapley_for_samples.append(raw_shapley)
        return calculated_shapley_for_samples
    else:
        for variables in sample_return.keys():
            raw_shapley = []
            for combs,weights in zip(sample_return[variables],weight_return):
                raw_shapley.append((segm_difference(dataframe, combs[0], function_transformed, var_transformed,real_variable_pos)-
                                    segm_difference(dataframe, combs[1], function_transformed, var_transformed,real_variable_pos))*weights)

            calculated_shapley_for_samples.append(raw_shapley)

        return calculated_shapley_for_samples

def decomposition(dataframe, function, cagr = False, print_progress = False):
    """
    Creates final output for shapley_change decomposition.

    frame_maker(), shapley_cahnge_calc() and cagr_calc() functions interact under shapley_change() function.

    Parameters:
    ----------
        dataframe (pandas.core.frame.DataFrame) : Inital dataframe
        function (str) : Input function in text format (right hand side of equation)
        cagr (bool, optional) : Calculate cagr results, default false.

    Returns:
    ----------
        df_fin (pandas.core.frame.DataFrame) : Final output for shapley_change

    Notes:
    ----------
        .. versionchanged:: 0.0.2
    """

    dataframe = frame_maker(dataframe)
    dep_y = dataframe.index[0]
    warnings.warn("Check the dataframe as the dependent variable(y) should be the first in position i.e at index 0")

    df_fin = dataframe.copy()
    results = shapley_values(dataframe, function, progress_report = print_progress)

    df_fin["dif"] = dataframe.iloc[:,1] - dataframe.iloc[:,0]
    df_fin["shapley"] = [df_fin.iloc[0,2]]+[sum(result) for result in results]
    df_fin["contribution"] = [m/df_fin.loc[dep_y,"shapley"] for m in df_fin["shapley"].tolist()]

    if 0.9999 < df_fin["contribution"].sum()-1 < 1.0001:
        pass
    else:
        raise ValueError('Contribution of variables either exceeds or fail to reach 1.0 within +-0.0001 precision. Check both the input function and data.')

    if cagr == True:
        df_fin["yearly_growth"]=[cagr_calc(dataframe.loc[dep_y, dataframe.columns[0]], dataframe.loc[dep_y,dataframe.columns[1]], (float(dataframe.columns[1])-float(dataframe.columns[0])))*n
        for n in df_fin.contribution.tolist()]
    return df_fin
