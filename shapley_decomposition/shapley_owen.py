import pandas
import numpy
from itertools import combinations
from copy import deepcopy
import warnings
from sklearn.linear_model import LinearRegression
from shapley_decomposition.shared_tools import flatten, frame_maker, weighter, s_compute

def samples(dataframe, force=False):
    """
    Create unique combinations of variables.

    Parameters:
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

        force (bool, optional): Force to calculate for more than 10 variables

    Returns:
        change_pairs_dict (dict) : A dictionary of variable instance change pairs
    """

    main_variables = dataframe.columns.tolist()[:-1]

    if len(main_variables) > 10 and force == False:
        raise ValueError('Number of variables exceeds the limit. In your own discretion you can force more than 20 variables by inputting - force=True - to the function. However, beware computation may take time')
    elif len(main_variables) > 10 and force == True:
        warnings.warn("As the number of variables increase, computation time and cost increase exponentially")
    else:
        pass

    comb_var = [list(combinations(main_variables,size)) for size in range(1,len(main_variables)+1)]
    comb_var2 = flatten(comb_var)
    comb_var3 = [list(n) for n in comb_var2]

    change_pairs_dict = {}

    for variable in main_variables:
        comb_var4 = deepcopy(comb_var3)
        samp = []
        # in order to take combinations with the variable in loop included
        for segment in comb_var4:
            if variable in segment:
                samp.append(segment)
            change_pairs_dict[variable] = samp

    return change_pairs_dict

def shapley_values(dataframe):
    """
    Calculates shapley values for all variables/independent xs.

    Using shapley_owen_samples(), calculates differences between combinations
    with and without choosen independent variables. Weighted differences give
    shapley values for all variables.

    Parameters:
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

    Returns:
        shapley_owen_results (array) : Array with shapley values of variables
    """

    def rsquared(x, y):
        model = LinearRegression()
        model.fit(x, y)
        r_squared = model.score(x, y)
        return r_squared

    sample=samples(dataframe)

    shapley_owen_results=[]
    for variables in sample.keys():
        b_with = [rsquared(dataframe.loc[:,elements], dataframe.iloc[:,-1]) for elements in sample[variables]]
        b_wo = []
        for elements in sample[variables]:
            elements.remove(variables)
            # we remove the variable itself to calc. r2 without it
            if len(elements) == 0:
                b_wo.append(0)
            else:
                b_wo.append(rsquared(dataframe.loc[:,elements], dataframe.iloc[:,-1]))
        diff = [x-t for x,t in zip(b_with,b_wo)]
        shapley_value = diff*weighter(len(sample.keys()), sample[variables], variables, owen=True)
        shapley_owen_results.append(shapley_value)
    return shapley_owen_results

def decomposition (dataframe):
    """
    Creates final output for shapley_owen decomposition of R^2 attribute of the module.

    frame_maker() and shapley_owen_calc() functions interact under shapley_owen() function.

    Parameters:
        dataframe (pandas.core.frame.DataFrame) : Input dataframe

    Returns:
        df_fin (pandas.core.frame.DataFrame) : Final output for shapley_owen
    """

    if type(dataframe) != pandas.core.frame.DataFrame:
        dataframe = frame_maker(dataframe, mode=2)

    results=shapley_values(dataframe)
    df_fin = pandas.DataFrame(index = dataframe.columns.tolist()[:-1], columns = ["contribution"])
    df_fin["contribution"] = [res.sum() for res in results]
    return df_fin
