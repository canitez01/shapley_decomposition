import pandas
import numpy
from itertools import combinations
from copy import deepcopy
from math import factorial
import warnings
from sklearn.linear_model import LinearRegression

operators={"+": [numpy.add, 0, "left"] ,
           "-": [numpy.subtract, 0, "left"] ,
           "*": [numpy.multiply, 1, "left"] ,
           "/": [numpy.divide, 1, "left"],
           "÷": [numpy.divide, 1, "left"],
           "**": [numpy.power,2, "right"],
           "^": [numpy.power,2, "right"],
           "(": ["para_open",3],
           ")": ["para_close",3]}

def flatten (list_of_list):
    """
    Flattens the list of lists (of nth level) without breaking the integrity of the non-list elements.

    Parameters:
        list_of_list (list): A list with any levels of sub-lists

    Returns:
        flatten_list (list): Flattened list
    """

    list_of_list_copy=deepcopy(list_of_list)
    flatten_list=[]
    while len(list_of_list_copy) > 0:
        for element in list_of_list_copy:
            if type(element)==list:
                del list_of_list_copy[0]
                element.reverse()
                for n in element:
                    list_of_list_copy.insert(0,n)
                break
            else:
                flatten_list.append(element)
                del list_of_list_copy[0]
                break
    return flatten_list

def shunting_yard(function):
    """
    Splits input text by arithmetic operators, then parses to post-fix notation.

    operator_splitter function splits text by predefined operators. Output
    list of the split operation is parsed to post-fix notation to relieve the
    parantheses dependencies. Dijkstra's Shunting Yard Algorithm is used for
    the parsing.

    Parameters:
        function (str) : Input function in text format (right hand side of equation)

    Returns:
        [flatten(main_stack),var_count] (list) : A list containing, function in reverse polish notation (list) and count of variables (int)
    """

    def operator_splitter(func,op_list):
        only_var = deepcopy(func)
        for operator in op_list:
            if type(func) == str:
                only_var = only_var.split(operator)
                func = func.split(operator)
                if len(func) > 1:
                    for index in range(1,len(func)):
                        #operators are inserted between what they splitted, thus odd indices
                        func.insert(index*2-1,operator)
            else:
                func = [element.split(operator) for element in flatten(func)]
                only_var = [element.split(operator) for element in flatten(only_var)]
                for parts in func:
                    if len(parts) > 1:
                        for index in range(1, len(parts)):
                            #operators are inserted between what they splitted, thus odd indices
                            parts.insert(index*2-1,operator)

        #clean the left-over empty lists from split operations
        func = [element for element in flatten(func) if len(element) != 0]
        only_var = [element for element in flatten(only_var) if len(element) != 0]

        #in order not to confuse power-** with two multiplications-*
        for index,element in enumerate(func):
            if element == "*" and func[index+1] == "*":
                del func[index : index+2]
                func.insert(index,"**")
        return [func, only_var]

    returning=operator_splitter(function,operators)
    splitted_function=returning[0]

    var_count = 0
    for variable in returning[1]:
        #Don't count if it is a constant
        try:
            float(variable)
        except:
            var_count += 1

    par_stack = 0
    for ops in returning[0]:
        if ops == "(" or ops == ")":
            par_stack += 1

    if par_stack %2:
        raise ValueError("Uneven/Wrong number of paranthesis, check the input function")

    if splitted_function[-1] in operators.keys() and splitted_function[-1] != ")":
        raise ValueError("last character of the input function is an operator, function incomplete")

    main_stack = []
    operator_stack = []
    for char in splitted_function:
        if char in operators.keys():
            if len(operator_stack) == 0:
                #if operator stack is empty, append it
                operator_stack.append(char)
            else:
                # 4 conditions to check due to paranthesis operations
                if "(" not in operator_stack and char != "(":
                    if operators[char][1] > operators[operator_stack[0]][1]:
                        # if the new operator is of higher degree than the ones in the stack, we add it into the stack first position
                        operator_stack.insert(0,char)
                    else:
                        # if the new operator is of lower degree or equal to the ones in the stack, we remove those from the operator stack and add to the main_stack
                        if char == "**" or char == "^":
                            # power operator is right associative thus we add
                            operator_stack.insert(0,char)
                        else:
                            to_rem = [c for c in operator_stack if operators[char][1] <= operators[c][1]]
                            operator_stack = [i for i in operator_stack if i not in to_rem]
                            main_stack.append(to_rem)
                            operator_stack.insert(0,char)

                elif "(" not in operator_stack and char == "(":
                    operator_stack.insert(0,char)

                elif "(" in operator_stack and char != "(":
                    # assess shunting yard within the boundaries of open paranthesis, i.e. after par_holder
                    par_holder = operator_stack.index("(")
                    if char == ")":
                        operator_stack.insert(0,char)
                        to_rem = [[m for m,n in enumerate(operator_stack) if ")" in n][0],[m for m,n in enumerate(operator_stack) if "(" in n][0]]
                        main_stack.append(operator_stack[to_rem[0]+1:to_rem[1]])
                        operator_stack = operator_stack[:to_rem[0]]+operator_stack[to_rem[1]+1:]

                    elif len(operator_stack) == 1:
                        operator_stack.insert(0,char)
                    else:
                        if operators[char][1] > operators[operator_stack[0]][1]:
                            operator_stack.insert(0,char)
                        else:
                            if char == "**" or char == "^":
                                operator_stack.insert(0,char)
                            else:
                                to_rem = [[m,n] for m,n in enumerate(operator_stack[:par_holder]) if operators[char][1] <= operators[n][1]]
                                operator_stack = operator_stack[len(to_rem):]
                                main_stack.append([x[1] for x in to_rem])
                                operator_stack.insert(0,char)
                else:
                    operator_stack.insert(0,char)
        else:
            try:
                if type(float(char)) == float:
                    main_stack.append(float(char))
            except:
                main_stack.append(char)
    for remaining_operators in operator_stack:
        main_stack.append(remaining_operators)

    return [flatten(main_stack),var_count]

def RPN_calc(finish_func):
    """
    Reverse Polish Notation evaluater

    Parameters:
        finish_func (list): The function with real parameters

    Returns:
        stack (list): Final numerical value after operations

    Code credit:
    Josh Haberman
    https://blog.reverberate.org/2013/07/ll-and-lr-parsing-demystified.html
    """

    stack = []
    for c in finish_func:
        if c in operators.keys():
            arg2 = stack.pop()
            arg1 = stack.pop()
            result = operators[c][0](arg1, arg2)
            stack.append(result)
        else:
            stack.append(c)
    return stack.pop()

def s_compute(sample, t1, owen=False):
    """
    Counts number of variables other than choosen variable for shapley_owen and other than second instance, t1, for shapley_change.

    Parameters:
        sample (list) : List of pruned shapley_sample
        t1 (str) : name of the second instance or time
        owen (kwarg) : compute s for shapley_owen or shapley_change, default false.

    Returns:
        s_counts (int) : number of s
    """

    s_counts = []
    for variable_set in sample:
        counter = 0
        if owen == True: #for owen_shapley decomposition, number of variables other than
            counter = len(variable_set)
        else:
            for variable_instance_couple in variable_set:
                variable_instance_list = variable_instance_couple.split("-")
                if t1 in variable_instance_list[1]:
                    counter +=1
        s_counts.append(counter)
    return s_counts

def weighter(m,sample,t1, owen=False):
    """
    Calculates weights for combinations/segments.

    Parameters:
        m (int) : Total number of variables
        sample (list) : List of pruned shapley_sample
        t1 (str) : name of the second instance or time
        owen (kwarg) : compute s for shapley_owen or shapley_change, default false.

    Returns:
        weights (array) : array of weights for segments
    """

    weights = []
    if owen == True: # for owen_shapley decomposition
        computed_s = s_compute(sample,t1, owen = True)
    else:
        computed_s = s_compute(sample,t1)
    for ss in computed_s:
        weight = (factorial(ss)*factorial(m-ss-1))/factorial(m)
        weights.append(weight)
    return numpy.array(weights)

def shapley_change_samples(dataframe):
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

    dep_y = dataframe.index.tolist()[0]
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

def shapley_change_calc (dataframe, function):
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

    sample = shapley_change_samples(dataframe)
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
            #put the values of variables in function according to the inverse of variable dictionary we created
            for variable in raw_func:
                if variable in inv_map.keys():
                    finish_func.append(inv_map[variable][1])
                else:
                    #constants and operators which are not in variable_dict
                    finish_func.append(variable)
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

def shapley_owen_samples(dataframe, force=False):
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

def shapley_owen_calc(dataframe):
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

    sample=shapley_owen_samples(dataframe)

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

def cagr_calc(start,end,dur):
    """ Calculates compound annual growth rate with start-end dates and duration between. """

    return ((((int(end)/int(start))**(1/dur)))-1)*100

def frame_maker(array, mode=1):
    """ Converts arrays or list of lists as inputs to pandas dataframe. Seperate modes for shapley_change(1) and shapley_owen(2)"""

    if mode == 1:
        data=pandas.DataFrame(array,index=["x"+str(ext) if ext !=0 else "y" for ext in range(0,len(array))])
        data.columns=[str(m) for m in data.columns.tolist()]
        return data
    elif mode == 2:
        data=pandas.DataFrame(array,columns=["x"+str(ext) if ext != len(array[0]) else "y" for ext in range(1,len(array[0])+1)])
        return data

def shapley_owen (dataframe):
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

    results=shapley_owen_calc(dataframe)
    df_fin = pandas.DataFrame(index = dataframe.columns.tolist()[:-1], columns = ["contribution"])
    df_fin["contribution"] = [res.sum() for res in results]
    return df_fin


def shapley_change(dataframe, function, cagr=False):
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
    if type(dataframe) != pandas.core.frame.DataFrame:
        dataframe = frame_maker(dataframe)

    dep_y = dataframe.index[0]
    warnings.warn("Check the dataframe as the dependent variable(y) should be the first in position i.e at index 0")

    t_cols = [str(col) for col in dataframe.columns.tolist()]
    dataframe.columns = t_cols

    df_fin=dataframe.copy()

    results = shapley_change_calc(dataframe, function)

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
