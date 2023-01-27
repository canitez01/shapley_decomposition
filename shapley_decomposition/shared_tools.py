import pandas
import numpy
from itertools import combinations
from copy import deepcopy
from math import factorial
import warnings
from sklearn.linear_model import LinearRegression

operators={"+": [numpy.add, 0, "left"],
           "-": [numpy.subtract, 0, "left"],
           "*": [numpy.multiply, 1, "left"],
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

def shunting_yard(input_function):
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
            if element == "*":
                if index != len(func)-1:
                    if func[index+1] == "*":
                        del func[index : index+2]
                        func.insert(index,"**")
                else:
                    raise ValueError("last character of the input function is an operator, function incomplete")
        return [func, only_var]

    function = input_function.replace(" ","")
    #remove any whitespaces if any
    if len(function) != len(input_function):
        warnings.warn("Whitespaces in input function is detected and removed")

    returning = operator_splitter(function,operators)
    splitted_function = returning[0]

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
