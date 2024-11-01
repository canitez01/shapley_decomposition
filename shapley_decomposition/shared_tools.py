import pandas
from itertools import chain, combinations
from math import factorial
import operator
import warnings
from sklearn.linear_model import LinearRegression

operators={"+": [operator.add, 0, "left"],
           "-": [operator.sub, 0, "left"],
           "*": [operator.mul, 1, "left"],
           "/": [operator.truediv, 1, "left"],
           "÷": [operator.truediv, 1, "left"],
           "**": [operator.pow,2, "right"],
           "^": [operator.pow,2, "right"],
           "(": ["para_open",3],
           ")": ["para_close",3]}

def powerset(iterable):
    """Returns powerset of iterables"""
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))

def weighter_r2 (x,y):
    """Shapley weights for r2 module"""
    return factorial(x)*factorial(y-x-1)/factorial(y)

def flatten(list_of_list, preserve_nest=True):
    """
    Flattens the list of lists (of nth level) without breaking the integrity of
    the non-list elements. preserve_nest boolean variable is True if the input
    variable (nested list) is to be unaltered after the execution of flatten. if
    preserve_nest is false, after the execution, input variable nested list
    become an empty list, unpreserved with its original content. This option
    provides a slight improvement in speed performance.

    Parameters:
    ----------
        list_of_list (list): A list with any levels of sub-lists

    Returns:
    ----------
        flatten_list (list): Flattened list

    Notes:
    ----------
        .. versionchanged:: 0.0.2
    """

    if preserve_nest == True:
        list_of_list_copy=list_of_list[:]
        flatten_list=[]
        while len(list_of_list_copy) > 0:
            for element in list_of_list_copy:
                if isinstance(element, list):
                    del list_of_list_copy[0]
                    element.reverse()
                    for n in element:
                        list_of_list_copy.insert(0,n)
                    element.reverse()
                    break
                else:
                    flatten_list.append(element)
                    del list_of_list_copy[0]
                    break
        return flatten_list
    else:
        flatten_list=[]
        while len(list_of_list) > 0:
            for element in list_of_list:
                if isinstance(element, list):
                    del list_of_list[0]
                    element.reverse()
                    for n in element:
                        list_of_list.insert(0,n)
                    break
                else:
                    flatten_list.append(element)
                    del list_of_list[0]
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
    ----------
        function (str) : Input function in text format (right hand side of equation)

    Returns:
    ----------
        [flatten(main_stack),var_count] (list) : A list containing, function in
        reverse polish notation (list) and count of variables (int)

    Notes:
    ----------
        .. versionchanged:: 0.0.2
    """

    def operator_splitter(func,op_list):
        only_var = func[:]
        for operator in op_list:
            if isinstance(func, str):
                only_var = only_var.split(operator)
                func = func.split(operator)
                if len(func) > 1:
                    for index in range(1,len(func)):
                        #operators are inserted between what they splitted, thus odd indices
                        func.insert(index*2-1,operator)
            else:
                func = [element.split(operator) for element in flatten(func, preserve_nest=False)]
                only_var = [element.split(operator) for element in flatten(only_var, preserve_nest=False)]
                for parts in func:
                    if len(parts) > 1:
                        for index in range(1, len(parts)):
                            #operators are inserted between what they splitted, thus odd indices
                            parts.insert(index*2-1,operator)

        #clean the left-over empty lists from split operations
        func = [element for element in flatten(func, preserve_nest=False) if len(element) != 0]
        only_var = [element for element in flatten(only_var, preserve_nest=False) if len(element) != 0]

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
    real_variables=[]
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
    real_variable_pos = []
    for char in splitted_function:
        if char in operators.keys():
            if len(operator_stack) == 0:
                #if operator stack is empty, append it
                operator_stack.append(char)
            else:
                # 4 conditions to check due to paranthesis operations
                if "(" not in operator_stack and char != "(":
                    if operators[char][1] > operators[operator_stack[0]][1]:
                        # if the new operator is of higher degree than the ones in the stack, add it into the stack first position
                        operator_stack.insert(0,char)
                    else:
                        # if the new operator is of lower degree or equal to the ones in the stack, remove those from the operator stack and add to the main_stack
                        if char == "**" or char == "^":
                            # power operator is right associative thus add
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
                if isinstance(float(char),float):
                    main_stack.append(float(char))
            except:
                main_stack.append(char)
                real_variable_pos.append(len(flatten(main_stack))-1)

    for remaining_operators in operator_stack:
        main_stack.append(remaining_operators)

    return [flatten(main_stack, preserve_nest=False), var_count, real_variable_pos]

def rpn_calc(finish_func):
    """
    Reverse Polish Notation evaluater

    Parameters:
    ----------
        finish_func (list): The function with real parameters

    Returns:
    ----------
        operation_stack[0] (float/int): Final numerical value after operations

    Notes:
    ----------
        .. versionchanged:: 0.0.2
    """

    operation_stack=[]
    for char in finish_func:
        if char in operators.keys():
            result = operators[char][0](operation_stack[-2], operation_stack[-1])
            del operation_stack[-2:]
            operation_stack.append(result)
        else:
            operation_stack.append(char)
    return operation_stack[0]

def cagr_calc(start,end,dur):
    """ Calculates compound annual growth rate with start-end dates and duration between. """

    return ((((int(end)/int(start))**(1/dur)))-1)*100

def frame_maker(array, mode=1):
    """ Converts arrays or list of lists as inputs to pandas dataframe. Seperate modes for shapley_change(1) and shapley_owen(2)"""

    if mode == 1:
        if type(array) != pandas.core.frame.DataFrame:
            data=pandas.DataFrame(array,index=["x"+str(ext) if ext !=0 else "y" for ext in range(0,len(array))])
            data.columns=[str(m) for m in data.columns.tolist()]
            return data
        else:
            t_cols = [str(col) for col in array.columns.tolist()]
            array.columns = t_cols
            return array
    elif mode == 2:
        if type(array) != pandas.core.frame.DataFrame:
            data=pandas.DataFrame(array,columns=["x"+str(ext) if ext != len(array[0]) else "y" for ext in range(1,len(array[0])+1)])
            return data
        else:
            return array

def s_sequence(k):
    """s parameter in weight component follows the A000120 from The On-line
    Encyclopedia of Integer Solutions (oeis.org) due to cartesian product
    implementation for variables in samples function (variables with 2 instances)
    thus s_sequence is its generation function.

    Parameters:
    ----------
        k (int): input to 2**k, number of variables - 1

    Returns:
    ----------
        seq (list): Value of s in cartesian product

    Notes:
    ----------
        .. versionadded:: 0.0.2
    """

    seq=[0]
    m=0
    while m != k:
        seq=seq[:2**m]+[n+1 for n in seq[:2**m]]
        m+=1
    yield from seq


def rsquared(x, y):
    """Calculates r_squared for x,y pair"""
    if x.shape[1] == 0:
        #array is zero
        r_squared=0
    else:
        model = LinearRegression()
        model.fit(x, y)
        r_squared = model.score(x, y)
    return r_squared
