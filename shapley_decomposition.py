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

def prune(dataframe, dep_y, ind_x):
    """
    Prunes the input dataframe to not include the y and selected x by index filter.
    shapley_set() uses pruned dataframe.
    """
    df2= dataframe[(dataframe.index != dep_y)&(dataframe.index != ind_x)]
    return df2

def shapley_set(dataframe):
    """
    Create combinations of variables and time/instances in their original positions.

    Uses pruned dataframe. main_var lists all x-t combinations. var_group lists
    the list of x-t combinations seperately ([x1-t1,x1-t2],[...]) and their original
    position. shapley_list's loop selects combinations only with xs in their
    original position.

    master() uses shapley sets created by shapley_set().

    Parameters:
        dataframe (pandas.core.frame.DataFrame) : Pruned dataframe, free off y and selected x

    Returns:
        shapley_set (list) : A list of ordered combinations of variable-instance pairs
    """

    main_var=[]
    var_group=[]
    for xs in dataframe.index.tolist():
        var_with_instance=[]
        for instance in dataframe.columns.tolist():
            var_with_instance.append(xs+"-"+str(instance))
            main_var.append(xs+"-"+str(instance))
        var_group.append(var_with_instance)
    comb_list=list(combinations(main_var,len(dataframe.index)))

    shapley_list=[]
    for combos in comb_list:
        count=0
        for i, varis in enumerate(combos):
            if varis in var_group[i]:
                count +=1
        if count == len(dataframe.index):
            shapley_list.append(combos)

    shapley_list=[list(i) for i in shapley_list]
    return shapley_list

def flatten (list_of_list):
    """
    Flattens the list of lists (of nth level) without breaking the integrity of the non-list elements.

    Parameters:
        list_of_list (list) : The list to be flattened off of sub-lists
    Returns:
        flatten_list (list) : Sub-list free flattened list
    """
    list_of_list_copy=deepcopy(list_of_list)
    flatten_list=[]
    while len(list_of_list_copy) > 0:
        for element in list_of_list_copy:
            if type(element)==list:
                #remove the sublist, extract its elements and add them back to the main list
                del list_of_list_copy[0]
                element.reverse()
                for n in element:
                    list_of_list_copy.insert(0,n)
                break
            else:
                #remove the non-list, add it to the final list
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
        only_var=deepcopy(func)
        for operator in op_list:
            if type(func)==str:
                only_var=only_var.split(operator)
                func=func.split(operator)
                if len(func)>1:
                    for index in range(1,len(func)):
                        #operators are inserted between what they splitted, thus odd indices
                        func.insert(index*2-1,operator)
            else:
                func=[element.split(operator) for element in flatten(func)]
                only_var=[element.split(operator) for element in flatten(only_var)]
                for parts in func:
                    if len(parts) > 1:
                        for index in range(1, len(parts)):
                            #operators are inserted between what they splitted, thus odd indices
                            parts.insert(index*2-1,operator)

        #clean the residue empty lists from split operations
        func=[element for element in flatten(func) if len(element) != 0]
        only_var=[element for element in flatten(only_var) if len(element) != 0]

        #in order not to confuse power-** with two multiplications-*
        for index,element in enumerate(func):
            if element=="*" and func[index+1] == "*":
                del func[index : index+2]
                func.insert(index,"**")
        return [func, only_var]

    returning=operator_splitter(function,operators)
    varis=returning[0]

    var_count=0
    for la in returning[1]:
        try:
            float(la)
        except:
            var_count +=1

    par_stack=0
    for ops in returning[0]:
        if ops == "(" or ops ==")":
            par_stack +=1

    if par_stack %2:
        raise ValueError("Uneven/Wrong number of paranthesis, check the input function")

    if varis[-1] in operators.keys() and varis[-1] != ")":
        raise ValueError("last character of the input function is an operator, function incomplete")

    main_stack=[]
    operator_stack=[]
    for char in varis:
        if char in operators.keys():
            if len(operator_stack)==0: #if operator stack is empty, append it
                operator_stack.append(char)
            else:
                # 4 conditions to check because of the paranthesis operations
                if "(" not in operator_stack and char !="(":
                    if operators[char][1] > operators[operator_stack[0]][1]:
                        #if the new operator is of higher degree than the ones in the stack, we add it into the stack first position
                        operator_stack.insert(0,char)

                    else:
                        # if the new operator is of lower degree or equal to the ones in the stack, we pop those from the operator stack and add to the main_stack
                        if char == "**" or char =="^":
                            # power operator is right associative thus we add
                            operator_stack.insert(0,char)
                        else:
                            to_rem=[c for c in operator_stack if operators[char][1]<=operators[c][1]]
                            operator_stack = [i for i in operator_stack if i not in to_rem]
                            main_stack.append(to_rem)
                            operator_stack.insert(0,char)

                elif "(" not in operator_stack and char =="(":
                    operator_stack.insert(0,char)

                elif "(" in operator_stack and char != "(":
                    par_holder = operator_stack.index("(")
                    if char == ")":
                        operator_stack.insert(0,char)
                        to_rem=[[m for m,n in enumerate(operator_stack) if ")" in n][0],[m for m,n in enumerate(operator_stack) if "(" in n][0]]
                        main_stack.append(operator_stack[to_rem[0]+1:to_rem[1]])
                        operator_stack=operator_stack[:to_rem[0]]+operator_stack[to_rem[1]+1:]

                    elif len(operator_stack) ==1:
                        operator_stack.insert(0,char)
                    else:
                        if operators[char][1] > operators[operator_stack[0]][1]: #if the new operator is of higher degree than the ones in the stack, we add it into the stack first position
                            operator_stack.insert(0,char)
                        else:
                            if char == "**" or char =="^": # power operator is right associative thus we add
                                operator_stack.insert(0,char)
                            else:
                                to_rem=[[m,n] for m,n in enumerate(operator_stack[:par_holder]) if operators[char][1]<= operators[n][1]]
                                operator_stack = operator_stack[len(to_rem):]
                                main_stack.append([x[1] for x in to_rem])
                                operator_stack.insert(0,char)
                else:
                    operator_stack.insert(0,char)
        else:
            try:
                if type(float(char))==float:
                    main_stack.append(float(char))
            except:
                main_stack.append(char)
    for remaining_operators in operator_stack:
        main_stack.append(remaining_operators)

    return [flatten(main_stack),var_count]

def RPN_calc(finish_func):
    stack=[]
    for c in finish_func:
        if c in operators.keys():
            arg2=stack.pop()
            arg1=stack.pop()
            result = operators[c][0](arg1, arg2)
            stack.append(result)
        else:
            stack.append(c)
    return stack.pop()

def mapper(dataframe, segment, function):
    variable_dict={}
    for count, nom in enumerate(dataframe.index.tolist()):
        if count != 0:
            variable_dict[nom]=["x"+str(count)]
    for n in segment:
        variable_dict[n.split("-")[0]].append(dataframe.loc[n.split("-")[0],n.split("-")[1]])
    inv_map = {v[0]: [k,v[1]] for k, v in variable_dict.items()}
    if shunting_yard(function)[1] == len(inv_map.keys()):
        raw_func=shunting_yard(function)[0]
        finish_func=[]
        for n in raw_func:
            if n in inv_map.keys():
                finish_func.append(inv_map[n][1])
            else:
                finish_func.append(n)
        return RPN_calc(finish_func)
    else:
        raise ValueError('Number of variables in function and data are not equal. Check both the input function and data.')

def shapley_calc(dataframe,ind_x,y1,y2,sample,function):
    segments=[]
    for segm in sample:
        segm1=[ind_x+"-"+str(y1)] + segm
        segm2=[ind_x+"-"+str(y2)] + segm
        segments.append(mapper(dataframe,segm2,function)-mapper(dataframe,segm1,function))
    return segments

def s_compute(sample, t1, owen=False):
    s_counts=[]
    for w in sample:
        counter=0
        if owen == True: #for owen_shapley decomposition, number of variables other than
            counter = len(w)
        else:
            for d in w:
                a=d.split("-")
                if t1 in a[1]:
                    counter +=1
        s_counts.append(counter)
    return s_counts

def weighter(m,sample,t1, owen=False):
    weights=[]
    if owen ==True: # for owen_shapley decomposition
        computed_s=s_compute(sample,t1, owen=True)
    else:
        computed_s=s_compute(sample,t1)
    for ss in computed_s:
        weight=(factorial(ss)*factorial(m-ss-1))/factorial(m)
        weights.append(weight)
    return numpy.array(weights)

def master(dataframe,dep_y,ind_x,t0,t1,function):
    pruned=prune(dataframe,dep_y,ind_x)
    shapley_sample=shapley_set(pruned)
    return sum(numpy.array(shapley_calc(dataframe,ind_x,t0,t1,
                                        shapley_sample,function))*weighter(len(dataframe.index)-1,shapley_sample,t1))

def cagr_calc(start,end,dur):
    return ((((int(end)/int(start))**(1/dur)))-1)*100

def frame_maker(array, mode=1):
    if mode == 1:
        data=pandas.DataFrame(array,index=["x"+str(ext) if ext !=0 else "y" for ext in range(0,len(array))])
        data.columns=[str(m) for m in data.columns.tolist()]
        return data
    elif mode == 2:
        data=pandas.DataFrame(array,columns=["x"+str(ext) if ext != len(array[0]) else "y" for ext in range(1,len(array[0])+1)])
        return data

def shapley_change(dataframe,function, cagr=False):
    if type(dataframe) != pandas.core.frame.DataFrame:
        dataframe=frame_maker(dataframe)

    dep_y = dataframe.index[0]
    warnings.warn("Check the dataframe as the dependent variable(y) should be the first in position i.e at index 0")

    t_cols=[str(col) for col in dataframe.columns.tolist()]
    dataframe.columns=t_cols

    df_fin=dataframe.copy()
    df_fin["dif"]=[x-y for x,y in zip(dataframe.loc[:,dataframe.columns[1]].tolist(),dataframe.loc[:,dataframe.columns[0]].tolist())]
    df_fin["shapley"]=[master(dataframe,dep_y,x, dataframe.columns[0], dataframe.columns[1],function) if x !=dep_y else df_fin.loc[dep_y,"dif"] for x in df_fin.index.tolist()]
    df_fin["contribution"]=[m/df_fin.loc[dep_y,"shapley"] for m in df_fin["shapley"].tolist()]

    if 0.9999 < df_fin["contribution"].sum()-1 < 1.0001:
        pass
    else:
        raise ValueError('Contribution of variables either exceeds or fail to reach 1.0 within +-0.0001 precision. Check both the input function and data.')

    if cagr==True:
        df_fin["yearly_growth"]=[cagr_calc(dataframe.loc[dep_y, dataframe.columns[0]], dataframe.loc[dep_y,dataframe.columns[1]], (float(dataframe.columns[1])-float(dataframe.columns[0])))*n
        for n in df_fin.contribution.tolist()]
    return df_fin

def shapley_owen(dataframe, force=False):

    def rsquared(x, y):
        model = LinearRegression()
        model.fit(x, y)
        r_squared = model.score(x, y)
        return r_squared

    if type(dataframe) != pandas.core.frame.DataFrame:
        dataframe=frame_maker(dataframe, mode=2)

    variables=dataframe.columns.tolist()[:-1]

    if len(variables) > 10 and force == False:
        raise ValueError('Number of variables exceeds the limit. In your own discretion you can force more than 20 variables by inputting - force=True - to the function. However, beware computation may take time')
    elif len(variables) > 10 and force == True:
        warnings.warn("As the number of variables increase, computation time and cost increase exponentially")
    else:
        pass

    comb_var=[list(combinations(variables,n)) for n in range(1,len(variables)+1)]
    comb_var2=flatten(comb_var)
    comb_var3=[list(n) for n in comb_var2]

    general=[]
    for n in variables:
        d2=deepcopy(comb_var3)
        samp=[] #in order to take combinations with the variable included
        for m in d2:
            if n in m:
                samp.append(m)
        b_with=[rsquared(dataframe.loc[:,f], dataframe["y"]) for f in samp]
        b_wo=[]
        for f in samp:
            f.remove(n) #we remove the variable itself to calc. r2 without it
            if len(f) == 0:
                b_wo.append(0)
            else:
                b_wo.append(rsquared(dataframe.loc[:,f], dataframe["y"]))
        diff=[x-t for x,t in zip(b_with,b_wo)]
        shapley_value=diff*weighter(len(variables),samp,n, owen=True)
        general.append(sum(shapley_value))

    df_fin=pandas.DataFrame(index=variables, columns=["contribution"])
    df_fin["contribution"]=general
    return df_fin
