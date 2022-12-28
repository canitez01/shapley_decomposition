import pandas
import numpy
from itertools import combinations
from copy import deepcopy
from math import factorial
import operator
import warnings

operators={"+": [operator.add, 0, "left"] ,
           "-": [operator.sub, 0, "left"] ,
           "*": [operator.mul, 1, "left"] ,
           "/": [operator.truediv, 1, "left"],
           "÷": [operator.truediv, 1, "left"],
           "**": [operator.pow,2, "right"],
           "^": [operator.pow,2, "right"],
           "(": ["para_open",3],
           ")": ["para_close",3]}

def prune(dataframe, dep_y, ind_x):
    df2= dataframe[(dataframe.index != dep_y)&(dataframe.index != ind_x)]
    return df2

def shapley_set_v2(dataframe, ind_x):
    main_var=[]
    var_group=[]
    for m in dataframe.index.tolist():
        c=[]
        for n in dataframe.columns.tolist():
            c.append(m+"-"+str(n))
            main_var.append(m+"-"+str(n))
        var_group.append(c)
    comb_list=list(combinations(main_var,len(dataframe.index)))
    shapley_list=[]

    for combos in comb_list:
        count=0
        for i, varis in enumerate(combos):
            if varis in var_group[i]:
                count +=1
        if count == len(dataframe.index):
            shapley_list.append(combos)

    shapley_list2=[list(i) for i in shapley_list]
    return shapley_list2

def shunting_yard(a2):
    def flatten (ms2): #flattening the list of lists (of every degree) without breaking the integrity of the elements (text with multiple characters)
        ms=deepcopy(ms2)
        rpn_result=[]
        while len(ms)>0:
            for val in ms:
                if type(val)==list:
                    del ms[0]
                    val.reverse()
                    for n in val:
                        ms.insert(0,n)
                    break
                else:
                    rpn_result.append(val)
                    del ms[0]
                    break
        return rpn_result

    def operator_splitter(func,op_list):
        only_var=deepcopy(func)
        for op in op_list:
            if type(func)==str:
                func=func.split(op)
                only_var=only_var.split(op)
                if len(func)>1:
                    for i in range(1,len(func)):
                        func.insert(i*2-1,op)
            else:
                func=[mid.split(op) for mid in flatten(func)]
                only_var=[mid.split(op) for mid in flatten(only_var)]
                for parts in func:
                    if len(parts) > 1:
                        for i in range(1, len(parts)):
                            parts.insert(i*2-1,op)
        func=[empt for empt in flatten(func) if len(empt) != 0]
        only_var=[empt for empt in flatten(only_var) if len(empt) != 0]

        for i,c in enumerate(func):
            if c=="*" and func[i+1] == "*":
                del func[i:i+2]
                func.insert(i,"**")
        return [func, only_var]

    returning=operator_splitter(a2,operators)
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
            if len(operator_stack)==0: #if operator stack is empty, without checking further ifs append the operator
                operator_stack.append(char)
            else:
                if "(" not in operator_stack and char !="(":
                    if operators[char][1] > operators[operator_stack[0]][1]: #if the new operator is of higher degree than the ones in the stack, we add it into the stack first position
                        operator_stack.insert(0,char)

                    else: # if the new operator is of lower degree or equal to the ones in the stack, we pop those from the operator stack and add to the main_stack
                        if char == "**" or char =="^": # power operator is right associative thus we add
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

def mapper_v3(dataframe, segment, function):
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

def shapley_calc_v2(dataframe,ind_x,y1,y2,sample,function):
    segments=[]
    for segm in sample:
        segm1=[ind_x+"-"+str(y1)] + segm
        segm2=[ind_x+"-"+str(y2)] + segm
        segments.append(mapper_v3(dataframe,segm2,function)-mapper_v3(dataframe,segm1,function))
    return segments

def s_compute(sample,t1):
    s_counts=[]
    for w in sample:
        counter=0
        for d in w:
            a=d.split("-")
            if t1 in a[1]:
                counter +=1
        s_counts.append(counter)
    return s_counts

def weighter(m,sample,t1):
    weights=[]
    computed_s=s_compute(sample,t1)
    for ss in computed_s:
        weight=(factorial(ss)*factorial(m-ss-1))/factorial(m)
        weights.append(weight)
    return numpy.array(weights)

def master_v2(dataframe,dep_y,ind_x,t0,t1,function):
    pruned=prune(dataframe,dep_y,ind_x)
    shapley_sample=shapley_set_v2(pruned,ind_x)
    return sum(numpy.array(shapley_calc_v2(dataframe,ind_x,t0,t1,
                                        shapley_sample,function))*weighter(len(dataframe.index)-1,shapley_sample,t1))

def cagr_calc(start,end,dur):
    return ((((int(end)/int(start))**(1/dur)))-1)*100

def frame_maker(array):
    data=pandas.DataFrame(array,index=["x"+str(ext) if ext !=0 else "y" for ext in range(0,len(array))])
    data.columns=[str(m) for m in data.columns.tolist()]
    return data

def shapley(dataframe,function, cagr=False):
    if type(dataframe) != pandas.core.frame.DataFrame:
        dataframe=frame_maker(dataframe)

    dep_y = dataframe.index[0]
    warnings.warn("Check the dataframe as the dependent variable(y) should be the first in position i.e at index 0")

    df_fin=dataframe.copy()
    df_fin["dif"]=[x-y for x,y in zip(dataframe.loc[:,dataframe.columns[1]].tolist(),dataframe.loc[:,dataframe.columns[0]].tolist())]

    df_fin["shapley"]=[master_v2(dataframe,dep_y,x, dataframe.columns[0], dataframe.columns[1],function) if x !=dep_y else df_fin.loc[dep_y,"dif"] for x in df_fin.index.tolist()]

    df_fin["contribution"]=[m/df_fin.loc[dep_y,"shapley"] for m in df_fin["shapley"].tolist()]

    if 0.9999 < df_fin["contribution"].sum()-1 < 1.0001:
        pass
    else:
        raise ValueError('Contribution of variables either exceeds or fail to reach 100 within +-0.0001 precision. Check both the input function and data.')

    if cagr==True:
        df_fin["yearly_growth"]=[cagr_calc(dataframe.loc[dep_y, dataframe.columns[0]], dataframe.loc[dep_y,dataframe.columns[1]], (int(dataframe.columns[1])-int(dataframe.columns[0])))*n
        for n in df_fin.contribution.tolist()]
    return df_fin
