# Shapley Decomposition

This package consists of two applications of shapley values in descriptive analysis: 1) a generalized module for decomposing change over time, using shapley values[^1] (initially influenced by the World Bank's Job Structure tool[^2]) and 2) shapley and owen values based decomposition of R^2 (contribution of independent variables to a goodness of fit metric -R^2 in this case-) for linear regression models[^3].

## Notes

Identities or functions with independently moving variables have independent contributions to the result as well. Therefore this module is better useful for functions or identities with dependently moving variables (though it works as well for the independent movements). It should be noted that being able to decompose the contribution of variables doesn't mean that the results are always clearly interpretable. Many features of variables like; scale, dependency mode, change dynamics (slow paced/fast paced, instant/lagged), etc. deserves attention when interpreting their individual contribution to the change or result.   

Both for the first and second application, the computation time increases exponentially as the number of variables increase. This is the result of powersets and so 2^n  calculations.

Shapley value follows:

$v(i) = \sum \limits _{S \subseteq M \setminus i} \phi(s) \cdot [V(S \cup \{i\})-V(S)]$

$\phi(s) = (m-1-s)! \cdot s!/m!$

where $i \in M$ and M is the main set of variables and $m=|M|, s=|S|$. For shapley change decomposition, $[V(S \cup \{i_{t_1} \})-V(S\cup \{i_{t_0} \})]$ and s is the number of variables with $t_1$ instance.  

Owen value follows:

$o(i) = \sum \limits _{R \subseteq N \setminus k} \sum \limits _{T \subseteq B_k \setminus i} \phi(r) \cdot \omega(t) \cdot [V(Q \cup T \cup \{i\})-V(Q \cup T)]$

$\phi(r) = (n-1-r)! \cdot r!/n!$

$\phi(t) = (b_k-1-t)! \cdot t!/b_k!$

where $i \in M$ and M is the main set of variables. N is the powerset of coalition/group set composed of i individuals.  $Q = \bigcup_{r \in R}B_r$ and $n=|N|, r=|R|, b_k=|B_k|, t=|T|$.

## Installation

Run the following to install

```python
pip install shapley_decomposition
```

## Workings

`shapley_decomposition.shapley_change` module consists of three functions: `samples()`, `shapley_values()` and `decomposition()`. `shapley_change.samples(dataframe)` returns powerset pairs that model uses. `shapley_change.shapley_values(dataframe, "your function")` returns weighted differences for each variable, sum of which gives the shapley value. `shapley_change.decomposition(dataframe, "your function")` returns decomposed total change by variable contributions. These functions of shapley_change module accepts either or both of the **data** and **function** inputs:

1. The structure of input data is **important**. Module accepts pandas dataframes or 2d arrays:
  * If pandas dataframe is used as input, both the dependent variable and the independent variables should be presented in the given format (variable names as index and years as columns):

    |  | year1 | year2 |
    | --- | ----------- | ----|
    | **y** | y_value | y_value |
    | **x1** | x1_value | x1_value |
    | **x2** | x2_value | x2_value |
    | **...** | ... | ... |
    | **xn** | xn_value | xn_value |

  * If an array is preferred, note that module will convert it to a pandas dataframe format and expects y and xs in the following order:
    ```
    [[y_value,y_value],
      [x1_value,x1_value],
      [x2_value,x2_value]]
      ...
    ```
2. Identity or function defines the relation between xs and y. Due to the characteristic of shapley decomposition the sum of xs' contributions should be equal to y, i.e. exact (with plus minus 0.0001 freedom in this module due to the residue of arithmetic operations), therefore no place for residuals, or an input relation that fails to create the given y will shoot a specific error. Function input is expected in text format. It is evaluated by a custom syntax parser (as the eval() function has its security risks). Expected format for the function input is the right hand side of the equation:

    * `"x1+x2*(x3/x4)**x5"`
    * `"(x1+x2)*x3+x4"`
    * `"x1*x2**2"`

    All arithmetic operators and paranthesis operations are usable:
    * `"+" , "-" , "*" , "/" or "÷", "**" or "^"`

3. If `shapley_change.decomposition(df,"your function", cagr=True)` is called, a yearly_growth (using compound annual growth rate - cagr) column will be added, which will index the decomposition to cagr of the y. Default is `cagr=False`.   

The `shapley_decomposition.shapley_r2` module consists of three functions as well: `samples()`, `shapley_decomposition()` and `owen_decomposition`. `shapley_r2.samples(dataframe)` returns powerset variable pairs that model uses. `shapley_r2.shapley_decomposition(dataframe)` returns the decomposition of model r^2 to the contributions of variables. `shapley_r2.owen_decomposition(dataframe, [["x1","x2"],[..]])` returns the owen decomposition of model r^2 to the contributions of variables and groups/coalitions. Input features expected by shapley_r2 functions are as:

  1. The expected format for the input dataframe or array is:

  |  | x1 | x2 | ... | xn | y |  
  | --- | --- | --- | --- | --- | --- |
  | **0** | x1_value | x2_value | ... | xn_value | y_value |
  | **1** | x1_value | x2_value | ... | xn_value | y_value |
  | **2** | x1_value | x2_value | ... | xn_value | y_value |
  | **...** | ... | ... | ... | ... | ... |
  | **n** | x1_value | x2_value | ... | xn_value | y_value |


  2. `shapley_r2.owen_decomposition` expects the group/coalition structure as the second input. This input should be a list of list showing the variables grouped within coalition/group lists. For example a model of 8 variables, x1,x2,...,x8 has three groups/coalitions which are formed as group1:(x1,x2,x3), group2:(x4) and group3:(x5,x6,x7,x8). Then the second input of owen_decomposition should be `[["x1","x2","x3"],["x4"],["x5","x6","x7","x8"]]`. Even if it is a singleton like group2 which has only x4, variable name should be in a list. If every group is a singleton, then the owen values will be equal to shapley values.

  3. As the computation time increases exponentially with the number of variables. For the shapley_decomposition function a default upper variable limit of 10 variables has been set. Same limit applies for owen_decomposition but as the number of groups, not individual variables. However in users' own discretion more variables can be forced by calling the function as `shapley_r2.shapley_decomposition(df, force=True)` or `shapley_r2.owen_decomposition(df, [groups], force=True)`.

## Examples

1. As the first influence for the model was from WB's Job Structure, accordingly first example is decomposition of change in value added per capita of Turkey from 2000 to 2018 according to `"x1*x2*x3*x4"` where x1 is value added per worker, x2 is employment rate, x3 is participation rate, x4 is share of 15-64 population in total population. This is an identity.

  ```python
  import pandas
  from shapley_decomposition import shapley_change

  df=pandas.DataFrame([[8237.599210,15026.707520],[27017.637990,43770.525560],[0.935050,0.891050],[0.515090,0.57619],[0.633046,0.668674]],index=["val_ad_pc","val_ad_pw","emp_rate","part_rate","working_age"], columns=[2000,2018])
  print(df)
  ```
  |  | 2000 | 2018 |
  | --- | ----------- | ----|
  | **val_ad_pc** | 8237.599210 | 15026.707520 |
  | **val_ad_pw** | 27017.637990 | 43770.525560 |
  | **emp_rate** | 0.935050 | 0.891050 |
  | **part_rate** | 0.515090 | 0.57619 |
  | **part_rate** | 0.633046 | 0.668674 |

  ```python
  shapley_change.decomposition(df,"x1*x2*x3*x4")
  ```
  |  | 2000 | 2018 | dif | shapley | contribution |
  | --- | --- | --- | --- | --- | --- |
  | **val_ad_pc** |	8237.599210 |	15026.707520 |	6789.108310 |	6789.108310 |	1.000000 |
  | **val_ad_pw** |	27017.637990 | 43770.525560 |	16752.887570 | 5431.365538 | 0.800012 |
  | **emp_rate** | 0.935050 |	0.891050 | -0.044000 | -556.985657 | -0.082041 |
  | **part_rate** |	0.515090 | 0.576190 | 0.061100 | 1285.200011 | 0.189303 |
  | **working_age** |	0.633046 | 0.668674 |	0.035628 | 629.528410 |	0.092726 |

2. Second example is the decomposition of change in non-parametric skewness of a normally distributed sample, after the sample is altered with additional data. We are trying to understand how the change in mean, median and standard deviation contributed to the change in skewness parameter. Non parametric skewness is calculated by `"(x1-x2)/x3"`, (mean-median)/standard deviation.

  ```python
  import numpy as np
  import pandas
  from shapley_decomposition import shapley_change

  np.random.seed(210)

  data = np.random.normal(loc=0, scale=1, size=100)

  add = [np.random.uniform(min(data), max(data)) for m in range(5,10)]

  altered_data = np.concatenate([data,add])

  med1, med2 = np.median(data), np.median(altered_data)
  mean1, mean2 = np.mean(data), np.mean(altered_data)
  std1, std2 = np.std(data, ddof=1), np.std(altered_data, ddof=1)
  sk1 = (np.mean(data)-np.median(data))/np.std(data, ddof=1)
  sk2 = (np.mean(altered_data)-np.median(altered_data))/np.std(altered_data, ddof=1)

  df=pandas.DataFrame([[sk1,sk2],[mean1,mean2],[med1,med2],[std1,std2]], columns=["0","1"], index=["non_par_skew","mean","median","std"])

  shapley_change.decomposition(df,"(x1-x2)/x3")
  ```
  |  | 0 | 1 | dif | shapley | contribution |
  | --- | --- | --- | --- | --- | --- |
  | **non_par_skew** |	0.065803 |	0.044443 |	-0.021359 |	-0.021359 |	1.000000 |
  | **mean** |	-0.247181 | -0.285440 	 |	-0.038259 | -0.036146 | 1.692288 |
  | **median** | -0.315957 |	-0.333088 | -0.017131 | 0.016184 | -0.757719 |
  | **std** |	1.045188 | 1.072090 | 0.026902 | -0.001398 | 0.065432 |

3. Third example uses shapley_r2 decomposition with the fish market database from kaggle[^4]:

  ```python
  import numpy as np
  import pandas
  from shapley_decomposition import shapley_r2

  df=pandas.read_csv("Fish.csv")
  #ignoring the species column
  shapley_r2.shapley_decomposition(df.iloc[:,1:])
  ```
  | |shapley_values | contribution |
  | --| -- | --|
  | **Length1** |	0.194879 |	0.220131 |
  | **Length2** |	0.195497 |	0.220829 |
  | **Length3** |	0.198097 |	0.223766 |
  | **Height** |	0.116893 |	0.132040 |
  | **Width** |	0.179920 |	0.203233 |

  ```python
  #using the same dataframe

  groups = [["Length1","Length2","Length3"],["Height","Width"]]

  shapley_r2.owen_decomposition(df.iloc[:,1:], groups)
  ```



  | | owen_values | contribution | group_owen |
  | --- | --- | --- | --- |
  | **Length1** |	0.157523 | 0.177934 | b1 |
  | **Length2** |	0.158178 | 0.178674 | b1 |
  | **Length3** |	0.160276 | 0.181045 | b1 |
  | **Height** |	0.141092 | 0.159374 | b2 |
  | **Width** |	0.268218 | 0.302972 | b2 |


  | | owen_values | contribution |
  | -- | -- | -- |                         
  | **b1** | 0.475977 | 0.537653 |
  | **b2** | 0.409309 | 0.462347 |

  ```python
  # if the species are included as categorical variables

  from sklearn.preprocessing import OneHotEncoder

  enc = OneHotEncoder(handle_unknown="ignore")
  encti = enc.fit_transform(df[["Species"]]).toarray()

  df2 = pandas.concat([pandas.DataFrame(encti,columns=["Species"+str(n) for n in range(len(encti[0]+1))]),df.iloc[:,1:]],axis=1)
  shapley_r2.shapley_decomposition(df2, force=True) # as the number of variables bigger than 10, force=True

  groups=[["Species0","Species1","Species2","Species3","Species4","Species5","Species6"],["Length1","Length2","Length3"],["Height","Width"]]

  shapley_r2.owen_decomposition(df2, groups) #no need for force as the number of groups does not exceed 10
  ```

[^1]: https://www.rand.org/content/dam/rand/pubs/papers/2021/P295.pdf
[^2]: https://datatopics.worldbank.org/jobsdiagnostics/jobs-tools.html
[^3]: https://www.scitepress.org/papers/2017/61137/
[^4]: https://www.kaggle.com/datasets/aungpyaeap/fish-market?resource=download
