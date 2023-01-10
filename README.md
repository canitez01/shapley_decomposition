# Shapley Decomposition

This package consists of two applications of shapley values in descriptive analysis: 1) a generalized module for decomposing change over time, using shapley values[^1] (initially influenced by the World Bank's Job Structure tool[^2]) and 2) shapley_owen decomposition of R^2 (contribution of independent variables to a goodness of fit metric -R^2 in this case-) in linear regression models.

Decomposition is used to understand the individual contributions of variables to their interaction/result (or change in interaction/result). Within the context of the first application of shapley method in poverty change decompositions, see [^3].

## Notes

Identities or functions with independently moving variables have independent contributions to the result as well. Therefore this module is better useful for functions or identities with dependently moving variables (though it works as well for the independent movements). However it should be noted that being able to decompose the contribution of variables doesn't mean that the results are always interpretable. Many features of variables like; scale, dependency mode, change dynamics (slow paced/fast paced, instant/lagged), etc. deserves thorough attention when interpreting their individual contribution to the change or result.   

Both for the first and second application, the computation time increases logarithmically as the number of variables increase.

Shapley method for decomposing consists of two main steps. First the method takes all the possible (time variant as well, for the first application) unique instances of variable combinations that define the results in a specific order. Second it weights the results of these combinatorial variable selections and compute the end result accordingly. With these two main steps we get the individual contribution of independent variables, to the change between two instances (different years, states etc.) or to the result directly.

## Installation

Run the following to install

```python
pip install decomposition
```

## Workings

`shapley_decomposition.shapley_change()` function works with two user inputs; data and function:

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

3. If `shapey_decomposition.shapley_change(df,"your function", cagr=True)` is called, a yearly_growth (using compound annual growth rate - cagr) column will be added, which will index the decomposition to cagr of the y. Default is `cagr=False`.   

`shapley_decomposition.shapley_owen()` function works with a dataframe or array input.

  1. The expected format for the input dataframe or array is:

    |  | x1 | x2 | .. | xn | y |  
    | --- | --- | --- | --- | --- | --- |
    | 0 | x1_value | x2_value | ... | xn_value | y_value |
    | 1 | x1_value | x2_value | ... | xn_value | y_value |
    | 2 | x1_value | x2_value | ... | xn_value | y_value |
    | ... | ... | ... | ... | ... | ... |
    | n | x1_value | x2_value | ... | xn_value | y_value |

  2. As the computation time increases exponentially with the increase in number of variables. For the shapley_owen function a default upper variable limit of 10 variables has been set. However in users' own discretion more variables can be forced by calling the function as `shapley_decomposition.shapley_owen(df, force=True)`

## Examples

1. As the first influence for the model was from WB's Job Structure, accordingly first example is decomposition of change in value added per capita of Turkey from 2000 to 2018 according to `"x1*x2*x3*x4"` where x1 is value added per worker, x2 is employment rate, x3 is participation rate, x4 is share of 15-64 population in total population. This is an identity.

  ```python
  import pandas
  import shapley_decomposition
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
  shapley_decomposition.shapley_change(df,"x1*x2*x3*x4")
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
  import shapley_decomposition

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

  shapley_decomposition.shapley_change(df,"(x1-x2)/x3")
  ```
  |  | 0 | 1 | dif | shapley | contribution |
  | --- | --- | --- | --- | --- | --- |
  | **non_par_skew** |	0.065803 |	0.044443 |	-0.021359 |	-0.021359 |	1.000000 |
  | **mean** |	-0.247181 | -0.285440 	 |	-0.038259 | -0.036146 | 1.692288 |
  | **median** | -0.315957 |	-0.333088 | -0.017131 | 0.016184 | -0.757719 |
  | **std** |	1.045188 | 1.072090 | 0.026902 | -0.001398 | 0.065432 |

3. Third example uses shapley_owen decomposition:

  ```python
  import numpy as np
  import pandas
  import shapley_decomposition

  #some random data generation
  def df_generator(val,scale): #val is # of variables, scale is the length of variables
      data = np.random.rand(val,scale)
      columns=["x"+str(m) if m < val else "y" for m in range(1,val+1)]
      return pandas.DataFrame(data.transpose(), columns=columns)

  np.random.seed(210)

  df=df_generator(8,1000) # the 8th variable is y

  shapley_decomposition.shapley_owen(df)
  ```

  | | contribution |
  | --- | --- |
  | **x1** |	0.000035 |
  | **x2** |	0.001639 |
  | **x3** |	0.000641 |
  | **x4** |	0.000030 |
  | **x5** |	0.000472 |
  | **x6** |	0.000006 |
  | **x7** |	0.001185 |

  ```python
  # if the number of variables are more than 10

  df=df_generator(12,1000) # the 8th variable is y

  shapley_decomposition.shapley_owen(df, force=True)
  ```


[^1]: https://www.rand.org/content/dam/rand/pubs/papers/2021/P295.pdf
[^2]: https://datatopics.worldbank.org/jobsdiagnostics/jobs-tools.html
[^3]: https://link.springer.com/article/10.1007/s10888-011-9214-z
