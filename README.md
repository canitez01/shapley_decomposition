# Shapley Decomposition

Influenced by the World Bank's Job Structure tool[^1], this package consists of a generalized module for decomposing change over time with Shapley method[^2]. Decomposition is used to understand the individual contribution of variables to the change.

## Notes

Identities or functions with independently moving variables have independent contributions to the result as well. Therefore this module is better useful for functions or identities with dependently moving variables (though it works as well for the independent movements, it's just you do not need a module for that computation other than simple arithmetic). However it should be noted that being able to decompose the contribution of variables doesn't mean that the results are always interpretable. Many features of variables like; scale, dependency mode, change dynamics (slow paced/fast paced, instant/lagged), etc. deserves thorough attention when interpreting their individual contribution to the change.   

Shapley method for decomposing consists of two main attributes. First the method takes all the possible time variant instances of variable combinations that define the results in a specific order. Second it weights the results of these combinatorial variable selections and compute the end result accordingly. With these two main steps we get the individual contribution of independent variables, to the change between two instances (different years, states etc.) of dependent variables.

## Installation

Run the following to install

```python
pip install decomposition
```

## Workings

Module works with two user inputs; data and function:

1. The structure of input data is important. Module accepts pandas dataframes, arrays or list of lists:
  * If pandas dataframe is used as input, both the dependent variable and the independent variables should be presented in the given format:

    |  | year1 | year2 |
    | --- | ----------- | ----|
    | y | y_value | y_value |
    | x1 | x1_value | x1_value |
    | x2 | x2_value | x2_value |
    | ... | ... | ... |
    | xn | xn_value | xn_value |

  * If an array or a list format is preferred, note that module will convert it to a pandas dataframe format and expects y and xs in the following order. Names for xs and y is not required, when inputted as shown below, module will create the output dataframe with y,x1,x2.. as index column and 0 and 1 as name columns:
    ```
    [[y_value,y_value],
      [x1_value,x1_value],
      [x2_value,x2_value]]
      ...
    ```
2. Identity defines the relation between xs and y. Bear in mind, due to the characteristic of shapley decomposition (and thus identities) the sum of xs' contributions should be equal to y (exact with 0.0001 freedom), therefore no place for residuals, or an input relation that fails to create the given y will not work. Function input is expected text format. It is evaluated by a custom syntax parser (as the eval() function has its security risks, a custom syntax parser which translates the function in txt format to operable python format is used). Expected format for the function input is the right hand side of the equation:
    * ```"x1+x2*(x3/x4)**x5"```
    * ```"(x1+x2)*x3+x4"```
    * ```"x1*x2**2"```

## Examples

1. As the first influence for the model was from WB's Job Structure, accordingly first example is decomposition of change in value added per capita of Turkey from 2000 to 2018 according to ```x1*x2*x3*x4``` where x1 is value added per worker, x2 is employment rate, x3 is participation rate, x4 is share of 15-64 population in total population. This is an identity.

  ```python
  import pandas
  import decomposition
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
  decomposition.shapley(df,"x1*x2*x3*x4")
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
  import decomposition

  seed(210)

  data = np.random.normal(loc=0, scale=1, size=100)

  add = [np.random.uniform(min(data), max(data)) for m in range(5,10)]

  altered_data = np.concatenate([data,add])

  med1, med2 = np.median(data), np.median(altered_data)
  mean1, mean2 = np.mean(data), np.mean(altered_data)
  std1, std2 = np.std(data, ddof=1), np.std(altered_data, ddof=1)
  sk1 = (np.mean(data)-np.median(data))/np.std(data, ddof=1),
  sk2 = (np.mean(altered_data)-np.median(altered_data))/np.std(altered_data, ddof=1)

  df=pandas.DataFrame([[sk1,sk2],[mean1,mean2],[med1,med2],[std1,std2]], columns=["0","1"], index=["non_par_skew","mean","median","std"])

  decomposition.shapley(df,"(x1-x2)/x3")
  ```
  |  | 0 | 1 | dif | shapley | contribution |
  | --- | --- | --- | --- | --- | --- |
  | **non_par_skew** |	0.065803 |	0.044443 |	-0.021359 |	-0.021359 |	1.000000 |
  | **mean** |	-0.247181 | -0.285440 	 |	-0.038259 | -0.036146 | 1.692288 |
  | **median** | -0.315957 |	-0.333088 | -0.017131 | 0.016184 | -0.757719 |
  | **std** |	1.045188 | 1.072090 | 0.026902 | -0.001398 | 0.065432 |













[^1]: https://datatopics.worldbank.org/jobsdiagnostics/jobs-tools.html
[^2]: https://www.rand.org/content/dam/rand/pubs/papers/2021/P295.pdf
