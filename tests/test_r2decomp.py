from shapley_decomposition import shapley_r2, shared_tools
import numpy
import pandas
import pytest

@pytest.fixture
def dataf():
    def df_generator(val,scale):
      data = numpy.random.rand(val,scale)
      columns = ["x"+str(m) if m < val else "y" for m in range(1,val+1)]
      return pandas.DataFrame(data.transpose(), columns=columns)
    df = df_generator(5,100)
    return(df)

def test_r2_samples(dataf):
    r2_shap_samps = shapley_r2.samples(dataf)
    assert len(list(r2_shap_samps.keys())) == len(dataf.columns.tolist())-1

def test_r2_shapley(dataf):
    r2_shapley = shapley_r2.shapley_decomposition(dataf)
    testing_val = shared_tools.rsquared(dataf.iloc[:,:-1], dataf.iloc[:,-1])
    assert testing_val - 0.00001 < r2_shapley["shapley_values"].values.sum() < testing_val + 0.00001

def test_r2_owen(dataf):
    groups = [["x1","x2"],["x3"],["x4"]]
    r2_owen = shapley_r2.owen_decomposition(dataf, groups)
    testing_val = shared_tools.rsquared(dataf.iloc[:,:-1], dataf.iloc[:,-1])
    assert testing_val - 0.00001 < r2_owen[1]["owen_values"].values.sum() < testing_val + 0.00001

def test_wrong_input_group_list(dataf):
    with pytest.raises(TypeError):
        r2_owen = shapley_r2.owen_decomposition(dataf,[["x1","x2"],["x3",[]],["x4"]])
