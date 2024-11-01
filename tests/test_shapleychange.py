from shapley_decomposition import shapley_change
import pandas
import pytest

@pytest.fixture
def df():
    #setup example dataframe
    return pandas.DataFrame([[8237.599210,15026.707520],[27017.637990,43770.525560],[0.935050,0.891050],[0.515090,0.57619],[0.633046,0.668674]],index=["val_ad_pc","val_ad_pw","emp_rate","part_rate","working_age"], columns=["2000","2018"])

def test_sample_generation_and_input_variablility():
    df = pandas.DataFrame([[8237.599210,15026.707520],[27017.637990,43770.525560],[0.935050,0.891050],[0.515090,0.57619],[0.633046,0.668674]])
    for cols,ind in zip([[2000,2018],["2000","2018"],[2000,"2018"]],[["val_ad_pc","val_ad_pw","emp_rate","part_rate","working_age"],["y","x1","x2","x3","x4"],["val_ad_pc","val_ad_pw","emp_rate","part_rate","working_age"]]):
        df.columns=cols
        df.index=ind
        sample = shapley_change.samples(df)
        assert type(sample[0]) == dict
        assert len(sample[0]) > 0

def test_wrong_input_function(df):
    with pytest.raises(ValueError):
        shapleyvals = shapley_change.shapley_values(df,"x1*x2*x3")

def test_shapleyvalues(df):
    shapleyvals = shapley_change.shapley_values(df,"x1*x2*x3*x4")
    assert len(shapleyvals) == len(df.index.tolist()[1:])
    assert len(shapleyvals[0]) == 2**(len(df.index.tolist())-2)

def test_change_decomposition(df):
    change_decomp = shapley_change.decomposition(df,"x1*x2*x3*x4")
    assert type(change_decomp) == type(df)
    assert 0.9999 < change_decomp["contribution"][1:].sum() < 1.0001

def test_change_decomposition_longData_withNoChangeVariables():
    df2=pandas.read_csv("tests/long_sample.csv", index_col=0)
    change_decomp_longData_withNoChangeVariables = shapley_change.decomposition(df2,"x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16")
    assert type(change_decomp_longData_withNoChangeVariables) == type(df2)
    assert 0.9999 < change_decomp_longData_withNoChangeVariables["contribution"][1:].sum() < 1.0001
