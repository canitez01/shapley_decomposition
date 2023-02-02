from shapley_decomposition import shared_tools

def test_flatten():
    list_to_flatten = [[[[[[[[["12"]],"b"],"ci"]]],"vob","i"]]]
    flatlist = shared_tools.flatten(list_to_flatten)
    assert len(flatlist) == 5
    assert flatlist == ["12","b","ci","vob","i"]

def test_shuntingyard():
    func = "x1*3**x2*(x3+x4)/2.1"
    shunted = shared_tools.shunting_yard(func)
    assert shunted[1] == 4
    assert len(shunted[0])-4 == 7

def test_rpncalc():
    func = [0.1,42,14,"+","*",31,"+"]
    rpn = shared_tools.RPN_calc(func)
    assert rpn == 36.6
