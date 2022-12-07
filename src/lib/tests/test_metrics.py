from src.lib.metrics import foo, pfbeta

def test_foo() -> None:
    assert foo(1) == 2

def test_pfbeta() -> None:
    labels = [0,0,0,1,1,1]
    predictions = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    beta = 1.0
    score = pfbeta(labels, predictions, beta)
    proper_score = 0.9
    assert abs(score - proper_score) < 1e-9
