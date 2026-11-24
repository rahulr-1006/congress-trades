import pytest

from congress_trades.backtest.metrics import hit_rate, max_drawdown, sharpe, total_return


def test_total_return_compounds():
    assert total_return([0.1, -0.1]) == pytest.approx(-0.01)


def test_max_drawdown():
    assert max_drawdown([0.5, -0.5, 0.2]) == pytest.approx(0.5)
    assert max_drawdown([0.01, 0.02]) == 0


def test_sharpe_sign_and_degenerate_cases():
    assert sharpe([0.01, 0.02, 0.015]) > 0
    assert sharpe([0.01]) == 0
    assert sharpe([0.01, 0.01]) == 0


def test_hit_rate():
    assert hit_rate([0.1, -0.2, 0.3, 0.0]) == 0.5
    assert hit_rate([]) == 0
