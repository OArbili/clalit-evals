from clalit_evals import hello_world


def test_hello_world_prints_greeting(capsys):
    hello_world()
    assert capsys.readouterr().out == "Hello, World!\n"


def test_hello_world_returns_none():
    assert hello_world() is None
