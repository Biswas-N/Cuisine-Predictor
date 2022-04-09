import project2


def test_say_hello():
    got = project2.say_hello("Biswas")

    assert got == "Hello, Biswas"
