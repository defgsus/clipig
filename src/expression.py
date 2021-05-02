from math import *
from random import *
from typing import Union, List


class Expression:

    def __init__(self, expression: str, *arguments: str):
        self.expression = expression
        self.arguments = list(arguments)

        params = ", ".join(self.arguments)
        code = f"lambda {params}: {self.expression}"

        self.code = compile(code, filename="expression", mode="eval")
        self.validate()

    def validate(self):
        args = {
            name: 0.
            for name in self.arguments
        }
        self(**args)

    def __call__(self, **arguments):
        return eval(self.code, globals())(**arguments)


class ExpressionContext:

    def __init__(self, **arguments):
        self.arguments = arguments

    def __call__(
            self,
            expr: Union[int, float, Expression, List[Union[int, float, Expression]]],
    ):
        def _convert(e):
            if isinstance(e, (int, float)):
                return e
            elif isinstance(e, Expression):
                return e(**self.arguments)
            else:
                raise TypeError(f"Can not evaluate expression of type '{type(e).__name__}'")

        if isinstance(expr, (list, tuple)):
            return [_convert(e) for e in expr]
        else:
            return _convert(expr)
