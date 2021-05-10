from math import *
from random import *
from typing import Union, List, Optional, Type


class Expression:

    def __init__(self, type: Type, expression: str, *arguments: str):
        self.type = type
        self.expression = expression
        self.arguments = list(arguments)

        params = ", ".join(self.arguments)
        code = f"lambda {params}: {self.expression}"

        self.code = compile(code, filename="expression", mode="eval")
        self.validate()

    def __repr__(self):
        args = ""
        if self.arguments:
            args = ", " + ", ".join(f"'{a}'" for a in self.arguments)
        return f"{self.__class__.__name__}('{self.expression}'{args})"

    def validate(self):
        args = {
            name: 0.
            for name in self.arguments
        }
        self(**args)

    def __call__(self, **arguments):
        return self.type(eval(self.code, globals())(**arguments))


class ExpressionContext:

    def __init__(self, **arguments):
        self.arguments = arguments.copy()

    def __call__(
            self,
            expr: Union[int, float, Expression, List[Union[int, float, Expression]]],
            type: Optional[Type] = None,
    ):
        def _convert(e):
            if isinstance(e, (int, float)):
                value = e
            elif isinstance(e, Expression):
                value = e(**self.arguments)
            else:
                raise TypeError(f"Can not evaluate expression of type '{type(e).__name__}': '{e}'")
            if type is not None:
                value = type(value)
            return value

        try:
            if isinstance(expr, (list, tuple)):
                return [_convert(e) for e in expr]
            else:
                return _convert(expr)
        except Exception as e:
            raise e.__class__(f"{e}, for expression '{expr}', context variables: {self.arguments}")

    def add(self, **arguments) -> "ExpressionContext":
        """
        Add more arguments and return new ExpressionContext
        """
        args = self.arguments
        args.update(arguments)
        return self.__class__(**args)
