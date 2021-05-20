from math import *
from random import *
from typing import Union, List, Optional, Type, Sequence


EXPRESSION_ARGS = {
    "basic": {
        "name": "basic",
        "doc": """
        Holds variables that reference the current frame time.
        """,
        "args": {
            "epoch": {
                "type": "int",
                "doc": "The current epoch / frame, starting at zero.",
            },
            "time": {
                "type": "float",
                "doc": """
                The current epoch / frame divided by the number of epochs, or in
                other words: A float ranging from **0.0** (start of training) to 
                **1.0** (end of training).
                """,
                "alias": "t",
            },
            "time2": {"alias": "t2"},
            "time3": {"alias": "t3"},
            "time4": {"alias": "t4"},
            "time5": {"alias": "t5"},
            "time_inverse": {
                "type": "float",
                "doc": """
                One minus the current epoch / frame divided by the number of epochs, or in
                other words: A float ranging from **1.0** (start of training) to 
                **0.0** (end of training).
                """,
                "alias": "ti",
            },
            "time_inverse2": {"alias": "ti2"},
            "time_inverse3": {"alias": "ti3"},
            "time_inverse4": {"alias": "ti4"},
            "time_inverse5": {"alias": "ti5"},
        }
    },

    "resolution": {
        "name": "resolution",
        "doc": """
        Holds the resolution of the training image.
        """,
        "args": {
            "resolution": {
                "type": "[int, int]",
                "doc": "The resolution of the training image as list of **width** and **height**.",
                "alias": "res",
            },
            "width": {
                "type": "int",
                "doc": "The width of the training image.",
            },
            "height": {
                "type": "int",
                "doc": "The width of the training image.",
            },
        }
    },

    "learnrate": {
        "name": "learnrate",
        "doc": """
        The current values of [learnrate](#learnrate) and [learnrate_scale](#learnrate_scale)
        which can be expressions themselves.
        """,
        "args": {
            "learnrate": {
                "type": "float",
                "doc": """The currently used [learnrate](#learnrate)""",
                "alias": "lr",
            },
            "learnrate_scale": {
                "type": "float",
                "doc": """The currently used [learnrate_scale](#learnrate_scale)""",
                "alias": "lrs",
            }
        }
    },

    "target_feature": {
        "name": "target feature",
        "doc": """
        Variables available to [target features](#targetsfeatures)
        """,
        "args": {
            "similarity": {
                "type": "float",
                "doc": """
                The [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
                of the CLIP-representation of the current, transformed image area 
                with the desired feature.
                
                The value is in the range [-100, 100].
                """,
                "alias": "sim",
            }
        }
    },

    "target_constraint": {
        "name": "target constraint",
        "doc": """
        Variables available to [constraints](#targetsconstraints)
        """,
        "args": {
            "similarity": {
                "type": "float",
                "doc": """
                The mean of all [cosine similarities](https://en.wikipedia.org/wiki/Cosine_similarity)
                of the CLIP-representation of the current, transformed image area 
                with the desired features of this target.
                
                The value is in the range [-100, 100].
                """,
                "alias": "sim",
            }
        }
    }

}


class Expression:

    def __init__(self, type: Type, expression: str, groups: Sequence[str] = ()):
        self.type = type
        self.expression = expression
        self.groups = list(groups)
        self.arguments = dict()
        for group in self.groups:
            self.arguments.update(EXPRESSION_ARGS[group]["args"])

        params = set(name for name in self.arguments.keys())
        for a in self.arguments.values():
            if a.get("alias"):
                params.add(a["alias"])
        self.params = params

        params = ", ".join(sorted(params))
        code = f"lambda {params}: {self.expression}"
        self.code = compile(code, filename="expression", mode="eval")
        self.function = eval(self.code, globals())

        self.validate()

    def __repr__(self):
        args = ""
        if self.groups:
            args = ", (" + ", ".join(f"'{a}'" for a in self.groups) + ")"
        return f"{self.__class__.__name__}({self.type.__name__}, '{self.expression}'{args})"

    def validate(self):
        import traceback
        try:
            args = {
                name: 0.
                for name in self.arguments
            }
            self(**args)
        except Exception as e:
            raise ValueError(
                f"{type(e).__name__} in expression '{self.expression}': {e}\n{traceback.format_exc(-1)}"
            )

    def __call__(self, **arguments):
        result = self.function(**self._with_aliases(arguments))
        return self.type(result)

    def _with_aliases(self, arguments: dict) -> dict:
        args = arguments.copy()
        for name in arguments:
            if name not in self.arguments:
                raise NameError(f"Argument '{name}' supplied but not defined for {self}")
            if self.arguments[name].get("alias"):
                args[ self.arguments[name]["alias"]] = arguments[name]
        return args


class ExpressionContext:

    def __init__(self, **arguments):
        self.arguments = arguments.copy()

    def __call__(
            self,
            expr: Union[int, float, Expression, List[Union[int, float, Expression]]],
    ):
        def _convert(e):
            if isinstance(e, (int, float, str)):
                value = e
            elif isinstance(e, Expression):
                value = e(**self.arguments)
            else:
                raise TypeError(f"Can not evaluate expression of type '{type(e).__name__}': '{e}'")
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
