import sympy
import sympy as sp
import torch


class _Function:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def pt_func(self, *args):
        return None

    def sp_func(self, *args):
        return None

    def __call__(self, *args, is_pt=True):
        if is_pt:
            return self.pt_func(*args)

        sym_args = []
        for arg in args:
            if isinstance(arg, str):
                sym_args.append(sp.Symbol(arg))
            else:
                sym_args.append(arg)

        return self.sp_func(*sym_args)

    def __repr__(self):
        return self.name


class ITF(_Function):
    def __init__(self):
        _Function.__init__(self, 'ITF', 3)

    def pt_func(self, *args):
        return args[1] if torch.all(args[0] <= 0) else args[2]

    def sp_func(self, *args):
        return "IfThenElse({}<=0, {}, {})".format(args[0], args[1], args[2])


class Add(_Function):
    def __init__(self):
        _Function.__init__(self, 'add', 2)

    def pt_func(self, *args):
        return args[0] + args[1]

    def sp_func(self, *args):
        return args[0] + args[1]


class Sub(_Function):
    def __init__(self):
        _Function.__init__(self, 'sub', 2)

    def pt_func(self, *args):
        return args[0] - args[1]

    def sp_func(self, *args):
        return args[0] - args[1]


class Mul(_Function):
    def __init__(self):
        _Function.__init__(self, 'mul', 2)

    def pt_func(self, *args):
        return args[0] * args[1]

    def sp_func(self, *args):
        return args[0] * args[1]


class Div(_Function):
    def __init__(self):
        _Function.__init__(self, 'div', 2)

    def pt_func(self, *args):
        return args[0] / args[1]

    def sp_func(self, *args):
        if args[1] == 0 and isinstance(args[0], float):
            return sp.zoo
        return args[0] / args[1]


class Sqrt(_Function):
    def __init__(self):
        _Function.__init__(self, 'sqrt', 1)

    def pt_func(self, *args):
        return torch.sqrt(args[0])

    def sp_func(self, *args):
        return sp.sqrt(args[0])


class Sqre(_Function):
    def __init__(self):
        _Function.__init__(self, 'sqre', 1)

    def pt_func(self, *args):
        return torch.square(args[0])

    def sp_func(self, *args):
        return args[0] ** 2


class Ln(_Function):
    def __init__(self):
        _Function.__init__(self, 'ln', 1)

    def pt_func(self, *args):
        return torch.log(args[0])

    def sp_func(self, *args):
        return sp.log(args[0])


class Sin(_Function):
    def __init__(self):
        _Function.__init__(self, 'sin', 1)

    def pt_func(self, *args):
        return torch.sin(args[0])

    def sp_func(self, *args):
        return sp.sin(args[0])


class Cos(_Function):
    def __init__(self):
        _Function.__init__(self, 'cos', 1)

    def pt_func(self, *args):
        return torch.cos(args[0])

    def sp_func(self, *args):
        return sp.cos(args[0])


class Tan(_Function):
    def __init__(self):
        _Function.__init__(self, 'tan', 1)

    def pt_func(self, *args):
        return torch.tan(args[0])

    def sp_func(self, *args):
        return sp.tan(args[0])


function_map = {
    'add': Add(),
    'sub': Sub(),
    'mul': Mul(),
    'div': Div(),
    'sqrt': Sqrt(),
    'sqre': Sqre(),
    'ln': Ln(),
    'sin': Sin(),
    'cos': Cos(),
    'tan': Tan(),
    'if_then_else': ITF()
}

default_functions = ['add', 'sub', 'mul', 'div', 'sqrt', 'sqre', 'ln', 'sin', 'cos', 'tan', 'if_then_else']


