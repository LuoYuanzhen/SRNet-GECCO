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
        return torch.where(torch.abs(args[1]) != torch.tensor(1e-5), args[0] / args[1], torch.tensor(0.))

    def sp_func(self, *args):
        x0, x1 = args
        if isinstance(x1, float):
            return 0. if abs(x1) <= 1e-5 else x0 / x1
        if not isinstance(x1, sp.Symbol) and x1.is_number and sp.Abs(x1) <= 1e-5:
            return 0.
        return x0 / x1


class Sqrt(_Function):
    def __init__(self):
        _Function.__init__(self, 'sqrt', 1)

    def pt_func(self, *args):
        return torch.where(args[0] >= 0, torch.sqrt(args[0]), args[0])

    def sp_func(self, *args):
        x0, = args
        if isinstance(x0, float):
            return x0 if x0 < 0 else sp.sqrt(x0)
        if not isinstance(x0, sp.Symbol) and x0.is_number and x0 < 0:
            return x0
        return sp.sqrt(x0)


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
        return torch.where(args[0] <= torch.tensor(1e-5), args[0], torch.log(args[0]))

    def sp_func(self, *args):
        x0, = args
        if isinstance(x0, float):
            return x0 if x0 <= 1e-5 else sp.log(x0)
        if not isinstance(x0, sp.Symbol) and x0.is_number and x0 <= 1e-5:
            return x0
        return sp.log(x0)


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
}


default_functions = ['add', 'sub', 'mul', 'div', 'sqrt', 'sqre', 'ln', 'sin', 'cos', 'tan']



