from CGPNet.functions import function_map


class CGPParameters:
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_rows,
                 n_cols,
                 levels_back,
                 function_set,
                 n_eph=0
                 ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.levels_back = levels_back
        self.n_eph = n_eph
        self.function_set = []
        self.max_arity = 1
        for str_fun in function_set:
            if str_fun not in function_map:
                raise ValueError("%s function is not in 'function_map' in functions.py." % str_fun)
            self.max_arity = max(function_map[str_fun].arity, self.max_arity)
            self.function_set.append(function_map[str_fun])

        if self.levels_back is None:
            self.levels_back = n_rows * n_cols + n_inputs + 1


class NetParameters:
    def __init__(self,
                 neurons,
                 n_rows,
                 n_cols,
                 levels_back,
                 function_set,
                 n_eph=0,
                 add_bias=False
                 ):
        self.neurons = neurons
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.levels_back = levels_back
        self.function_set = function_set
        self.n_eph = n_eph
        self.add_bias = add_bias

    def __repr__(self):
        str = f"NetParameters(\n" \
              f"    neurons:{self.neurons},\n" \
              f"    n_rows:{self.n_rows},\n" \
              f"    n_cols:{self.n_cols},\n" \
              f"    function_set:{self.function_set},\n" \
              f"    n_eph:{self.n_eph}\n" \
              f")"
        return str
