"""
Monkey-patched PrimitiveTree allowing us to convert string representations of genomes to DEAP individuals.
"""


from collections import deque
import re
from deap import gp  

class CustomPrimitiveTree(gp.PrimitiveTree):
    @classmethod
    def from_string(cls, string, pset):
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if token in pset.mapping:
                primitive = pset.mapping[token]

                expr.append(primitive)
                if isinstance(primitive, gp.Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if type_ is None:
                    type_ = type(token)

                expr.append(gp.Terminal(token, False, type_))
        return cls(expr)
        