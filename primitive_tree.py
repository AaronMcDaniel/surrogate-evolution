"""
Monkey-patched PrimitiveTree allowing us to convert string representations of genomes to DEAP individuals.
"""


from collections import deque
import copy
import random
import re
from deap import creator, gp, base, tools
import toml
import primitives
import itertools

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


def type_fixed_mut(individual, pset, type_prob=0.9):
    if len(individual) < 2:
        return individual,

    arch_prim_names = set(
        primitive.name for primitive in itertools.chain(
            pset.primitives[primitives.Tensor3D],
            pset.primitives[primitives.FinalTensor],
            pset.primitives[primitives.Optimizer],
            pset.primitives[primitives.Scheduler]
        )
    )

    arch_indices = [i for i, node in enumerate(individual) if node.name in arch_prim_names]
    hyperparam_indices = [i for i, node in enumerate(individual) if node.name not in arch_prim_names]

    if not arch_indices:
        return individual,
    if not hyperparam_indices:
        return individual,

    if random.random() < type_prob:
        index = random.choice(arch_indices)
    else:
        index = random.choice(hyperparam_indices)

    node = individual[index]

    if node.arity == 0:
        term = random.choice(pset.terminals[node.ret])
        if type(term) is gp.MetaEphemeral:
            term = term()
        individual[index] = term
    else:
        prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
        individual[index] = random.choice(prims)

    return individual,


def type_fixed_mut_shrink(individual, pset, type_prob=0.9):
    if len(individual) < 3 or individual.height <= 1:
        return individual,

    arch_prim_names = set(
        primitive.name for primitive in itertools.chain(
            pset.primitives[primitives.Tensor3D],
            pset.primitives[primitives.FinalTensor],
            pset.primitives[primitives.Optimizer],
            pset.primitives[primitives.Scheduler]
        )
    )
    arch_indices = [i for i, node in enumerate(individual) if node.name in arch_prim_names]
    hyperparam_indices = [i for i, node in enumerate(individual) if node.name not in arch_prim_names]

    if not arch_indices or not hyperparam_indices:
        return individual,

    if random.random() < type_prob and hyperparam_indices:
        target_indices = hyperparam_indices
    else:
        target_indices = arch_indices

    iprims = [
        (i, node)
        for i, node in enumerate(individual[1:], 1)
        if isinstance(node, Primitive) and node.ret in node.args and i in target_indices
    ]

    if iprims:
        index, prim = random.choice(iprims)
        arg_idx = random.choice([i for i, type_ in enumerate(prim.args) if type_ == prim.ret])
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)

        slice_ = individual.searchSubtree(index)
        individual[slice_] = subtree

    return individual,