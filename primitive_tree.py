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
        
# configs = toml.load('/home/hice1/tthakur9/scratch/surrogate-evolution/conf.toml')
# codec_config = configs["codec"]
# pipeline_config = configs["pipeline"]
# objectives = pipeline_config['objectives']

# def ensure_deap_classes(objectives, codec_config):
#     # Check if the 'FitnessMulti' class exists, if not, create it
#     if not hasattr(creator, 'FitnessMulti'):
#         creator.create("FitnessMulti", base.Fitness, weights=tuple(objectives.values()))

#     # TODO: add other cases for encoding strategy
#         genome_type = gp.PrimitiveTree # default
#         match codec_config["genome_encoding_strat"].lower():
#             case "tree":
#                 genome_type = gp.PrimitiveTree

#     # Check if the 'Individual' class exists, if not, create it
#     if not hasattr(creator, 'Individual'):
#         creator.create("Individual", genome_type, fitness=creator.FitnessMulti)

# ensure_deap_classes(objectives, codec_config)
# toolbox = base.Toolbox()
# toolbox.register("expr", gp.genHalfAndHalf, pset=primitives.pset, min_=4, max_=8)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("type_fixed_mut", eval("type_fixed_mut"))
# pop = toolbox.population(n=10)
# pset = primitives.pset
# arch_prim_names = set(
#     primitive.name for primitive in itertools.chain(
#         pset.primitives[primitives.Tensor3D],
#         pset.primitives[primitives.FinalTensor],
#         pset.primitives[primitives.Optimizer],
#         pset.primitives[primitives.Scheduler]
#     )
# )
# def layer_list(ind):
#     ll = []
#     for node in ind:
#         ll.append(node.name)
#     return ll

# for ind in pop[:1]:
#     print(layer_list(ind))
#     copy_ind = copy.deepcopy(ind)
#     mutated = toolbox.type_fixed_mut(copy_ind, pset)
#     mut = mutated[0]
#     print(layer_list(mut))