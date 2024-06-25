import re
import torch
import primitives


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Custom summary function for pytorch detection models
def detection_model_summary(model, input_size):
    print("----------------------------------------------------------------------------")
    print("Layer (type: depth-idx)                   Output Shape              Param #")
    print("============================================================================")
    dummy_input = torch.randn(1, *input_size).to(device)
    hooks = []
    module_idx = 0

    def register_hook(module):
        def hook(module, input, output):
            nonlocal module_idx
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            m_key = f"{class_name}-{module_idx + 1}"
            module_idx += 1

            if isinstance(input[0], torch.Tensor):
                input_shape = list(input[0].size())
            elif isinstance(input[0], (list, tuple)):
                input_shape = [list(i.size()) for i in input[0] if isinstance(i, torch.Tensor)]
            else:
                input_shape = 'multiple'

            if isinstance(output, (list, tuple)):
                output_shape = [list(o.size()) for o in output if isinstance(o, torch.Tensor)]
            else:
                output_shape = list(output.size()) if isinstance(output, torch.Tensor) else 'multiple'

            param_num = sum(p.numel() for p in module.parameters())
            print(f"{m_key:<40} {str(output_shape):<30} {param_num}")

        if not isinstance(module, torch.nn.Sequential) and \
           not isinstance(module, torch.nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    for h in hooks:
        h.remove()
    print("============================================================================")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print("----------------------------------------------------------------------------")


# Custom summary function for tree-encoded genome
def tree_genome_summary(genome, num_loss_components):
    print('----------Genome Architecture----------')
    # parse tree decoding into layer list
    layer_list = []
    expr = re.split(r'([(),])',genome)
    remove = [',', '']
    expr = [x for x in expr if x not in remove]
    stack = []
    for element in expr:
        if element != ')':
            stack.append(element)
        else:
            arguments = []
            while stack[-1] != '(':
                arguments.insert(0, stack.pop())
            stack.pop()
            function = stack.pop()
            try:
                stack.append(str(eval(f'primitives.{function}({','.join(arguments)})')))
            except: # this is where we add the layers
                layer_info = [function]+[__parse_arg(x) for x in arguments]
                layer_list.append(layer_info)
    for layer in layer_list:
        print(layer[0])
        layer_args = layer[1:]
        if layer[0] == "Detection_Head":
            print('----------Hyperparameters----------')
            loss_weights = layer_args[1:]
            if len(loss_weights) > num_loss_components:
                loss_weights = loss_weights[:num_loss_components]
            weights_sum = sum(loss_weights)
            loss_weights = [x/weights_sum for x in loss_weights]
            weight_tensor = torch.tensor(loss_weights, dtype=torch.float32)
            tensor = torch.zeros(num_loss_components, dtype=torch.float32)
            tensor[:len(weight_tensor)] = weight_tensor
            out_dict = {}
            optimizer_dict = eval(layer_args[0])
            out_dict['optimizer'] = optimizer_dict['optimizer']
            for k, v in optimizer_dict.items():
                if k not in ['optimizer', 'eta_lower', 'eta_upper', 'step_lower', 'step_upper']:
                    out_dict[f'optimizer_{k}'] = v
            if optimizer_dict['optimizer'] == 'Rprop':
                out_dict[f'optimizer_etas'] = (optimizer_dict['eta_lower'], optimizer_dict['eta_upper'])
                out_dict[f'optimizer_step_sizes'] = (optimizer_dict['step_lower'], optimizer_dict['step_upper'])
            out_dict['loss_weights'] = tensor
            for k, v in out_dict.items():
                print(f'{k}: {v}')
    print('=======================================')


def __parse_arg(s):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s