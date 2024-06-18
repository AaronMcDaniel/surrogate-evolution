import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Custom summary function for detection models
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
