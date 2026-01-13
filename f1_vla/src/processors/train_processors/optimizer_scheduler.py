from transformers.trainer import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def create_optimizer(opt_model, args):
    """
    Setup the optimizer with different learning rates for different parts of the model.
    """
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    params_dict = dict(opt_model.named_parameters())
    param_to_name = {id(p): n for n, p in params_dict.items()}

    # Group 0: vision encoder parameters
    vision_encoder_parameters = [
        name for name in params_dict.keys()
        if any(x in name for x in [
            "paligemma_with_expert.paligemma.vision_tower",
            "paligemma_with_expert.paligemma.multi_modal_projector"
        ])
    ]

    # Group 1: paligemma parameters (excluding vision encoder)
    paligemma_parameters = [
        name for name in params_dict.keys()
        if "paligemma_with_expert.paligemma" in name and name not in vision_encoder_parameters
    ]

    # Group 2: world model expert parameters and world model related parameters
    gen_expert_parameters = [name for name in params_dict.keys() if "paligemma_with_expert.gemma_wm_expert" in name]
    # Memory module parameters (KV memory bank + GRU update) - now separate group with independent lr
    # These are critical for convergence when use_memory=True; if excluded, memory stays randomly initialized.
    memory_parameters = [
        name for name in params_dict.keys()
        if any(x in name for x in ["memory_bank", "memory_manager", "memory_token", "memory_info_proj", "memory_gru", "init_memory"])
    ]
    # World model related parameters (excluding memory)
    gen_parameters = [
        name for name in params_dict.keys()
        if any(x in name for x in [
            "temporal_conv", "wm_embeddings", "wm_hist_pos_embs", 
            "wm_sos_token_embs", "wm_cond_pos_embs", "wm_output_position_embedding",
            "lvl_embed", "wm_out_layer_norm", "wm_out_proj"
        ]) and name not in memory_parameters  # Exclude memory params
    ]

    # Group 3: gemma expert parameters and action-related parameters
    # Note: module is named gemma_experts (ModuleDict), not gemma_expert
    act_expert_parameters = [name for name in params_dict.keys() if "paligemma_with_expert.gemma_experts" in name]
    action_parameters = [
        name for name in params_dict.keys()
        if any(x in name for x in [
            "state_proj", "action_in_proj", "action_out_proj", 
            "action_time_mlp_in", "action_time_mlp_out"
        ])
    ]

    optimizer_grouped_parameters = []

    # Group 0: vision encoder parameters (if vision_encoder_lr is provided)
    if len(vision_encoder_parameters) > 0:
        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n in decay_parameters and n in vision_encoder_parameters and p.requires_grad
                ],
                "weight_decay": args.weight_decay,
                "lr": args.vision_encoder_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n not in decay_parameters and n in vision_encoder_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": args.vision_encoder_lr,
            },
        ])

    # Group 1: paligemma parameters (excluding vision encoder if separate lr is provided)
    optimizer_grouped_parameters.extend([
        {
            "params": [
                p for n, p in opt_model.named_parameters() 
                if n in decay_parameters and n in paligemma_parameters and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
            "lr": args.und_expert_lr,
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() 
                if n not in decay_parameters and n in paligemma_parameters and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": args.und_expert_lr,
        },
    ])

    # Group 2: world model expert and world model parameters (excluding memory)
    if len(gen_parameters) > 0 or len(gen_expert_parameters) > 0:
        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n in decay_parameters and (n in gen_expert_parameters or n in gen_parameters) and n not in memory_parameters and p.requires_grad
                ],
                "weight_decay": args.weight_decay,
                "lr": args.gen_expert_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n not in decay_parameters and (n in gen_expert_parameters or n in gen_parameters) and n not in memory_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": args.gen_expert_lr,
            },
        ])

    # Group 2.5: Memory module parameters with independent learning rate
    # Memory needs higher lr since it's initialized from scratch (not pretrained)
    memory_lr = getattr(args, 'memory_lr', args.gen_expert_lr)  # Fallback to gen_expert_lr if not set
    if len(memory_parameters) > 0:
        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n in decay_parameters and n in memory_parameters and p.requires_grad
                ],
                "weight_decay": args.weight_decay,
                "lr": memory_lr,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if n not in decay_parameters and n in memory_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": memory_lr,
            },
        ])

    # Group 3: gemma expert and action parameters
    optimizer_grouped_parameters.extend([
        {
            "params": [
                p for n, p in opt_model.named_parameters() 
                if n in decay_parameters and (n in act_expert_parameters or n in action_parameters) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
            "lr": args.act_expert_lr,
        },
        {
            "params": [
                p for n, p in opt_model.named_parameters() 
                if n not in decay_parameters and (n in act_expert_parameters or n in action_parameters) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": args.act_expert_lr,
        },
    ])

    # Track names that have been grouped
    def _collect_names(param_list):
        return {param_to_name[id(p)] for p in param_list if id(p) in param_to_name}

    grouped_names = set()

    # Collect names after initial groups to add a fallback group for any remaining trainable params
    for group in optimizer_grouped_parameters:
        grouped_names.update(_collect_names(group["params"]))

    remaining_names = [
        n for n, p in params_dict.items()
        if p.requires_grad and n not in grouped_names
    ]
    if remaining_names:
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in params_dict.items() if n in remaining_names and n in decay_parameters and p.requires_grad],
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate,
            },
            {
                "params": [p for n, p in params_dict.items() if n in remaining_names and n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
                "lr": args.learning_rate,
            },
        ])

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
