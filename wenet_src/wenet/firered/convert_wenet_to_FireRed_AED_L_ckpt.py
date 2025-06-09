import argparse
import copy
import torch
import os
import shutil
import glob
import re
import yaml


def convert_to_firered_state_dict(wenet_state_dict, lora_alpha=None):
    firered_state_dict = {}
    unused = []
    lora_weights = {}  # Dictionary to store LoRA weights for later merging
    
    print("===================== start Reverse CKPT Conversion =========================")
    
    # First pass: collect all weights and identify LoRA weights
    for name in wenet_state_dict.keys():
        original_name = copy.deepcopy(name)
        
        # Handle encoder embed
        if 'encoder.embed' in name:
            name = name.replace("encoder.embed", "encoder.input_preprocessor")
            name = name.replace('encoder.input_preprocessor.out.0', 'encoder.input_preprocessor.out')
        
        # Handle decoder embed
        name = name.replace("decoder.embed.0", "decoder.token_embedding")
        
        # Layer stacks
        name = name.replace("encoder.encoders", "encoder.layer_stack")
        name = name.replace("decoder.decoders", "decoder.layer_stack")
        
        if 'embed' not in name:
            name = name.replace(".conv_module.", ".conv.")
            name = name.replace(".norm.", ".batch_norm.")
            
            
        # decoder
        if "decoder" in name:
            name = name.replace(".src_attn.linear_q", ".cross_attn.w_qs")
            name = name.replace(".src_attn.linear_k", ".cross_attn.w_ks")
            name = name.replace(".src_attn.linear_v", ".cross_attn.w_vs")
            name = name.replace(".src_attn.linear_out", ".cross_attn.fc")
            name = name.replace(".self_attn.linear_q", ".self_attn.w_qs")
            name = name.replace(".self_attn.linear_k", ".self_attn.w_ks")
            name = name.replace(".self_attn.linear_v", ".self_attn.w_vs")
            name = name.replace(".self_attn.linear_out", ".self_attn.fc")
            name = name.replace(".feed_forward.", ".mlp.")
            name = name.replace(".norm1.", ".self_attn_norm.")
            name = name.replace(".norm2.", ".cross_attn_norm.")
            name = name.replace(".norm3.", ".mlp_norm.")

        # encoder
        if "encoder" in name:
            name = name.replace(".self_attn.linear_q", ".mhsa.w_qs")
            name = name.replace(".self_attn.linear_k", ".mhsa.w_ks")
            name = name.replace(".self_attn.linear_v", ".mhsa.w_vs")
            name = name.replace(".self_attn.linear_out", ".mhsa.fc")
            name = name.replace(".self_attn.pos_bias_u", ".mhsa.pos_bias_u")
            name = name.replace(".self_attn.pos_bias_v", ".mhsa.pos_bias_v")
            name = name.replace(".self_attn.linear_pos", ".mhsa.linear_pos")
            name = name.replace(".norm_ff_macaron.", ".ffn1.net.0.")
            name = name.replace(".self_attn.layer_norm_q.", ".mhsa.layer_norm_q.",)
            name = name.replace(".self_attn.layer_norm_k.", ".mhsa.layer_norm_k.")
            name = name.replace(".self_attn.layer_norm_v.", ".mhsa.layer_norm_v.")
            name = name.replace(".norm_conv.", ".conv.pre_layer_norm.")
            name = name.replace(".norm_ff", ".ffn2.net.0")
            name = name.replace(".norm_final.", ".layer_norm.")
            name = name.replace(".feed_forward_macaron.w_1", ".ffn1.net.1")
            name = name.replace(".feed_forward_macaron.w_2", ".ffn1.net.4")
            name = name.replace(".feed_forward.w_1", ".ffn2.net.1")
            name = name.replace(".feed_forward.w_2", ".ffn2.net.4")


        # Extra replacements for decoder
        if "decoder" in name:
            name = name.replace("norm2", "cross_attn_ln")
            name = name.replace("norm3", "mlp_ln")
        else:
            name = name.replace("norm2", "mlp_ln")

        # Handle specific weights
        if original_name == "decoder.embed.0.weight":
            name = "decoder.tgt_word_emb.weight"
        if original_name == 'decoder.embed.1.pe':
            name = "decoder.positional_encoding.pe"
        if original_name == "decoder.output_layer.weight":
            name = "decoder.tgt_word_prj.weight"
        if original_name == 'encoder.embed.pos_enc.pe':
            name = "encoder.positional_encoding.pe"

        if 'decoder.after_norm.' in original_name:
            name = name.replace('decoder.after_norm', 'decoder.layer_norm_out')
            
        print("name  {} ==> {}".format(original_name, name))
        print("type  {} ==> {}".format(wenet_state_dict[original_name].dtype, 
                                      wenet_state_dict[original_name].dtype))
        print("shape {}\n".format(wenet_state_dict[original_name].shape))
        
        if (original_name == name):
            unused.append(name)
        elif 'lora_A' in name or 'lora_B' in name:
            # Store for later processing
            lora_weights[name] = wenet_state_dict[original_name]
        else:
            firered_state_dict[name] = wenet_state_dict[original_name].float()
    
    # Second pass: process and merge LoRA weights
    print("===================== Processing LoRA Weights =========================")
    pattern = re.compile(r'(.*?)\.lora_(A|B)$')
    lora_pairs = {}
    
    # Group LoRA A and B pairs
    for lora_name, lora_weight in lora_weights.items():
        match = pattern.match(lora_name)
        if match:
            base_name = match.group(1)
            lora_type = match.group(2)
            
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            
            lora_pairs[base_name][lora_type] = lora_weight
    
    # Merge LoRA weights with base weights
    for base_name, lora_pair in lora_pairs.items():
        assert lora_alpha is not None
        if 'A' in lora_pair and 'B' in lora_pair:
            # Check if the base weight exists in converted state dict
            if base_name + '.weight' in firered_state_dict:
                # Calculate LoRA update: base_weight + lora_A @ lora_B
                lora_A = lora_pair['A'].float()
                lora_B = lora_pair['B'].float()
                base_weight = firered_state_dict[base_name + '.weight']

                # Merge LoRA weights with base weight
                r = lora_A.shape[0]
                scaling = lora_alpha / r
                lora_update = lora_B @ lora_A * scaling
                updated_weight = base_weight + lora_update

                # Update the weight in the state dict
                firered_state_dict[base_name + '.weight'] = updated_weight

                del lora_weights[base_name + '.lora_A']
                del lora_weights[base_name + '.lora_B']
                print(f'Merge LoRA into {base_name}.weight\n')
            else:
                print(f"Warning: Could not find base weight for LoRA pair {base_name}\n")

    for name in unused:
        print("NOTE!!! drop {}".format(name))
    for name in lora_weights.keys():
        print("NOTE!!! Un-merged {}".format(name))
    print("===================== End Reverse CKPT Conversion =========================\n")
    return firered_state_dict


def get_args():
    parser = argparse.ArgumentParser(description='Convert WeNet model to FireRed format')
    parser.add_argument('--wenet_config_path', required=True, 
                        help='Path to the WeNet config.yaml file')
    parser.add_argument('--wenet_pt_path', required=True, 
                        help='Path to the WeNet model.pt file')
    parser.add_argument('--original_fireredaed_dir', required=True,
                        help='Path to original fireredasr dir')
    parser.add_argument('--output_dir', required=True,
                        help='Output path for FireRed model')
    return parser.parse_args()


def copy_files(src_dir, dst_dir):
    """Copy all files except model.pth.tar from src_dir to dst_dir"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Copy all files and directories except model.pth.tar
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        
        if item != 'model.pth.tar':
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item, dst_item)


def main():
    args = get_args()
    
    # 1. Copy all files except model.pth.tar from original_fireredaed_dir to output_dir
    print(f"Copying files from {args.original_fireredaed_dir} to {args.output_dir}...")
    copy_files(args.original_fireredaed_dir, args.output_dir)
    
    # 2. Load Wenet Config to get lora config
    with open(args.wenet_config_path, 'r') as file:
        config = yaml.safe_load(file)
    if config.get('lora_conf', None) is not None:
        lora_alpha = config['lora_conf'].get('lora_alpha', None)
    else:
        lora_alpha = None

    # 3. Load original FireRed model to get args
    print(f"Loading original FireRed model from {os.path.join(args.original_fireredaed_dir, 'model.pth.tar')}...")
    original_checkpoint = torch.load(os.path.join(args.original_fireredaed_dir, 'model.pth.tar'), 
                                    map_location="cpu")
    
    # 3. Load WeNet model state dict
    print(f"Loading WeNet model from {args.wenet_pt_path}...")
    wenet_state_dict = torch.load(args.wenet_pt_path, map_location="cpu")
    
    # 4. Convert state dict to FireRed format with LoRA weights merged
    firered_state_dict = convert_to_firered_state_dict(wenet_state_dict, lora_alpha=lora_alpha)
    
    # 5. Create FireRed checkpoint with original args and new state dict
    firered_checkpoint = {
        "model_state_dict": firered_state_dict,
        "args": original_checkpoint["args"],
        # Copy other keys from original checkpoint if needed
    }
    
    # If there were additional keys in the original checkpoint, copy them
    for key in original_checkpoint:
        if key != "model_state_dict" and key != "args" and key not in firered_checkpoint:
            firered_checkpoint[key] = original_checkpoint[key]
    
    # 6. Save FireRed checkpoint
    output_path = os.path.join(args.output_dir, "model.pth.tar")
    print(f"Saving FireRed model to {output_path}...")
    torch.save(firered_checkpoint, output_path)
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
