import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from thseq.data.vocabulary import Vocabulary
from thseq.models.transformer.model import Transformer

mapping = {
    'embeddings': {
        "encoder.embedding": "encoder.embed_tokens",
        "decoder.embedding": "decoder.embed_tokens",
    },
    'final_layer_norms': {
        "encoder.ln": "encoder.layer_norm",
        "decoder.ln": "decoder.layer_norm",
    },
    'logit': {
        "decoder.logit": "decoder.output_projection",
        # "decoder.logit.weight": "decoder.embed_out",
    },
    'encoder_layer': {
        "encoder.layers.%d.ln0": "encoder.layers.%d.self_attn_layer_norm",
        "encoder.layers.%d.self_attention.linear_q": "encoder.layers.%d.self_attn.q_proj",
        "encoder.layers.%d.self_attention.linear_k": "encoder.layers.%d.self_attn.k_proj",
        "encoder.layers.%d.self_attention.linear_v": "encoder.layers.%d.self_attn.v_proj",
        "encoder.layers.%d.self_attention.linear_o": "encoder.layers.%d.self_attn.out_proj",
        "encoder.layers.%d.ln1": "encoder.layers.%d.final_layer_norm",
        "encoder.layers.%d.ffn.linear_i": "encoder.layers.%d.fc1",
        "encoder.layers.%d.ffn.linear_o": "encoder.layers.%d.fc2",
    },
    'decoder_layer': {
        "decoder.layers.%d.ln0": "decoder.layers.%d.self_attn_layer_norm",
        "decoder.layers.%d.masked_self_attention.linear_q": "decoder.layers.%d.self_attn.q_proj",
        "decoder.layers.%d.masked_self_attention.linear_k": "decoder.layers.%d.self_attn.k_proj",
        "decoder.layers.%d.masked_self_attention.linear_v": "decoder.layers.%d.self_attn.v_proj",
        "decoder.layers.%d.masked_self_attention.linear_o": "decoder.layers.%d.self_attn.out_proj",
        "decoder.layers.%d.ln1": "decoder.layers.%d.encoder_attn_layer_norm",
        "decoder.layers.%d.encoder_decoder_attention.linear_q": "decoder.layers.%d.encoder_attn.q_proj",
        "decoder.layers.%d.encoder_decoder_attention.linear_k": "decoder.layers.%d.encoder_attn.k_proj",
        "decoder.layers.%d.encoder_decoder_attention.linear_v": "decoder.layers.%d.encoder_attn.v_proj",
        "decoder.layers.%d.encoder_decoder_attention.linear_o": "decoder.layers.%d.encoder_attn.out_proj",
        "decoder.layers.%d.ln2": "decoder.layers.%d.final_layer_norm",
        "decoder.layers.%d.ffn.linear_i": "decoder.layers.%d.fc1",
        "decoder.layers.%d.ffn.linear_o": "decoder.layers.%d.fc2"
    }
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('src_dict')
    parser.add_argument('tgt_dict')
    parser.add_argument('output_dir')
    return parser.parse_args()


def convert_module(module_mapping, fairseq_model, thseq_model, layer_index=None):
    for name in module_mapping:
        src_prefix = name if layer_index is None else name % layer_index
        tgt_prefix = module_mapping[name]
        tgt_prefix = tgt_prefix if layer_index is None else tgt_prefix % layer_index
        for suffix in ['', '.weight', '.bias']:
            src_name = src_prefix + suffix
            tgt_name = tgt_prefix + suffix
            if tgt_name in fairseq_model:
                thseq_model[src_name] = fairseq_model[tgt_name]


def convert_args(f_args):
    t_args = {}
    share_all = f_args.share_all_embeddings
    share_dec_in_out = share_all or f_args.share_decoder_input_output_embed
    share_enc_dec_in = f_args.share_all_embeddings

    t_args["model"] = 'transformer'
    t_args["input_size"] = f_args.encoder_embed_dim
    t_args["hidden_size"] = f_args.encoder_embed_dim
    t_args["ffn_hidden_size"] = f_args.encoder_ffn_embed_dim
    t_args["residual_dropout"] = f_args.dropout
    t_args["ffn_dropout"] = f_args.activation_dropout
    t_args["attention_dropout"] = f_args.attention_dropout
    t_args["num_encoder_layers"] = f_args.encoder_layers
    t_args["num_decoder_layers"] = f_args.decoder_layers
    t_args["num_heads"] = f_args.encoder_attention_heads
    t_args["share_all_embedding"] = share_all
    t_args["share_decoder_input_output_embedding"] = share_dec_in_out
    t_args["share_encoder_decoder_input_embedding"] = share_enc_dec_in
    t_args["force_share"] = False
    t_args["encoder_post_norm"] = int(not f_args.encoder_normalize_before)
    t_args["decoder_post_norm"] = int(not f_args.decoder_normalize_before)

    assert f_args.encoder_embed_dim == f_args.decoder_embed_dim
    assert f_args.encoder_ffn_embed_dim == f_args.decoder_ffn_embed_dim
    assert f_args.encoder_attention_heads == f_args.decoder_attention_heads

    return argparse.Namespace(**t_args)


def main(args):
    fairseq_checkpoint = torch.load(args.checkpoint, 'cpu')
    # 1. convert args
    t_args = convert_args(fairseq_checkpoint['args'])
    # 2. build dictionaries
    src_dict = Vocabulary(args.src_dict)
    tgt_dict = Vocabulary(args.tgt_dict)
    # 3. convert params
    f_params = fairseq_checkpoint['model']
    t_params = {}
    for module in ['embeddings', 'final_layer_norms', 'logit']:
        convert_module(mapping[module], f_params, t_params)
    for i in range(t_args.num_encoder_layers):
        convert_module(mapping['encoder_layer'], f_params, t_params, i)

    for i in range(t_args.num_decoder_layers):
        convert_module(mapping['decoder_layer'], f_params, t_params, i)

    try:
        model = Transformer(t_args, [src_dict, tgt_dict])
        model_state = model.state_dict()
        missing_keys = []
        for k in model_state:
            if k not in t_params:
                t_params[k] = model_state[k]
                missing_keys.append(k)
                t_params[k][:] = 0

        # work-around to deal with old-fashioned fairseq logit
        if t_args.share_encoder_decoder_input_embedding and 'decoder.logit.weight' in missing_keys:
            t_params['decoder.logit.weight'] = t_params['decoder.embedding.weight']
            del missing_keys[missing_keys.index('decoder.logit.weight')]

        print(f'Missing keys after conversion: {missing_keys}. Setting all missing values to zeros.')
        model.load_state_dict(t_params, False)
    except Exception as e:
        raise Exception('conversion failed, check mapping rules carefully') from e

    state_dict = {
        'args': t_args,
        'vocabularies': [src_dict, tgt_dict],
        'model': t_params
    }

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)

    config_file = output_dir / 'config.json'

    with config_file.open('w', encoding='utf-8') as w:
        w.write(json.dumps(t_args.__dict__, indent=4, sort_keys=True))

    torch.save(state_dict, output_dir / 'checkpoint.pt')
    print('Success!')


if __name__ == '__main__':
    args = parse_args()
    main(args)
