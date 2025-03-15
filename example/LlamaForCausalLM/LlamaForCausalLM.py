import dnndraw


RAND_COLOR=True

def add_detailed_decoder(dnn, decoder_name, input_name, mask_name):
    cd, d = dnn.create_graph(decoder_name, directed=True, has_border=True)

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L333
    dnn.add_node(
        in_nodes=[input_name],
        node_info={
            'name': f'{decoder_name}_input_layernorm',
            'operator': 'LlamaRMSNorm',
            'eps': 'config.rms_norm_eps',
            'shape': '[batch_size, seq_length, hidden_size]'
        },
        node_attr=dnn.node_colors[0],
        graph=d
    )

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L258
    attn_name = decoder_name + '_attn'
    c_attn, attn = dnn.create_graph(attn_name, directed=True, has_border=True)

    dnn.add_node(
        in_nodes=[f'{decoder_name}_input_layernorm', 'position_embeddings'],
        node_info={
            'name': f'{attn_name}_q_proj',
            'operator': 'Linear',
            'pos_encode': 'rotary_pos',
            'shape': '[batch_size, seq_length, hidden_size]',
        },
        node_attr=dnn.node_colors[1],
        graph=attn
    )

    dnn.add_node(
        in_nodes=[f'{decoder_name}_input_layernorm'],
        node_info={
            'name': f'{attn_name}_k_proj',
            'operator': 'Linear',
            'shape': '[batch_size, seq_length, hidden_size]',
        },
        node_attr=dnn.node_colors[1],
        graph=attn
    )

    dnn.add_node(
        in_nodes=[f'{decoder_name}_input_layernorm'],
        node_info={
            'name': f'{attn_name}_v_proj',
            'operator': 'Linear',
            'shape': '[batch_size, seq_length, hidden_size]',
        },
        node_attr=dnn.node_colors[1],
        graph=attn
    )

    dnn.add_node(
        in_nodes=[f'{attn_name}_q_proj', f'{attn_name}_k_proj'],
        node_info={
            'name': f'{attn_name}_matmul1',
            'operator': 'Matmul',
            'shape': '[batch_size, seq_length, seq_length]',
        },
        node_attr=dnn.node_colors[1],
        graph=attn
    )

    dnn.add_node(
        in_nodes=[f'{attn_name}_matmul1', mask_name],
        node_info={
            'name': f'{attn_name}_attn_mask',
            'type': 'Mask',
        },
        node_attr=dnn.node_colors[1],
        graph=attn
    )

    dnn.add_node(
        in_nodes=[f'{attn_name}_attn_mask'],
        node_info={
            'name': f'{attn_name}_softmax',
            'type': 'Softmax',
        },
        node_attr=dnn.node_colors[1],
        graph=attn
    )

    dnn.add_node(
        in_nodes=[f'{attn_name}_softmax', f'{attn_name}_v_proj'],
        node_info={
            'name': f'{attn_name}_matmul2',
            'type': 'Matmul',
            'shape': '[batch_size, seq_length, hidden_size]',
        },
        node_attr=dnn.node_colors[1],
        graph=attn
    )

    dnn.add_node(
        in_nodes=[f'{attn_name}_matmul2'],
        node_info={
            'name': f'{attn_name}_o_proj',
            'operator': 'Linear',
            'shape': '[batch_size, seq_length, hidden_size]',
        },
        node_attr=dnn.node_colors[1],
        graph=attn
    )

    dnn.merge_graph(attn, c_attn)
    dnn.merge_graph(c_attn, d)

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L347
    dnn.add_node(
        in_nodes=[f'{attn_name}_o_proj', input_name],
        node_info={
            'name': f'{decoder_name}_attn_residual',
            'operator': 'Add',
        },
        node_attr=dnn.node_colors[2],
        graph=d
    )

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L351
    dnn.add_node(
        in_nodes=[f'{decoder_name}_attn_residual'],
        node_info={
            'name': f'{decoder_name}_post_attention_layernorm',
            'operator': 'LlamaRMSNorm',
            'hidden_size': 'config.hidden_size',
            'eps': 'config.rms_norm_eps',
            'shape': '[batch_size, seq_length, hidden_size]'
        },
        node_attr=dnn.node_colors[0],
        graph=d
    )

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L178
    mlp_name = decoder_name + '_mlp'
    c_mlp, mlp = dnn.create_graph(mlp_name, directed=True, has_border=True)

    dnn.add_node(
        in_nodes=[f'{decoder_name}_post_attention_layernorm'],
        node_info={
            'name': f'{mlp_name}_gate_proj',
            'operator': 'Linear',
            'act': 'silu',
            'shape': '[batch_size, seq_length, intermediate_size]'
        },
        node_attr=dnn.node_colors[3],
        graph=mlp
    )

    dnn.add_node(
        in_nodes=[f'{decoder_name}_post_attention_layernorm'],
        node_info={
            'name': f'{mlp_name}_up_proj',
            'operator': 'Linear',
            'shape': '[batch_size, seq_length, intermediate_size]'
        },
        node_attr=dnn.node_colors[3],
        graph=mlp
    )

    dnn.add_node(
        in_nodes=[f'{mlp_name}_gate_proj', f'{mlp_name}_up_proj'],
        node_info={
            'name': f'{mlp_name}_dot',
            'operator': 'Dot',
            'shape': '[batch_size, seq_length, intermediate_size]'
        },
        node_attr=dnn.node_colors[3],
        graph=mlp
    )

    dnn.add_node(
        in_nodes=[f'{mlp_name}_dot'],
        node_info={
            'name': f'{mlp_name}_down_proj',
            'operator': 'Linear',
            'shape': '[batch_size, seq_length, hidden_size]'
        },
        node_attr=dnn.node_colors[3],
        graph=mlp
    )

    dnn.merge_graph(mlp, c_mlp)
    dnn.merge_graph(c_mlp, d)

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L353
    dnn.add_node(
        in_nodes=[f'{mlp_name}_down_proj', f'{decoder_name}_attn_residual'],
        node_info={
            'name': f'{decoder_name}_mlp_residual',
            'operator': 'Add',
        },
        node_attr=dnn.node_colors[2],
        graph=d
    )

    # Merge the decoder subgraph
    dnn.merge_graph(d, cd)
    dnn.merge_graph(cd)
    return f'{decoder_name}_mlp_residual'


# https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L308
def add_decoder(dnn, decoder_name, input_node, mask_name):
    # Create a subgraph for decoder layers
    cd, d = dnn.create_graph(decoder_name, directed=True, has_border=True)

    dnn.add_node(
        in_nodes=[input_node],
        node_info={
            'name': f'{decoder_name}_input_layernorm',
            'operator': 'LlamaRMSNorm',
        },
        node_attr=dnn.node_colors[0],
        graph=d
    )

    dnn.add_node(
        in_nodes=[f'{decoder_name}_input_layernorm', mask_name],
        node_info={
            'name': f'{decoder_name}_self_attn',
            'operator': 'LlamaAttention',
        },
        node_attr=dnn.node_colors[1],
        graph=d
    )

    dnn.add_node(
        in_nodes=[f'{decoder_name}_self_attn', input_node],
        node_info={
            'name': f'{decoder_name}_attn_residual',
            'operator': 'Add'
        },
        node_attr=dnn.node_colors[2],
        graph=d
    )

    dnn.add_node(
        in_nodes=[f'{decoder_name}_attn_residual'],
        node_info={
            'name': f'{decoder_name}_post_attention_layernorm',
            'operator': 'LlamaRMSNorm'
        },
        node_attr=dnn.node_colors[0],
        graph=d
    )

    dnn.add_node(
        in_nodes=[f'{decoder_name}_post_attention_layernorm'],
        node_info={
            'name': f'{decoder_name}_mlp',
            'operator': 'LlamaMLP'
        },
        node_attr=dnn.node_colors[3],
        graph=d
    )

    dnn.add_node(
        in_nodes=[f'{decoder_name}_mlp', f'{decoder_name}_attn_residual'],
        node_info={
            'name': f'{decoder_name}_mlp_residual',
            'operator': 'Add'
        },
        node_attr=dnn.node_colors[2],
        graph=d
    )

    # Merge the decoder subgraph
    dnn.merge_graph(d, cd)
    dnn.merge_graph(cd)
    return f'{decoder_name}_mlp_residual'


def LlamaForCausalLM(name='LlamaForCausalLM', label_align=False):
    dnn = dnndraw.graph(name=name, layout='TB', label_align=label_align)

    # Input nodes
    dnn.add_node(
        in_nodes=[],
        node_info={
            'name': 'input_ids',
            'operator': 'Data',
            'shape': '[batch_size, seq_length]'
        },
        rand_color=RAND_COLOR,
    )

    dnn.add_node(
        in_nodes=[],
        node_info={
            'name': 'position_ids',
            'operator': 'Data (Optional)',
            'shape': '[batch_size, seq_length]'
        },
        rand_color=RAND_COLOR,
    )

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L550
    dnn.add_node(
        in_nodes=['input_ids'],
        node_info={
            'name': 'embed_tokens',
            'operator': 'Embedding',
            'vocab_size': 'config.vocab_size',
            'embedding_dim': 'config.hidden_size',
            'shape': '[batch_size, seq_length, hidden_size]'
        },
        rand_color=RAND_COLOR,
    )

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L564
    dnn.add_node(
        in_nodes=['position_ids'],
        node_info={
            'name': 'causal_mask',
            'type': 'update_causal_mask',
            'shape': '[batch_size, seq_length, hidden_size]'
        },
        rand_color=RAND_COLOR,
    )

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L571C9-L571C29
    dnn.add_node(
        in_nodes=['position_ids', 'embed_tokens'],
        node_info={
            'name': 'position_embeddings',
            'type': 'LlamaRotaryEmbedding',
            'shape': '[batch_size, seq_length, hidden_size]'
        },
        rand_color=RAND_COLOR,
    )

    node_name = add_detailed_decoder(dnn, 'decoder_1', 'embed_tokens', 'causal_mask')

    node_name = add_decoder(dnn, 'decoder_2', node_name, 'causal_mask')

    node_name = add_decoder(dnn, 'decoder_3', node_name, 'causal_mask')

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L611
    dnn.add_node(
        in_nodes=[node_name],
        node_info={
            'name': 'norm',
            'operator': 'LlamaRMSNorm',
            'shape': '[batch_size, seq_length, hidden_size]'
        },
        rand_color=RAND_COLOR,
    )

    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llama/modeling_llama.py#L859
    dnn.add_node(
        in_nodes=['norm'],
        node_info={
            'name': 'lm_head',
            'operator': 'Linear',
            'bias': 'False',
            'shape': '[batch_size, seq_length, vocab_size]'
        },
        rand_color=RAND_COLOR,
    )

    # Output logits
    dnn.add_node(
        in_nodes=['lm_head'],
        node_info={
            'name': 'logits',
        },
        rand_color=RAND_COLOR,
    )
    return dnn


dnn = LlamaForCausalLM('LlamaForCausalLM', True)
print(dnn.source())
dnn.export(format='svg')  # format: png, svg, pdf, ...
