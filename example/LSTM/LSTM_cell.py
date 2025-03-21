import dnndraw

def Add_LSTM_Cell(dnn, name, node_in, node_h, node_c):
    """Add a LSTM cell to the graph"""
    # Gates
    forget_gate = f"{name}_Ft"
    input_gate = f"{name}_It"
    output_gate = f"{name}_Ot"
    cell_update = f"{name}_ct"
    cell_state = f"{name}_Ct"
    hidden_state = f"{name}_Ht"

    # Add forget gate
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': forget_gate,
            'gate': 'forget_gate Ft',
            'formula': 'Matmul(W_if,It)+Matmul(W_hf,H[t-1])+Bf',
            'shape W_if': '[hidden_size, input_size]',
            'shape W_hf': '[hidden_size, Proj_size]',
            'shape out': ['batch', 'hidden_size']
        }
    )

    # Add input gate
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': input_gate,
            'gate': 'input_gate It',
            'formula': 'Matmul(W_ii,It)+Matmul(W_hi,H[t-1])+Bi',
            'shape W_ii': '[hidden_size, input_size]',
            'shape W_hi': '[hidden_size, Proj_size]',
            'shape out': ['batch', 'hidden_size']
        }
    )

    # Add output gate
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': output_gate,
            'gate': 'output_gate Ot',
            'formula': 'Matmul(W_io,It)+Matmul(W_ho,H[t-1])+Bo',
            'shape W_io': '[hidden_size, input_size]',
            'shape W_ho': '[hidden_size, Proj_size]',
            'shape out': ['batch', 'hidden_size']
        }
    )

    # Add cell update
    dnn.add_node(
        in_nodes=[node_in, node_h],
        node_info={
            'name': cell_update,
            'type': 'cell_update ct',
            'formula': 'tanh(Matmul(W_ic,It)+Matmul(W_hc,H[t-1])+Bc)',
            'shape W_io': '[hidden_size, input_size]',
            'shape W_ho': '[hidden_size, Proj_size]',
            'shape out': ['batch', 'hidden_size']
        }
    )

    # Add cell state
    dnn.add_node(
        in_nodes=[node_c, forget_gate, input_gate, cell_update],
        node_info={
            'name': cell_state,
            'type': 'cell_state Ct',
            'formula': 'EleWiseMul(Ft,C[t-1])+EleWiseMul(It,ct)',
            'shape out': ['batch', 'hidden_size']
        }
    )

    # Add hidden state
    dnn.add_node(
        in_nodes=[cell_state, output_gate],
        node_info={
            'name': hidden_state,
            'type': 'hidden_state Ht',
            'formula': 'W_hr*EleWiseMul(Ot, tanh(Ct))',
            'shape W_hr': '[hidden_size, Proj_size]',
            'shape out': ['batch', 'Proj_size'],
            'note': 'W_hr only used for Proj_size!=hidden_size'
        }
    )
    return hidden_state, cell_state

# https://arxiv.org/abs/1402.1128
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm
# https://d2l.ai/chapter_recurrent-modern/lstm.html
def LSTM(name, seq_len=1, num_layers=1):
    # Create main graph
    dnn = dnndraw.graph(name=name, layout="BT")

    it_cache = [None for _ in range(seq_len)]
    ht_cache = [None for _ in range(num_layers)]
    ct_cache = [None for _ in range(num_layers)]

    # initial input data
    for i in range(seq_len):
        it = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'in_t{i}',
                'type': 'Input_Data',
                'shape': ['batch', 'input_size']
            },
            node_attr=dnn.node_colors[0]
        )
        it_cache[i] = it
    # Add initial states
    for i in range(num_layers):
        ht = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'h_l{i}',
                'type': 'Initial_Hidden',
                'shape': ['batch', 'proj_size']
            },
            node_attr=dnn.node_colors[1]
        )
        ct = dnn.add_node(
            in_nodes=[],
            node_info={
                'name': f'c_l{i}',
                'type': 'Initial_Cell',
                'shape': ['batch', 'hidden_size']
            },
            node_attr=dnn.node_colors[2]
        )
        ht_cache[i] = ht
        ct_cache[i] = ct
    # Add LSTM cell
    for i in range(num_layers):
        for j in range(seq_len):
            ht, ct = Add_LSTM_Cell(
                dnn,
                name=f'Cell_l{i}_t{j}',
                node_in=it_cache[j],
                node_h=ht_cache[i],
                node_c=ct_cache[i],
            )
            it_cache[j] = ht
            ht_cache[i] = ht
            ct_cache[i] = ct
    return dnn

if __name__ == '__main__':
    dnn = LSTM("LSTM CELL", seq_len=1, num_layers=1)
    print(dnn.source())
    dnn.export(format='svg')
