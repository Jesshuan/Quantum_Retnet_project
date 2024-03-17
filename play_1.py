
import torch

import pdb

from retnet.configuration_retnet import RetNetConfig
from retnet.modeling_retnet import RetNetModel, RetNetForCausalLM


torch.manual_seed(0)
config = RetNetConfig(decoder_layers=2,
                      decoder_embed_dim=8,
                      decoder_value_embed_dim=8,
                      decoder_retention_heads=4,
                      decoder_ffn_embed_dim=16)

model = RetNetModel(config)
model.eval()

device = 'cpu'  # cuda, cpu, mps for m1 mac
model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

input_ids = torch.LongTensor([[1, 2, 3, 4, 0]]).to(device)

"""
parallel_outputs = model(input_ids, forward_impl='parallel', use_cache=True)
parallel_state = parallel_outputs.last_hidden_state
parallel_cache = parallel_outputs.past_key_values

print(parallel_outputs)

"""
past_kv = None
rnn_state = []
for i in range(input_ids.shape[1]):
    print("--------------")
    print(f"STEP : {i}")
    print("--------------")
    print(f"Send to model : {input_ids[:, :i+1]}")
    print(f"Past key_value : {past_kv}")
    rnn_out = model(input_ids[:, :i+1], forward_impl='recurrent', past_key_values=past_kv, use_cache=True)
    rnn_state.append(rnn_out.last_hidden_state)
    past_kv = rnn_out.past_key_values
rnn_state = torch.cat(rnn_state, dim=1)
rnn_cache = rnn_out.past_key_values

"""
chunk_outputs = model(input_ids, forward_impl='chunkwise', use_cache=True, recurrent_chunk_size=2)
chunk_state = chunk_outputs.last_hidden_state
chunk_cache = chunk_outputs.past_key_values"""
