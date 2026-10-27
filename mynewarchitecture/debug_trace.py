"""Debug: trace exactly what happens during train_step."""
from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(state_dim=64, input_dim=64, max_vocab=512)
m.add_training_pair("hello", "world")

# Step 1: encode
tok_id, x = m.symbols.encode("hello")
tgt_id, _ = m.symbols.encode("world")
print(f"hello id={tok_id}, world id={tgt_id}")
print(f"x nonzero: {np.sum(x != 0)} / {len(x)}")

# Step 2: W_in @ x
accum_in = m.W_in.W.astype(np.int32) @ x.astype(np.int32)
s_in = tproj(accum_in, theta=1)
print(f"W_in accum range: [{accum_in.min()}, {accum_in.max()}]")
print(f"s_in nonzero: {np.sum(s_in != 0)} / {len(s_in)}")

# Step 3: full forward  
info = m.forward_step("hello")
print(f"State norm after forward: {info['state_norm']}")
print(f"State: {m.state}")
print(f"Predicted: {info['pred_id']} ({info['pred_token']})")

# Step 4: W_out scores
scores = m.W_out.W.astype(np.int32) @ m.state.astype(np.int32)
print(f"W_out score for target (world={tgt_id}): {scores[tgt_id]}")
print(f"W_out score for pred ({info['pred_id']}): {scores[info['pred_id']]}")
print(f"W_out scores range: [{scores.min()}, {scores.max()}]")
print(f"Unique scores: {len(np.unique(scores))}")

# Step 5: train 3 steps and check
m.reset_state()
for i in range(3):
    info = m.train_step("hello", "world")
    scores = m.W_out.W.astype(np.int32) @ m.state.astype(np.int32)
    print(f"Train {i}: correct={info['correct']}, target_score={scores[tgt_id]}, pred_score={scores[info['pred_id']]}, pred={info['pred_token']}")
