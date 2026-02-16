import torch
from student.transformer import TransformerLM
from student.tokenizer import Tokenizer
from student.decoder import generate_text


vocab_size = 10000
context_length = 128
d_model = 256
num_layers = 6
num_heads = 8
d_ff = 1024
rope_theta = 10000.0

device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = Tokenizer.from_files(
    vocab_filepath="tinystories_bpe_vocab.pkl",       
    merges_filepath="tinystories_bpe_merges.pkl",     
    special_tokens=["<|endoftext|>"]   
)

# Build model
model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta,
    device=device,
    dtype=torch.float32
)

# Load checkpoint
checkpoint = torch.load("./checkpoints/best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])  # adjust key if needed
model = model.to(device)


eos_token_id = tokenizer.byte_to_token_id.get("<|endoftext|>".encode("utf-8"), None)
tokenizer.eos_token_id = eos_token_id

# Generate with different params
prompt = "Once upon a time"

for temp, top_p in [(0.7, 0.85), (0.8, 0.9), (0.5, 0.8)]:
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=256,
        temprature=temp,
        top_p=top_p,
        device=device
    )
    print(f"\n--- temp={temp}, top_p={top_p} ---")
    print(text)
    print()