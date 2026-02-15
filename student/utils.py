import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy(logits,targets):
    if logits.dim() == 3:
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    # log probs
    N = logits.shape[0]
    indices = torch.arange(N, device=logits.device)
    target_log_probs = log_probs[indices, targets]
    
    # nll
    loss = -target_log_probs.mean()
    
    return loss

def data_loader(x,batch_size,context_length,device):
    if not isinstance(x , np.ndarray):
        x = np.array(x)
    
    seq_len = x.shape[0]

    # get starting position
    max_start_idx = seq_len - context_length - 1

    # sample
    start_idx_arr = np.random.randint(0, max_start_idx +1, size=batch_size)

    # initialize batch tensors
    input = np.zeros((batch_size, context_length), dtype=np.int64)
    target = np.zeros((batch_size, context_length), dtype=np.int64)

    #extract seq
    for i,start_idx in enumerate(start_idx_arr):
        # input
        input[i] = x[start_idx : start_idx + context_length]
        # target
        target[i] = x[start_idx + 1 : start_idx + context_length + 1]

    # convert to tensors
    input_tensor = torch.from_numpy(input).to(device)
    target_tensor = torch.from_numpy(target).to(device)

    return input_tensor, target_tensor



def save_checkpoint(model,optimizer,iteration, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src,model,optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration