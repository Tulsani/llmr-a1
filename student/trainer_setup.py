import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
# custom
from student.transformer import TransformerLM
from student.custom_optimizer import AdamW , gradient_clipping, get_lr_cosine_schedule
from student.utils import data_loader , save_checkpoint, load_checkpoint ,cross_entropy
import time



def train(
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    training_data_path,
    val_data_path,
    device,
    rope_theta=10000,

    # training params
    batch_size=32,
    max_epochs=10000,
    learning_rate=1e-3,
    min_learning_rate=1e-5,
    warmup_iterations=1000,
    cosine_iterations=8000,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.999,
    grad_clip_norm=1.0,

    # path primitives
    checkpoint_dir="./checkpoints",
    save_every=1000,

    log_every=10,
    eval_every=100,
    eval_iterations=20,
    restart_epoch_path=None
):
    seed=42
    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)

    #setup checkoint dir
    checkpoint_dir= Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)

    #Starting training loop
    
    print(f"\nLoading training data {training_data_path}")
    training_data = np.load(training_data_path,mmap_mode='r')
    print(f"\loaded training data: {len(training_data)}")

    #load validation data
    val_data = None
    if val_data_path is not None:
        print(f"\nLoading validation data {val_data_path}")
        val_data = np.load(val_data_path,mmap_mode='r')
        print(f"\nLoaded val data: {len(val_data)}")

    #setup the model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads = num_heads,
        d_ff=d_ff,
        rope_theta = rope_theta,
        device=device,
        dtype=torch.float32
    )
    # push model to devide
    model = model.to(device)

    #param count
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model parameters count {num_params}")

    # setting up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1,beta2),
        eps = 1e-8,
        weight_decay=weight_decay
    )

    # start epoch
    start_epoch = 0

    if restart_epoch_path is not None:
        print(f"\n restarting from checkpoint {restart_epoch_path}")
        start_epoch = load_checkpoint(restart_epoch_path,model,optimizer)
        print(f"Restarted from {epoch}")

    best_val_loss = float('inf')
    train_losses = []

    print("\n"+ "="*80)
    print("Startin training")
    print("="*80)

    #setting model to train
    model.train()
    start_time = time.time()

    for epoch in range(start_epoch,max_epochs):
        iter_start_time = time.time()

        # lr with cosine schedule
        lr = get_lr_cosine_schedule(
            t=epoch,
            alpha_max=learning_rate,
            alpha_min=min_learning_rate,
            T_w = warmup_iterations,
            T_c = cosine_iterations
        )

        for param_group in optimizer.param_groups:
            param_group['lr']= lr
        
        # get training batch
        inputs,targets = data_loader(
            training_data,
            batch_size,
            context_length,
            device
        )

        # forward pass
        logits =  model(inputs)
        loss = cross_entropy(logits,targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient cliping
        gradient_clipping(model.parameters(),max_norm=grad_clip_norm)

        #optimizer step
        optimizer.step()

        # loss collection
        train_losses.append(loss.item())

        #logging
        if (epoch+1) % log_every ==0 :
            avg_loss = np.mean(train_losses[-log_every:])
            time_elapsed = time.time() - start_time
            iteration_time = time.time() - iter_start_time
            tokens_per_sec = (batch_size * context_length) / iteration_time

            print(
                f"Epoch: {epoch+1 } / {max_epochs} |"
                f"Loss: {avg_loss:.4f} | "
                f"Learning Rate: {lr:.2e} |"
                f"Tokens/sec: {tokens_per_sec:.0f} |"
                f"Time : {time_elapsed:.1f} " 
            )
        
        #validation
        if val_data is not None and (epoch + 1) % eval_every ==0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for i in range(eval_iterations):
                    val_inputs,val_targets = data_loader(
                        val_data,
                        batch_size,
                        context_length,
                        device
                    )

                    val_logits = model(val_inputs)
                    val_loss =cross_entropy(val_logits,val_targets)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)

            print(f"{'=' * 40}")
            print(f"Validation loss : {avg_val_loss:.4f}")
            print(f"{'=' * 40}")


            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(model,optimizer,epoch,best_path)
                print(f"Best model savd: {avg_val_loss:.4f}")
            
            model.train()

        # save checkpoint
        if (epoch +1)% save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_iter_{epoch + 1}.pt'
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # final checkpoint
    final_path =  checkpoint_dir / 'final_model.pt'
    save_checkpoint(model,optimizer,epoch-1,final_path)


    #total training time
    total_time =time.time()- start_time

    # training complete
    print(f"\n"+"=" * 80)
    print(f"\n Training complete")
    print(f"\n" + "="* 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Final checkpoint saved: {final_path}")

    if val_data is not None:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)



def main():
    """getting some help"""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")

    
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10000)  # renamed from max_iterations
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_iterations", type=int, default=1000)
    parser.add_argument("--cosine_iterations", type=int, default=8000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

   
    parser.add_argument("--train_data_path", default="./encoded_datasets/tinystories_train.npy", type=str)
    parser.add_argument("--val_data_path", default="./encoded_datasets/tinystories_val.npy")

    
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--restart_epoch_path", type=str, default=None)  # renamed from resume_from

    
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_iterations", type=int, default=20)

    
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

   
    train(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        training_data_path=args.train_data_path,  
        val_data_path=args.val_data_path,
        device=args.device,
        rope_theta=args.rope_theta,

        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iterations=args.warmup_iterations,
        cosine_iterations=args.cosine_iterations,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip_norm=args.grad_clip_norm,

        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,

        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_iterations=args.eval_iterations,

        restart_epoch_path=args.restart_epoch_path,
    )

if __name__ == "__main__":
    main()



    