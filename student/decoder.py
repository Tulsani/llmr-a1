import torch
import torch.nn.functional as F
from student.transformer import TransformerLM


def decode(model,prompt,max_new_tokens =50, 
           temperature = 1.0, top_p=None, eos_token_id=None,device='cuda'):
    
    #set model to eval
    model.eval()

    #prompt to tensor
    generated = torch.tensor([prompt],dtype=torch.long,device=device)

    #no grad
    with torch.no_grad():
    
        for i in range(max_new_tokens):
            if generated.size(1) > model.context_length:
                input_seq = generated[:, -model.context_length:]
            else:
                input_seq = generated

            #forward pass
            logits = model(input_seq)

            # logits for the last position
            next_token_logits = logits[:,-1,:]

            # temp scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            #convert logits to probs
            probs = F.softmax(next_token_logits,dim=-1)

            # applying top-p if needed
            if top_p is not None:
                
                #sort probs
                sorted_probs,sorted_indices = torch.sort(probs, descending=True,dim=-1)

                # cumulating probs
                cumulative_probs = torch.cunsum(sorted_probs,dim=-1)

                # get smallest set where cumulative prob >=top_k
                # remove tokens with probs > threholds
                sorted_indices_to_remove = cumulative_probs>top_p

                #keep one
                sorted_indices_to_remove[:,1:] = sorted_indices_to_remove[:,:-1].clone()
                sorted_indices_to_remove[:,0] = False

                #setup the mask
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1,sorted_indices,sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0.0

                #normalize probs
                probs = probs/probs.sum(dim=-1,keepdim=True)

            # sample from prob dist
            next_token = torch.multinomial(probs,num_samples=1)

            #append
            generated = torch.cat([generated,next_token],dim=1)

            # checking for eos
            if eos_token_id is not None  and next_token.item() == eos_token_id:
                break

    return generated[0].tolist()
    

def generate_text(
        model: TransformerLM,
        tokenizer,
        prompt,
        max_new_tokens=50,
        temprature = 1.0,
        top_p = None,
        device = 'cuda'
):
    #encode prompt
    prompt_tokens = tokenizer.encode(prompt)

    # eos
    eos_token_id = getattr(tokenizer,'eos_token_id',None)

    #generate token
    generated_tokens = decode(
        model=model,
        prompt= prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temprature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device
    )

    #decode
    generated_text= tokenizer.decode(generated_tokens)

    return generated_text


        