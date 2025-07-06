# @title train test generate
import torch
from torch.nn import functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
scaler = torch.GradScaler()

# https://www.comet.com/site/blog/perplexity-for-llm-evaluation/
def Perplexity(logits, target): # [b,t,vocab_size], [b,t]
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1) # [b,t]
    perplexity = nll.mean().exp()
    return perplexity

# logits = torch.randn(2, 4, 10)
# target = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
# perplexity = Perplexity(logits, target)
# # perplexity = Perplexity(logits[:,:-1], y[:,1:])
# print(f'Perplexity: {perplexity}')

def strain(model, dataloader, optimizer, scheduler=None): # train function with automatic mixed precision
    model.train()
    for i, x in enumerate(dataloader):
        x, y = x[:,:-1], x[:,1:]
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # causal_mask = torch.ones(x.size(1), x.size(1), dtype=bool, device=device).tril(diagonal=0).repeat(x.shape[0],1,1) # for F.scaled_dot_product_attention
            # # causal_mask = ~torch.ones(x.size(1), x.size(1), dtype=bool, device=device).tril(diagonal=0).repeat(x.shape[0],1,1)
            # logits = model(x, mask=causal_mask) #output = [batch size, trg len - 1, output dim]
            logits = model(x) #output = [batch size, trg len - 1, output dim]
            # logits, _ = model(x) # rnn
            loss = F.cross_entropy(logits.flatten(0,1), y.flatten().to(int)) # [b*t,d], [b*t]
            # loss = F.cross_entropy(logits.flatten(0,1), y.flatten()) # [b*t,d], [b*t]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # scaler.unscale_(optim)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        # if scheduler is not None: scheduler.step()
        if i % 100 == 0: print("strain",loss.item())
            # perplexity = Perplexity(logits.detach(), y).item()
        try: wandb.log({"train loss": loss.item()/len(y)})
        except NameError: pass

# def test(loader, model):
#     model.eval()
#     total_loss = 0
#     for i, (x, y) in enumerate(loader):
#         x, y = x.to(device), y.to(device)
#         with torch.no_grad():
#             logits = model(x)
#             # logits, _ = model(x) # rnn
#             perplexity = Perplexity(logits[:,:-1], y[:,1:])
#         loss = F.cross_entropy(logits.flatten(0,-2), y.flatten()) # [batch*seq_len, vocab_size], [batch*seq_len]
#         total_loss+=loss.item()
#         # if i % 100 == 0: print("test",loss.item())
#         if i % 100 == 0: print("test",loss.item(),"ppty",perplexity.item())
#         try: wandb.log({"test loss": loss.item()/len(y)})
#         except NameError: pass
#     return total_loss / len(loader)

def generate(model, context, max_steps=64, temperature=1):
    x = encode(context)#.to(device)
    model.eval()
    for n in range(max_steps):
        with torch.no_grad():
            output = model(x)
            # output, hidden = model(x, hidden) # rnn
        # print('generate', output.shape, hidden.shape)
        # hidden = hidden[:, -1, :].unsqueeze(1) # RNN/GRU
        output = output[:, -1] # get logit for last character
        output = output/temperature
        output = F.softmax(output, dim=-1) # vocab_size to char
        ix = torch.multinomial(output, num_samples=1) # rand sample by output distribution
        x = torch.cat((x, ix), dim=1)
    completion = decode(x.squeeze(0))
    return completion

# import time
# start = begin = time.time()
for i in range(1):
# for i in range(30):
    # train_loss = strain(model, train_loader, optim, scheduler=None)
    strain(model, train_loader, optim, scheduler=None)
    # test_loss = test(test_loader, model)
    # print('Test Loss:', test_loss)
    # print(generate(model, "this is what"))
    # print(i, 'time:',time.time() - start, (time.time()-begin)/(i+1))
    # start = time.time()
