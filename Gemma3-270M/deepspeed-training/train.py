from torch import amp
import torch
import torch.nn as nn

def train_loop(model, train_dataloader, optimizer, lr_scheduler, criterion, vocab_size,
               max_steps, gradient_accumulation_steps, log_n_steps, num_epochs, device, save_every):

    model.train()
    optimizer.zero_grad()
    step = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(train_dataloader):
            if step >= max_steps:
                print("Reached max steps. Ending training.")
                torch.save(model.state_dict(), "model_final.pt")
                torch.save(optimizer.state_dict(), "optimizer_final.pt")
                return

            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)

            with amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                loss = loss / gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % log_n_steps == 0:
                    print(f"Step {step} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}")
            
            # save according to save_every
            if step % save_every == 0 and step > 0:
                torch.save(model.state_dict(), f"model_step{step}.pt")
                torch.save(optimizer.state_dict(), f"optimizer_step{step}.pt")
