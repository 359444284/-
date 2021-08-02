import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_packed_sequence


def train_epoch(
        model,
        optimizer,
        scheduler,
        device,
        loss_fn,
        train_dataloader,
        val_dataloader=None,
        epochs=10,
        is_lstm=False
):

    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    # print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc': ^ 9} | {'Elapsed': ^ 9}")
    print("-" * 60)

    for epoch_i in range(epochs):
        t0_epoch = time.time()
        total_loss = 0
        model = model.train()

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            # attention_mask = batch["attention_mask"].to(device)

            if is_lstm:
                lengths = batch["real_len"]
                a_lengths, idx = lengths.sort(0, descending=True)

                _, un_idx = torch.sort(idx, dim=0)

                input_ids = input_ids[idx]

                output = model(
                    input_ids=input_ids,
                    input_lens=a_lengths
                )

                output = torch.index_select(output, 0, un_idx)

            else:
                output = model(
                    input_ids=input_ids,
                    # attention_mask=attention_mask
                )

            loss = loss_fn(output, targets)
            total_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        # evaluation

        if val_dataloader is not None:
            val_loss, val_accuracy, f1 = evaluate(model, val_dataloader, device, loss_fn, is_lstm)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # torch.save(model.state_dict(), 'best_model_state.bin')
                # Print performance over the entire training data
                # time_elapsed = time.time() - t0_epoch
                # print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss: ^ 10.6f} | {val_accuracy: ^ 9.2f} | {time_elapsed: ^ 9.2f}")
            print([epoch_i + 1, avg_train_loss,val_loss, val_accuracy, f1])

        print("\n")
        # print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
        print(best_accuracy)


def evaluate(model, val_dataloader, device, loss_fn, is_lstm):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    pred_all = []
    lable_all = []
    # For each batch in our validation set...
    with torch.no_grad():
        for batch in val_dataloader:
            # Load batch to GPU
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            # attention_mask = batch["attention_mask"].to(device)

            # Compute logits

            if is_lstm:
                lengths = batch["real_len"]
                a_lengths, idx = lengths.sort(0, descending=True)

                _, un_idx = torch.sort(idx, dim=0)

                input_ids = input_ids[idx]

                output = model(
                    input_ids=input_ids,
                    input_lens=a_lengths
                )

                output = torch.index_select(output, 0, un_idx)

            else:
                output = model(
                    input_ids=input_ids,
                    # attention_mask=attention_mask
                )

            # Compute loss
            loss = loss_fn(output, targets)
            val_loss.append(loss.item())

            # Get the predictions
            m = nn.Sigmoid()

            preds = m(output)
            # preds = torch.argmax(output, dim=1).flatten()
            # pred_all.extend(preds.cpu().numpy())
            # lable_all.extend(targets.cpu().numpy())
            # Calculate the accuracy rate
            for idx in range(targets.size()[1]):
                pred = np.asarray(preds[:, idx])
                target = targets[:, idx].cpu().numpy()
                pred = [1 if p >= 0.5 else 0 for p in pred]
                accuracy = (pred == target).mean() * 100
                val_accuracy.append(accuracy)

                pred_all.extend(pred)
                lable_all.extend(target)




    # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy, f1_score(lable_all,pred_all)
