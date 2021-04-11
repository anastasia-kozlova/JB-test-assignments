import torch 
import time
from save_load import save_model, save_metrics

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(file_path, 
          model,
          optimizer,
          dataloader_train,
          dataloader_valid,
          eval_every=100,
          num_epochs = 1,
          best_valid_loss = float('inf')):
    
    # initialize 
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training 
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader_train:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)

            output[0].backward()
            optimizer.step()

            running_loss += output[0].item()
            global_step += 1

            # validation
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                    for batch in dataloader_valid:
                        
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        output = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)
                    
                        valid_running_loss += output[0].item()

                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(dataloader_valid)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # reinitialize
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                print('epoch {}/{}, step {}/{}, train loss: {:.4f}, valid loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(dataloader_train),
                              average_train_loss, average_valid_loss))
                
                # save
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_model(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

def evaluate(model, dataloader):
    y_pred = []
    y_true = []
    model.eval()
    since = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            
            y_pred.extend(output[0].cpu().numpy())
            
            if "labels" in batch:
                labels = batch['labels'].numpy()
                y_true.extend(labels)
    time_eval = time.time() - since
    return y_pred, y_true, time_eval