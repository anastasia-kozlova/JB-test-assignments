from sklearn.model_selection import train_test_split
import time
import torch 
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW

from preprocess import create_data_loader
from train_eval_fun import train, evaluate
from save_load import load_model
import logging


logger = logging.basicConfig(level=logging.INFO)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)


def training(tweets, 
             labels, 
             file_path,
             model=model,
             optimizer=optimizer,
             tokenizer=tokenizer,
             max_len=512,
             eval_step=100, 
             epochs=1, 
             batch_size=1, 
             best_valid_loss=float('inf')):
    
    tweets_train, tweets_valid, labels_train, labels_valid= train_test_split(tweets, labels, test_size=0.3)

    dataloader_train = create_data_loader(tokenizer, 
                                          tweets_train, 
                                          labels_train, 
                                          max_len=max_len, 
                                          batch_size=batch_size)
    dataloader_valid = create_data_loader(tokenizer, 
                                          tweets_valid, 
                                          labels_valid, 
                                          max_len=max_len, 
                                          batch_size=batch_size)        
    logging.info("Данные для обучения обработаны")
    logging.info("Начато обучение модели")

    train(file_path=file_path,
          model=model,
          optimizer=optimizer,
          dataloader_train=dataloader_train,
          dataloader_valid=dataloader_valid,
          eval_every=eval_step,
          num_epochs=epochs,
          best_valid_loss=best_valid_loss)
    logging.info("Обучение завершено")
        
def evaluation(tweets, 
               labels=None,
               file_path=None,
               model=model,
               tokenizer=tokenizer, 
               max_len=512,
               batch_size=1):
    since = time.time()
    dataloader_test = create_data_loader(tokenizer,
                                         tweets,
                                         labels,
                                         max_len=max_len,
                                         batch_size=batch_size)
    logging.info("Данные для предсказания обработаны")
    # загрузить модель исходную или дообученную модель
    if file_path:
        try:
            load_model(file_path + '/model.pt', model)
        except FileNotFoundError:
            print('Дообученная модель отсутствует. Используется исходная')
            
    labels_pred, labels_true, time_eval = evaluate(model, dataloader_test)
    all_time = time.time() - since
    logging.info("Предсказание завершено")
    return {'labels_pred': labels_pred,
            'labels_true': labels_true, 
            'time_eval': time_eval, 
            'all_time': all_time}

