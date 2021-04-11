import click
import numpy as np
from execution import evaluation

MAX_LEN = 80
BATCH_SIZE = 32
FOLDER = 'roberta_result' # папка с дообученной моделью (модель не поместилась)


@click.command()
@click.argument('text')
def main(text):
    tweets = [text]
    outputs = evaluation(file_path=FOLDER,
                          tweets=tweets,
                          max_len=MAX_LEN,
                          batch_size=BATCH_SIZE)
    
    labels_pred = np.argmax(outputs['labels_pred'], axis=1)[0]
    print("Класс и время работы модели, без учета загрузок библиотек")
    print(labels_pred)
    print(outputs['time_eval'])
    
    
if __name__ == "__main__":
    main()