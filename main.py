# from numpy.doc.constants import m
import datetime
from pathlib import Path
import os
import dill
import numpy
import time as t
import multi_training
import config
from config import training_base
from config import base
from midi_to_statematrix import noteStateMatrixToMidi
from model import Model



# Funkcja generująca adaptacyjną sekwencję muzyczną na podstawie modelu
def gen_adaptive(model, pieces, times, keep_thoughts=False, name="final"):
    # Przypisywanie zmiennych 
    global start_time
    xIpt, xOpt = map(
        lambda x: numpy.array(x, dtype="int8"), multi_training.getPieceSegment(pieces)
    )
    # Tworzenie listy i inicjalizacja pierwszym elementem xOpt[0]
    all_outputs = [xOpt[0]]
    # Tworzenie pustej listy przechowującej wszystkie wyniki
    all_thoughts = []
    # Wywołanie funkcji start_slow_walk przekazującej wartość xIpt[0]
    model.start_slow_walk(xIpt[0])
    # Inicjalizacja zmiennej 
    cons = 1
    # Pętla wykonuje się "time"-ilość razy 
    for time in range(multi_training.BATCH_LENGTH * times):
        start_time = t.time() # Początkowy czas dla obliczeń
        # Wywouje metodę slow_walk_fun przekazującą wartość cons i zapisuje wynik do zmiennej "resdata"
        resdata = model.slow_walk_fun(cons)
        # Obliczenie sumy elementów macierzy  "resdata" w pierwszej kolumnie i zapisanie do zmiennej "nnotes"
        nnotes = numpy.sum(resdata[-1][:, 0])
        if nnotes < 2:
            if cons > 1:
                cons = 1
            cons -= 0.02
        else:
            cons += (1 - cons) * 0.3
        all_outputs.append(resdata[-1])
        if keep_thoughts:
            all_thoughts.append(resdata)
    # Konwertowanie all_outputs na format numpy.array i zapisanie do pliku
    noteStateMatrixToMidi(numpy.array(all_outputs), "output/" + name)
    if keep_thoughts:
        # spróbuję tutaj dodać ten finalny plik z oryginalnego treningu
         with open(f"output/{name}.pkl", "wb") as thoughts_file:
        # with open(f"/Users/adrianrobak/Documents/GitHub/biaxial-rnn-music-composition2/final_learned_config.pkl", "wb") as thoughts_file:
            dill.dump(all_thoughts, thoughts_file)

    # Obliczenie czasu trwania kroku
    step_duration = t.time() - start_time

    # Wyświetlanie czasu trwania kroku
    print(f"Duration of generating sequence: {t.time() + 1}, Duration: {step_duration} seconds")

def fetch_train_thoughts(model, pieces, batches, name="trainthoughts"):
    all_thoughts = []
    start_time = t.time() # Początkowy czas rozpoczęcia procesu
    for _ in range(batches):
        ipt, opt = multi_training.getPieceBatch(pieces)
        thoughts = model.update_thought_fun(ipt, opt)
        all_thoughts.append((ipt, opt, thoughts))
    
    end_time = t.time() # Pomiar czasu zakończenia procesu
    duration = end_time - start_time # Obliczenie czasu trwania procesu
    
    print(f"Fetch train toughts duration: {duration:.2f} seconds")

    with open(f"output/{name}.pkl", "wb") as thoughts_file:
        dill.dump(all_thoughts, thoughts_file)


if __name__ == "__main__":
    training_data_path = Path(config.base)
    if not training_data_path.exists():
        raise Exception(f"No data found in {training_data_path}")

    output_path = Path("./output")
    if not output_path.exists():
        output_path.mkdir()

    start_time = t.time() # Początkowy czas dla obliczeń

    pieces = multi_training.loadPieces(training_data_path)


    load_data_duration = t.time() - start_time # Obliczenie czasu trwania wczytywania danych
    print("Duration of loading all files data: {:.2f} seconds".format(load_data_duration))

    model = Model([300, 300], [100, 50], dropout=config.dropout)

    training_start_time = t.time() # Pomiar czasu rozpoczęcia treningu

    # Wywołanie funkcji trainPiece przekazującej wartość model, pieces i liczby epok w treningu równej 15000
    multi_training.trainPiece(model, pieces, 15000)

    training_duration = t.time() - training_start_time # Obliczenie czasu trwania treningu

    date = datetime.datetime.now().strftime("%H:%M:%S-%d.%m.%Y")

    with open(f"{output_path}/final_learned_config_{date}_{config.training_base}.pkl", "wb") as saved_state:
    # Alternatywnie można już wytrenowany wcześniej model podać do generowania sekwencji
    # with open(f"/Users/adrianrobak/Documents/GitHub/biaxial-rnn-music-composition2/final_learned_config.pkl", "wb") as saved_state:
        dill.dump(model.learned_config, saved_state)
        
    # parametr model... 
    # parametr pieces...
    # parametr 20 oznacza liczbę wywołań dla jednego batcha czyli wpływa
    # na długość generowanego zbioru danych czyli na długość generowanego utworu
    # parametr name...
    date = datetime.datetime.now().strftime("%H:%M:%S-%d.%m.%Y")

    gen_start_time = t.time() # Pomiar czasu rozpoczęcia generowania sekwencji

    gen_adaptive(model, pieces, 80, name=f"{config.training_base}_{date}_dropout-{model.dropout}")

    gen_duration = t.time() - gen_start_time # Obliczenie czasu trwania generowania sekwencji

    print("Generating adaptive sequence duration: {:.2f} seconds".format(gen_duration))
# save model weights
# pickle.dump( m.learned_config, open( "path_to_weight_file.pkl", "wb" ) )

# load model weight
# learned_config = pickle.load(open( "/Users/adrianrobak/Documents/GitHub/biaxial-rnn-music-composition2/final_learned_config.pkl", "rb" ) )
