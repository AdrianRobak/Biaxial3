import datetime
import random
import signal
from itertools import chain
from pathlib import Path
import config
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import dill
import numpy
import data
import music21
import os
from config import training_base
from config import dropout
from scipy.signal import savgol_filter
from datetime import datetime

from midi_to_statematrix import (
    midiToNoteStateMatrix,
    noteStateMatrixToMidi,
)

BATCH_WIDTH = 10  # number of sequences in a batch
BATCH_LENGTH = 16 * 8  # length of each sequence
DIVISION_LENGTH = 16  # interval between possible start locations


# Deklarowanie funkcji wczytywania utworów muzycznych z path Funkcja wczytuje wszystkie utwory z folderu dirpath i
# zwraca słownik, w którym kluczami są nazwy utworów, a wartościami są ich macierze zawierającej informacje o
# pozycjach nut na utworze.
def loadPieces(dirpath: Path) -> dict[str, list]:
    pieces = {}
    start_time = time.time()  # Czas uruchomienia funkcji
    # Wywołanie funkcji glob z parametrem "*.mid" wyszukuje wszystkie pliki midi w folderze dirpath.
    # Funkcja konwertuje pliki midi na macierze zawierające informacje o pozycjach nut na utworze.
    # Wyniki zwracane są w postaci listy Path.
    for path in chain(dirpath.glob("*.mid"), dirpath.glob("*.MID")):
        start_loading_time = time.time()  # Czas rozpoczęcia wczytywania pliku
        outMatrix = midiToNoteStateMatrix(path)
        loading_time = time.time() - start_loading_time  # Czas wczytania pliku
        # Sprawdzanie czy długość utworu jest większa niż BATCH_LENGTH. Jeśli tak, to utwór jest ignorowany.
        if len(outMatrix) < BATCH_LENGTH:
            continue
        # Jeżeli utwór ma odpowiednią długość, to utwör jest dodany do słownika pieces. Kluczem jest nazwa pliku bez rozszerzenia.
        pieces[path.stem] = outMatrix
        # Wyświetlenie informacji o utworze, który został dodany do słownika pieces.
        print("Loaded", path.stem)
        print(f"Loading time {path.stem}: {loading_time * 1000:.2f} ms")
    total_time = time.time() - start_time  # Czas wykonania funkcji
    print(f"Total time of loading pieces: {total_time: .4f} seconds")

    return pieces


# Deklarowanie funkcji wczytywania modelu z path Funkcja wczytuje model z pliku modelPath. Słownik zawiera teraz wczytane utwory.
def getPieceSegment(pieces):
    start_time = time.time()
    # Losowanie klucza z słownika pieces. Klucz jest nazwa utworu. Wartość jest macierz zawierająca informacje o
    # pozycjach nut na utworze. Utwör jest wylosowany z listy pieces.values(). Wartość zwracana jest macierz
    # zawierająca informacje o pozycjach nut na utworze. Utwör jest wylosowany z listy pieces.values().
    piece_output = random.choice(list(pieces.values()))

    # total_duration = 0  # Inicjalizacja zmiennej przechowującej łączny czas wykonania

    # num_steps = BATCH_WIDTH  # Liczba kroków wykonywania

    # for step in range(num_steps):
    #    start_time = time.time()
    # Kod wykonywany w kroku
    # Pomiar czasu po wywołaniu random.choice()
    #    end_time = time.time()
    #    duration = end_time - start_time
    #    total_duration += duration  # Dodanie czasu wykonania kroku do łącznego czasu

    # nie jest potrzebne zaśmiecanie widoku :)
    # print("Time taken for step {}: {:.2f} msec".format(step, duration * 1000))

    # nie jest potrzebne zaśmiecanie widoku :)
    # print("Total duration of getting pieces segment: {:.2f} msec".format(total_duration * 1000))

    # Wylosowany jest punkt początkowy w macierzy piece_output. Wartość zwracana jest macierz zawierająca informacje
    # o pozycjach nut na utworze. Punktem początkowym jest punkt o wartości random.randrange(0, len(piece_output) -
    # BATCH_LENGTH, DIVISION_LENGTH). Krokiem jest 16. Punkt początkowy jest przypisany do zmiennej start.
    start = random.randrange(0, len(piece_output) - BATCH_LENGTH, DIVISION_LENGTH)
    # print(f"Range is {0} {len(piece_output)-BATCH_LENGTH} {DIVISION_LENGTH} -> {start}")

    # Tworzenie segmentu wyjściowego seg_out, który zawiera utwor o pozycjach nut na utworze, zaczynając od punktu
    # start, i ma długość BATCH_LENGTH.
    seg_out = piece_output[start: start + BATCH_LENGTH]
    # Tworzenie segmentu wejściowego seg_in, ktöry zawiera utwor o pozycjach nut na utworze. Następuje konwertowanie
    # segmentu wyjściowego seg_out na formę wejściową dla modelu.
    seg_in = data.noteStateMatrixToInputForm(seg_out)

    # Zwracanie segmentu wejściowego seg_in i segmentu wyjściowego seg_out.
    return seg_in, seg_out
    # # Te segmenty wejściowe i wyjściowe są wykorzystywane w procesie treningu modelu do przewidywania kolejnych nut
    # na podstawie wcześniejszych nut.


# Funkcja zwraca macierz zawierającą informacje o pozycjach nut na utworze. Utwör jest wylosowany z listy
# pieces.values(). Funkcja getPieceSegment pobiera pojedynczy segment (wejścia i wyjścia) i tworzy partię danych
# poprzez powtórzenie tego procesu BATCH_WIDTH razy.
def getPieceBatch(pieces):
    # Wywołanie funkcji getPieceSegment BATCH_WIDTH razy, która wywołuje getPieceSegmentt(pieces) dla każdego
    # powtórzenia od 0 do BATCH_WIDTH. Zapisywane są wartości zwracane przez funkcję getPieceSegment (wejścia i
    # wyjścia) w oddzielnych zmienny 'i' i 'o'.
    i, o = zip(*[getPieceSegment(pieces) for _ in range(BATCH_WIDTH)])
    # Funkcja tworzy tablicę numpy.array, która z tych wartości dla wszystkich segmentów i  zamienia zbiory wartości
    # zwracane przez funkcję getPieceSegment na macierze - dwie tablice numpy.array.
    return numpy.array(i), numpy.array(o)
    # Tak powstają dwie tablice numpy.array, ktére są wykorzystywane w procesie treningu modelu do przewidywania
    # kolejnych nut na podstawie wcześniejszych nut. Każda tablica zawiera BATCH_WIDTH segmentów danych.


# Deklaruje funkcję trainPiece z parametrami model, pieces, epochs i opcjonalnym parametrem start, który domyślnie ma
# wartość 0.
def trainPiece(model, pieces, epochs, start=0):
    # Tworzenie listy stopflag zawierającą jeden element o wartości logicznej False. Ta zmienna jest używana do
    # kontrolowania zatrzymywania treningu.
    global date
    stopflag = [False]

    # Deklarowanie funkcji signal_handler, która będzie obsługiwać sygnały, które mogą zatrzymać trening.
    def signal_handler(signame, sf):
        stopflag[0] = True

    # Przypisywanie oryginalnej obsługę sygnału przerwania (Ctrl+C) do zmiennej old_handler, a następnie ustawia nową
    # obsługę sygnału, któna wykorzystuje funkcję signal_handler. Allow interrupt to stop training only
    old_handler = signal.signal(signal.SIGINT, signal_handler)

    last_epoch = 0

    start_time = datetime.now()
    epoch_start_time = time.time()  # Zapisanie czasu rozpoczęcia epoki
    print(f"Start time of train piece: {start_time}")

    epoch_list = []
    error_list = []

    start_counting_epochs = 1  # Przykładowa wartość

    # Przygotowanie nowego okna wykresu
    plt.figure(figsize=(10, 8))

    for epoch in range(start, start + epochs):
        # Jeśli wartość stopflag wynosi True, przerywa pętlę treningową.
        if stopflag[0]:
            print("last_epoch:", last_epoch)
            break
        else:
            print("Epoch:", epoch + 1)

        # Aktualizacja wag modelu i pobranie błędu
        error = model.update_fun(*getPieceBatch(pieces))

        # Jeżeli jest dostępna poprzednia wartość błędu, możesz ją wyświetlić lub wykonać inne operacje
        # if previous_error is not None:
        #    print("Previous error:", previous_error)

        # Aktualizacja poprzedniej wartości błędu od drugiej epoki
        # if epoch >= start + 1:
        #    previous_error = error

        # Inicjalizacja zmiennych licznika błędu
        # epochs_without_improvement = 0  # Deklarowanie ilości epok bez spadku błędu
        # start_counting_epochs = 10  # Epoka, od której zaczynamy liczyć spadki błędu
        # max_epochs_without_improvement = 20  # Maksymalna liczba epok bez spadku błędu
        # num_epochs = epochs  # Liczba epok treningowych

        # Przygotowanie list do przechowywania danych dla wykresu
        # epoch_list = []
        # error_list = []

        # Wypisuje postęp treningu co 1 epokę, wraz z numerem epoki, wartością błędu oraz informacjami o czasie.
        if epoch % 1 == 0:
            last_epoch = epoch

            epoch_start_time = epoch_start_time

            end_time = datetime.now()
            elapsed_time = end_time - start_time

            average_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - epoch - 1
            remaining_time = average_time_per_epoch * remaining_epochs

            days = elapsed_time.days
            hours, remainder = divmod(elapsed_time.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            eta_days = remaining_time.days
            eta_hours, eta_remainder = divmod(remaining_time.seconds, 3600)
            eta_minutes, eta_seconds = divmod(eta_remainder, 60)

            # epoch_duration = datetime.datetime.now() - epoch_start_time
            # epoch_hours, epoch_remainder = divmod(epoch_duration.seconds, 3600)
            # epoch_minutes, epoch_seconds = divmod(epoch_remainder, 60)

            date = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

            print(f"Epoch {epoch}, error={error}, Finish Time: {date}, ")
            # print(f"Epoch duration: {epoch_hours} h, {epoch_minutes} m, {epoch_seconds} s")
            print(
                f"Etime: {days} d, {hours} h, {minutes} m, {seconds} s, ETA: {eta_days} d, {eta_hours} h, {eta_minutes} m, {eta_seconds} s")

            ## WYKRES

            last_errors = []

            # Dodawanie danych do list dla wykresu
            epoch_list.append(epoch)
            error_list.append(error)
            last_errors.append(error)

            # Wykres błędu
            if epoch >= start_counting_epochs:
                start_counting_epochs = epoch + 1

                # Dodanie informacji o obecnej epoce
                # plt.text(0.98, 0.98, f"Current Epoch: {epoch}", transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')
                plt.text(0.50, 0.98, f"Current Epoch: {epoch}", transform=plt.gcf().transFigure,
                         verticalalignment='top')
                plt.text(0.50, 0.96, f"Estimated time of End of the Training: {eta_days} d, {eta_hours} h, {eta_minutes} m, {eta_seconds} s", transform=plt.gcf().transFigure, verticalalignment='top')
                plt.text(0.50, 0.94, f"Average time per epoch: {average_time_per_epoch}", transform=plt.gcf().transFigure, verticalalignment='top')

                # Dodanie krzywej błędu
                plt.plot(epoch_list, error_list, "r", label="Error")  # Linia błędu czerwona

                # Znalezienie indeksu najmniejszego błędu i wyświetlenie go na wykresie
                min_error_index = numpy.argmin(error_list)
                min_error = error_list[min_error_index]
                min_error_epoch = epoch_list[min_error_index]

                # Wyświetlenie informacji o najmniejszym błędzie
                min_error_rounded = numpy.round(min_error, 5)
                plt.scatter(min_error_epoch, min_error, color='red', label="Min Error")
                plt.text(0.02, 0.98, f"Min Error: {min_error_rounded}", transform=plt.gcf().transFigure,
                         verticalalignment='top', fontsize=10)

                # Dodanie informacji o dropout
                plt.text(0.02, 0.96, f"Dropout: {config.dropout}", transform=plt.gcf().transFigure,
                         verticalalignment='top', fontsize=10)

                # Dodanie informacji o liczbie epok do końca treningu
                Epochs_to_finish = 15000 - epoch
                plt.text(0.02, 0.94, f"Number of epochs to end of the training: {Epochs_to_finish}",
                         transform=plt.gcf().transFigure, verticalalignment='top', fontsize=10)

                # Linia trendu dla błędu w funkcji liniowej
                z = numpy.polyfit(epoch_list, error_list, 1)
                p = numpy.poly1d(z)
                plt.plot(epoch_list, p(epoch_list), color=(147 / 255, 246 / 255, 0 / 255, 1.0), linestyle='--',
                         label="Linear Trend")  # Linia trendu limonkowa ;)

                # Dodanie linii trendu dla błędu w funkcji wielomianowej z jej dopasowaniem deg
                # Dopasowanie wielomianu do danych
                best_deg = None
                best_aic = float('inf')

                for deg in range(1, 10):
                    coefficients = numpy.polyfit(epoch_list, error_list, deg)
                    polynomial = numpy.poly1d(coefficients)
                    y_fit = polynomial(epoch_list)

                    residuals = error_list - y_fit
                    mse = numpy.mean(residuals ** 2)
                    n = len(epoch_list)
                    aic = 2 * deg + n * numpy.log(mse)

                    if aic < best_aic:
                        best_aic = aic
                        best_deg = deg

                # Dopasowanie danych z optymalnym stopniem wielomianu
                best_coefficients = numpy.polyfit(epoch_list, error_list, best_deg)
                best_polynomial = numpy.poly1d(best_coefficients)
                x_fit = numpy.linspace(min(epoch_list), max(epoch_list), 100)
                y_fit = best_polynomial(x_fit)

                # Linia trendu dla błędu funkcji wielomianowej z optymalnym stopniem
                plt.plot(x_fit, y_fit, "b-",
                         label=f"Polynominal Trend (deg={best_deg})")  # Wielomianowa linia trendu niebieska

                # Przykładowe prognozowanie błędu dla kolejnych 3 epok
                forecast_epochs = range(max(epoch_list) + 1, max(epoch_list) + 4)
                forecast_errors = [best_polynomial(epoch) for epoch in forecast_epochs]

                # Płynna zmiana liczby prognozowanych błędów (linia prognozowanych błędów będzie zajmowała 20 część wykresu)
                num_forecast_epochs = max(5, int(len(
                    epoch_list) / 20))  # Liczba prognozowanych epok jako 1/20 długości listy epoch_list
                forecast_epochs_smooth = numpy.linspace(max(epoch_list) + 1, max(epoch_list) + num_forecast_epochs,
                                                        num_forecast_epochs)
                forecast_errors_smooth = [best_polynomial(epoch) for epoch in forecast_epochs_smooth]

                forecast_display_epochs = 50
                if epoch < forecast_display_epochs:
                    plt.plot(forecast_epochs, forecast_errors, "g", label="Forecast")
                else:
                    lines = plt.gca().get_lines()
                    lines.pop(1)
                    plt.plot(forecast_epochs_smooth, forecast_errors_smooth, "g",
                             label="Forecast")  # Linia prognozy ciągła zielona

                # Wyświetlanie trzech najbliższych prognoz błędu
                forecast_text = ", ".join([f"{forecast_error}" for forecast_error in forecast_errors[:3]])
                print(f"Forecast error for epoch {forecast_epochs[0] + 1} to {forecast_epochs[0] + 3}: {forecast_text}")

                # Wyświetlanie ostatnich pięciu błędów ver.3
                last_errors_values = list(zip(epoch_list, error_list))
                last_errors_values.reverse()
                last_errors_values = last_errors_values[:5]

                plt.annotate("Values of the last errors", xy=(0.02, 0.4), xycoords='axes fraction', fontsize=10,
                             fontweight='bold')
                for i, (epoch_val, error_val) in enumerate(last_errors_values, 1):
                    rounded_error_val = numpy.round(error_val,
                                                    5)  # Zaokrąglenie wartości błędu do pięciu miejsc po przecinku
                    plt.annotate(f"Error of the Epoch {epoch_val}  {rounded_error_val:.5f}", xy=(0.02, 0.4 - i * 0.02),
                                 xycoords='axes fraction', fontsize=8)

                # plt.text(0.98, 0.98, f"Current Epoch: {epoch}", transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')

                plt.xlabel('Epoch')
                plt.ylabel('Error')
                plt.title(f'Error Progress of the {config.training_base} Neural Network Training')
                plt.legend()  # To-do: Zmień położenie legendy na zewnątrz wykresu po prawej stronie
                plt.show(block=False)  # Wyświetlenie wykresu bez blokowania kodu
                plt.pause(0.1)  # Poczekaj krótko, aby umożliwić odświeżenie okna wykresu
                plt.clf()  # Wyczyść aktualny wykres

        # Warunek sprawdzający, czy należy zapisać połowę postępu treningu. Sprawdza, czy numer epoki jest podzielny
        # przez 200 lub (jest podzielny przez 100 i mniejszy niż 1000). Save halfway
        if epoch % 200 == 0 or (epoch % 100 == 0 and epoch < 1000):
            # Przypisywanie wyników funkcji getPieceSegment(pieces) do zmiennych xIpt i xOpt. Funkcja numpy.array
            # jest zastosowana, aby przekonwertować wyniki na tablice numpy.
            xIpt, xOpt = map(numpy.array, getPieceSegment(pieces))
            # Konwertuje wyniki na format MIDI i zapisuje je do pliku w folderze "output".
            noteStateMatrixToMidi(
                numpy.concatenate(
                    (
                        numpy.expand_dims(xOpt[0], 0),
                        # BATCH_LENGTH*2, oznacza średnią liczbę sekund w których zostanie wykonana predykcja,
                        # czyli długość fragmentu muzycznego do ewaluacji.
                        model.predict_fun(BATCH_LENGTH * 2, 1, xIpt[0]),
                    ),
                    axis=0,
                ),
                f"output/sample_{epoch}_{date}_{config.training_base}"
            )
            # Zapisuje stan modelu (model.learned_config) do pliku w folderze "output" pod nazwą zależną od numeru
            # epoki.
            with open(f"output/params_{epoch}.pkl", "wb") as saved_state:
                dill.dump(model.learned_config, saved_state)

    # Przywracanie oryginalnej obsługę sygnału przerwania (Ctrl+C), aby można było poprawnie obsłużyć zakończenie
    # treningu.
    signal.signal(signal.SIGINT, old_handler)
