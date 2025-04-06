# Backpropagation - Implementacja

Prosta implementacja sieci neuronowej z algorytmem propagacji wstecznej do klasyfikacji cyfr z zestawu danych MNIST. Projekt zawiera różne warianty implementacji algorytmu uczenia.

## Struktura projektu

-   `back_propagation.py`: Główny skrypt implementujący sieć neuronową z uczeniem metodą stochastycznego spadku gradientu (SGD)
-   `back_propagation_soft_max.py`: Wariant sieci wykorzystujący funkcję aktywacji Softmax i funkcję kosztu entropii krzyżowej w ostatniej warstwie
-   `back_propagation_mini_batch.py`: Wariant sieci wykorzystujący uczenie za pomocą mini-batchy
-   `mnist_loader.py`: Skrypt pomocniczy do ładowania i przygotowywania danych MNIST
-   `mnist.pkl.gz`: Plik danych MNIST

