from mnist_loader import load_data_wrapper
from mnist_loader import load_data
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_pochodna(x):
    return sigmoid(x)*(1-sigmoid(x))

class Warstwa:
    def __init__(self, input_rozmiar, output_rozmiar):
        self.input_rozmiar = input_rozmiar
        self.output_rozmiar = output_rozmiar
        self.wagi = np.random.uniform(0, 0.001, (output_rozmiar, input_rozmiar+1))

    def propagacja_w_przod(self, macierz):
        self.X = np.insert(macierz, 0, 1, axis=0)
        self.net = self.wagi @ self.X
        self.a = sigmoid(self.net)
        return self.a
    
    def propagacja_w_tyl(self, blad, wspolczynnik_uczenia):
        sygnal_delta = blad * sigmoid_pochodna(self.net)
        delta_LW = sygnal_delta @ self.X.T
        delta_poprzednia = self.wagi.T @ sygnal_delta
        self.wagi -= wspolczynnik_uczenia * delta_LW
        return delta_poprzednia[1:]

class Siec:
    def __init__(self, rozmiary_warstw):
        self.warstwy = []
        for i in range(len(rozmiary_warstw)-1):
            self.warstwy.append(Warstwa(rozmiary_warstw[i], rozmiary_warstw[i+1]))

    def fit(self, training_data, epoki=5, wspolczynnik_uczenia=0.1):
        for epoka in range(epoki):
            for X, y in training_data:
                for warstwa in self.warstwy:
                    X = warstwa.propagacja_w_przod(X)
                
                blad = X - y
                for warstwa in reversed(self.warstwy):
                    blad = warstwa.propagacja_w_tyl(blad, wspolczynnik_uczenia)
            
            print(f"Epoka nr {epoka+1}")

    def predict(self, X):
        for warstwa in self.warstwy:
            X = warstwa.propagacja_w_przod(X)
        return np.argmax(X)

if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()

    training_data = list(training_data)
    test_data = list(test_data)

    siec2l = Siec([784, 128, 10])
    siec2l.fit(training_data)
        
    odpowiedzi = []
    liczba_poprawnych = 0
    for X, oczekiwana_cyfra in test_data:
        przewidziana_cyfra = siec2l.predict(X)
        if przewidziana_cyfra == oczekiwana_cyfra:
            liczba_poprawnych += 1
        odpowiedzi.append((int(przewidziana_cyfra), int(oczekiwana_cyfra)))

    skutecznosc = liczba_poprawnych/len(test_data) * 100
        
    print("Lista wyników dla danych ze zbioru testowego w formacie (przewidziana cyfra, oczekiwana cyfra)")
    print(odpowiedzi)
    print(f"Skuteczność predykcji {skutecznosc} % - stochastic gradient descent")