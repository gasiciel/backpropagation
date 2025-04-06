from mnist_loader import load_data_wrapper
from mnist_loader import load_data
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_pochodna(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Warstwa:
    def __init__(self, input_rozmiar, output_rozmiar):
        self.input_rozmiar = input_rozmiar
        self.output_rozmiar = output_rozmiar
        limit = np.sqrt(6 / (input_rozmiar + output_rozmiar))
        self.wagi = np.random.uniform(-limit, limit, (output_rozmiar, input_rozmiar + 1))

    
    def propagacja_w_przod(self, macierz):
        self.X = np.insert(macierz, 0, 1, axis=0)
        self.net = self.wagi @ self.X
        self.a = sigmoid(self.net)
        return self.a
    
    def propagacja_w_tyl(self, blad):
        sygnal_delta = blad * sigmoid_pochodna(self.net)
        delta_LW = sygnal_delta @ self.X.T
        delta_poprzednia = self.wagi.T @ sygnal_delta
        return delta_poprzednia[1:], delta_LW

class Siec:
    def __init__(self, rozmiary_warstw):
        self.warstwy = []
        for i in range(len(rozmiary_warstw) - 1):
            self.warstwy.append(Warstwa(rozmiary_warstw[i], rozmiary_warstw[i + 1]))


    def update_mini_batch(self, mini_batch, wspolczynnik_uczenia):
        grad_warstw = [np.zeros_like(warstwa.wagi) for warstwa in self.warstwy]
        for X, y in mini_batch:
            aktywacja = X
            aktywacje = [X]  
            for warstwa in self.warstwy:
                aktywacja = warstwa.propagacja_w_przod(aktywacja)
                aktywacje.append(aktywacja)
            
            blad = aktywacje[-1] - y
            for i in reversed(range(len(self.warstwy))):
                blad, delta_LW = self.warstwy[i].propagacja_w_tyl(blad)
                grad_warstw[i] += delta_LW
        for i, warstwa in enumerate(self.warstwy):
            warstwa.wagi -= wspolczynnik_uczenia * (grad_warstw[i]/len(mini_batch))

    
    def fit(self, training_data, epoki=5, wspolczynnik_uczenia=0.01, batch_rozmiar=8):
        n = len(training_data)
        for epoka in range(epoki):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_rozmiar] for k in range(0, n, batch_rozmiar)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, wspolczynnik_uczenia)

            print(f"Epoka nr {epoka + 1}")
    
    def predict(self, X):
        for warstwa in self.warstwy:
            X = warstwa.propagacja_w_przod(X)
        return np.argmax(X)


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)

    siec2l = Siec([784, 128, 10])
    siec2l.fit(training_data, epoki=5, wspolczynnik_uczenia=0.1, batch_rozmiar=10)

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
    print(f"Skuteczność predykcji {skutecznosc} % - mini-batch gradient descent")
