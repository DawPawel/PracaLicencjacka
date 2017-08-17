import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import cross_validation as cv
from sklearn.mixture import GaussianMixture as gm
from multiprocessing import Pool
from sklearn.cluster import KMeans

minimum = 100000
maksimum = 0
for wiersz in X:
    bid = wiersz.bid[0, 0]
    ask = wiersz.ask[-1, 0]
    if(ask > maksimum):
        maksimum = ask
    if(bid < minimum):
        minimum = bid 
liczbaWspolrzednych = int((maksimum - minimum) * 10 + 1)

param_grid = {'C': [1, 1e1, 1e2, 1e3, 1e4],
              'gamma': [0.0001, 0.001, 0.01, 0.1], 
             }
svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, n_jobs = 6)

def normalizuj(tab):
    tabMean = np.mean(tab, axis = 0)
    tabStd = np.std(tab, axis = 0)
    tabStd[(tabStd == 0)] = 1
    return ((tab - tabMean) / tabStd)

def calkaBezDzielenia(a, b, gmm, eps):
    x = np.arange(a, b + eps, eps)
    y = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    return np.sum(y[:n-1] + y[1:])


def calka(a, b, gmm, eps):
    n = (int) ((b - a) / eps)
    return calkaBezDzielenia(a, b, gmm, eps)*(b - a) / n / 2

def obieCalki(a, b, gmm, eps):
    n = (int) ((b - a) / eps)
    wynik = calkaBezDzielenia(a, b, gmm, eps)
    return wynik, wynik*(b-a)/n/2

class LOB:
    def __init__(self, wiersz):
        self.zmiana = 0
        self.dzien = wiersz[0]
        self.czas = wiersz[1]
        i = 3
        while (wiersz[i] != 'ASK'):
            i += 1
        bid = wiersz[3:i]
        ask = wiersz[(i+1):]
        self.bid = np.zeros((int((len(bid) / 2)), 2))
        self.ask = np.zeros((int((len(ask) / 2)), 2))
        for i in range(0, len(bid), 2):
            self.bid[int(i / 2)][0] = float(bid[i]) 
            self.bid[int(i / 2)][1] = float(bid[i+1])
        for i in range(0, len(ask), 2):
            self.ask[int(i / 2)] = [float(ask[i]), float(ask[i+1])]
        self.midPrice = (self.bid[len(self.bid) - 1][0] + self.ask[0][0]) / 2
    def __str__(self):
        return "Dzień: " + self.dzien + ", Czas: " + self.czas + ", midPrice: " + str(self.midPrice)
    
# ustawianie wartości zmian i tworzenie dla nich osobnego wektora
def generujZmiane(X):
    for i in range(len(X) - 1):
        if (X[i].midPrice > X[i + 1].midPrice):
            X[i].zmiana = 1
        if (X[i].midPrice < X[i + 1].midPrice):
            X[i].zmiana = -1
        if (X[i].midPrice == X[i + 1].midPrice):
            X[i].zmiana = 0
    T = generujIndeksy(X)
    return np.array([i.zmiana for i in X])[T]

# interesują nas te loby, po którch nastąpiła zmiana mid price
def generujIndeksy(X):
    return np.arange(len(X))[np.array([(i.zmiana != 0) for i in X])]

# pierwszy sposób - imbalance to różnica najlepszego bid'a i ask'a przez ich sumę
def generujImbalance1(X):
    T = generujIndeksy(X)
    i = 0
    Imbalance1 = np.zeros((T.shape[0], 1))
    for wiersz in X[T]:
        bid = wiersz.bid[-1, 1]
        ask = wiersz.ask[0, 1]
        Imbalance1[i] = (bid - ask) / (bid + ask)
        i += 1
    return Imbalance1

def generujPrzedzialy(X, tablica, k, il):
    wynik = np.zeros((len(X), 2 * il))
    i = 0
    for wiersz in X:
        pozMidPrice = int((wiersz.midPrice - minimum) * 10)
        for j in range(il):
            if (pozMidPrice - (j+1) * k >= 0):
                wynik[i][2 * j] = np.sum(tablica[i][(pozMidPrice - (j+1) * k): (pozMidPrice - j * k)])
            else:
                wynik[i][2 * j] = 0
            if (pozMidPrice + (j+1) * k < tablica.shape[1]):
                wynik[i][2 * j + 1] = np.sum(tablica[i][(pozMidPrice + j * k): (pozMidPrice + (j+1) * k)])
            else:
                wynik[i][2 * j + 1] = 0
        i += 1
    return wynik

def testujDecisionFunction(xTest, yTest, najDecFun, clf):
    wynik = clf.decision_function(xTest).reshape(yTest.shape)
    yTestWiekszeOdZera = yTest > 0
    if (najDecFun[1]):
        return np.sum((wynik > najDecFun[0]) == yTestWiekszeOdZera) / yTest.shape[0]
    return np.sum((wynik < najDecFun[0]) == yTestWiekszeOdZera) / yTest.shape[0]

def najlepszeDecFun(yTrain, wynikDecFun):
    yTrainWiekszeOdZera = yTrain > 0
    tabWynikow = []
    for i in range(wynikDecFun.shape[0]):
        wy = np.sum((wynikDecFun < wynikDecFun[i]) == yTrainWiekszeOdZera) / yTrain.shape[0]
        tabWynikow.append(wy)
        wy = np.sum((wynikDecFun > wynikDecFun[i]) == yTrainWiekszeOdZera) / yTrain.shape[0]
        tabWynikow.append(wy)
    pozycjaMaxArg = np.argmax(tabWynikow)
    if (pozycjaMaxArg % 2 == 0):
        return (wynikDecFun[int(pozycjaMaxArg / 2)], 0)
    else:
        return (wynikDecFun[int(pozycjaMaxArg / 2)], 1)

def crossValidation(clf, data, dataClass, k):
    size = data.shape[0]
    arr = np.arange(size)
    np.random.shuffle(arr)
    err = 0
    n = int(size / k)
    for i in range(k):
        mask2 = arr[np.arange(i*n,min((i+1)*n,size))]
        mask1 = arr[np.concatenate((np.arange(i*n),np.arange((i+1)*n,size)))]
        X_train = data[mask1]
        y_train = dataClass[mask1]
        X_test = data[mask2]
        y_test = dataClass[mask2]  
        clf = clf.fit(X_train, y_train)
        if isinstance(clf, GridSearchCV):
            clf = clf.best_estimator_
        y_train = y_train.reshape((y_train.shape[0], 1))
        wynikDecFun = clf.decision_function(X_train).reshape(y_train.shape)
        najDecFun = najlepszeDecFun(y_train, wynikDecFun)        
        err += testujDecisionFunction(X_test, y_test, najDecFun, clf)
    return float(err) / k 

def generujPrzedzialyRosnace(tablica, il):
    wynik = np.zeros((len(tablica), 2 * il))
    for i in range(len(tablica)):
        a = 0
        b = 1
        for where in range(il):
            for j in range(a, b):
                wynik[i][where*2] += tablica[i][2*j]
                wynik[i][where*2 + 1] += tablica[i][2*j + 1]
            wynik[i][where*2] /= (b-a)
            wynik[i][where*2 + 1] /= (b-a)
            a = b
            b += where + 1
    return wynik

def wyliczCViWypisz(nazwa, tablica, mnoznik, liczbaKolumn, iloscIteracji):
    if mnoznik != 1:
        nazwa += '*' + str(mnoznik)
    wynik = crossValidation(svm, (((tablica[indeksy]).T[:liczbaKolumn]).T)*mnoznik, zmiana, iloscIteracji)
    print(nazwa, "zredukowana do ", liczbaKolumn, " kolumn, Cross-Vall score: ", wynik)