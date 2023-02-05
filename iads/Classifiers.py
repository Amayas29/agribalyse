# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2022


import random
import numpy as np
import pandas as pd
import copy
import math
import sys
import graphviz as gv


class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        if len(desc_set) == 0:
            return 1.

        succes = 0
        for row, label in zip(desc_set, label_set):
            if self.predict(row) == label:
                succes += 1

        return succes / len(desc_set)


class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.k = k
        self.input_dimension = input_dimension
        self.desc_set = None
        self.label_set = None

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist_points = np.asarray([self.distance(x, p) for p in self.desc_set])
        index_sorted = np.argsort(dist_points)

        ik_points = index_sorted[0:self.k]
        n = 0

        for i in ik_points:
            if self.label_set[i] == 1.:
                n += 1

        p = n / self.k
        return 2 * (p - 0.5)

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        s = self.score(x)
        return 1 if s >= 0 else -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set

    def distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))


class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        w = np.random.uniform(-1, 1, input_dimension)
        self.w = w / np.linalg.norm(w)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        print("Pas d'apprentissage pour ce classifieur")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        s = self.score(x)
        return 1 if s >= 0 else -1


class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """

    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.w = np.zeros(input_dimension) if init == 0 else 0.001 * \
            (2 * np.random.uniform(0, 1, input_dimension) - 1)
        self.norms = []

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        index = np.arange(len(desc_set))
        np.random.shuffle(index)

        for i in index:
            if self.predict(desc_set[i]) == label_set[i]:
                continue

            self.w = self.w + self.learning_rate * label_set[i] * desc_set[i]

    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence

                # - liste des valeurs de norme de différences
        """
        self.norms = []
        nb_iterations = 0
        diff = seuil
        old_w = None
        while nb_iterations < niter_max and diff >= seuil:
            old_w = self.w.copy()
            self.train_step(desc_set, label_set)
            nb_iterations += 1
            diff = np.linalg.norm(np.abs(self.w - old_w))
            self.norms.append(diff)

        return self.norms

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))


class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt kernelisé
    """

    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.noyau = noyau
        self.learning_rate = learning_rate
        dim = noyau.get_output_dim()
        self.w = np.zeros(dim) if init == 0 else 0.001 * \
            (2 * np.random.uniform(0, 1, dim) - 1)
        self.norms = []

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        desc_set = self.noyau.transform(desc_set)

        index = np.arange(len(desc_set))
        np.random.shuffle(index)

        for i in index:
            if self.predict(desc_set[i]) == label_set[i]:
                continue

            self.w = self.w + self.learning_rate * label_set[i] * desc_set[i]

    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """

        self.norms = []
        nb_iterations = 0
        diff = seuil
        old_w = None
        while nb_iterations < niter_max and diff >= seuil:
            old_w = self.w.copy()
            self.train_step(desc_set, label_set)
            nb_iterations += 1
            diff = np.linalg.norm(np.abs(self.w - old_w))
            self.norms.append(diff)

        return self.norms

    def score(self, x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        if len(x) == self.input_dimension:
            x = np.asarray([x])
            x = self.noyau.transform(x)[0]

        return np.dot(self.w, x)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description (dans l'espace originel)
        """
        return np.sign(self.score(x))


class ClassifierPerceptronBiais(Classifier):

    def __init__(self, input_dimension, learning_rate, init=0):
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.allw = []
        self.w = np.zeros(input_dimension) if init == 0 else 0.001 * \
            (2 * np.random.uniform(0, 1, input_dimension) - 1)

    def get_allw(self):
        return self.allw

    def train_step(self, desc_set, label_set):
        index = np.arange(len(desc_set))
        np.random.shuffle(index)

        for i in index:
            fxi = self.score(desc_set[i])
            if fxi * label_set[i] >= 1:
                continue

            self.w = self.w + self.learning_rate * \
                (label_set[i] - fxi) * desc_set[i]

            self.allw.append(self.w.copy())

    def train(self, desc_set, label_set, niter_max=2000, seuil=0.001):
        self.allw = []

        nb_iterations = 0
        diff = seuil
        old_w = None
        norms = []

        while nb_iterations < niter_max and diff >= seuil:
            old_w = self.w.copy()
            self.train_step(desc_set, label_set)
            nb_iterations += 1
            diff = np.linalg.norm(np.abs(self.w - old_w))
            norms.append(diff)

        return norms

    def score(self, x):
        return np.dot(self.w, x)

    def predict(self, x):
        return np.sign(self.score(x))


class ClassifierMultiKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.k = k
        self.input_dimension = input_dimension
        self.desc_set = None
        self.label_set = None

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist_points = np.asarray([self.distance(x, p) for p in self.desc_set])
        index_sorted = np.argsort(dist_points)

        ik_points = index_sorted[0:self.k]

        classes = [0 for _ in np.unique(self.label_set)]
        for i in ik_points:
            classes[self.label_set[i]] += 1

        return classes.index(max(classes))

    def predict(self, x):
        return self.score(x)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set

    def distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))


class ClassifierMultiOAA(Classifier):

    def __init__(self, classifer):
        self.classifier = classifer
        self.classifiers = []

    def train(self, desc_set, label_set):

        classes = np.unique(label_set)
        nCl = len(classes)

        self.classifiers = [copy.deepcopy(self.classifier) for _ in range(nCl)]

        def refresh_y(y, c): return 1 if y == c else -1
        refresh_y = np.vectorize(refresh_y)

        for i in range(nCl):
            Y = refresh_y(label_set, classes[i])
            self.classifiers[i].train(desc_set, Y)

    def score(self, x):
        return [c.score(x) for c in self.classifiers]

    def predict(self, x):
        return np.argmax(self.score(x))

    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()


class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """

    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max
        self.w = 2 * np.random.uniform(0, 1, input_dimension) - 1
        self.allw = []

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        n_iter = 0

        while n_iter <= self.niter_max:
            index = np.arange(len(desc_set))
            np.random.shuffle(index)

            for i in index:
                xi = desc_set[i].reshape((1, self.input_dimension))

                delta_wc = xi.T @ (xi @ self.w - label_set[i])
                self.w = self.w - self.learning_rate * delta_wc

                if self.history:
                    self.allw.append(self.w)

            n_iter += 1

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))


class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.w = None

    def train(self, desc_set, label_set):
        """
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        a = desc_set.T @ desc_set
        b = desc_set.T @ label_set
        self.w = np.linalg.solve(a, b)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))


def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    values, counts = np.unique(Y, return_counts=True)
    max_i = np.argmax(counts)

    return values[max_i]


def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    hs = 0.

    for pi in P:
        if pi != 0:
            hs += pi * math.log(pi)

    return -1 * hs


def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """

    values, counts = np.unique(Y, return_counts=True)
    len_Y = len(Y)

    P = [c/len_Y for c in counts]
    return shannon(P)


def construit_AD(X, Y, epsilon, LNoms=[]):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """

    # dimensions de X:
    (nb_lig, nb_col) = X.shape

    entropie_classe = entropie(Y)

    if (entropie_classe <= epsilon) or (nb_lig <= 1):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        # meilleur gain trouvé (initalisé à -infinie)
        gain_max = sys.float_info.min
        i_best = -1         # numéro du meilleur attribut
        Xbest_valeurs = None

        #############

        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.

        for j in range(nb_col):

            vj, vj_counts = np.unique(X[:, j], return_counts=True)
            sum_vj = sum(vj_counts)

            hs_y_xj = 0

            for vjl, vjl_c in zip(vj, vj_counts):
                p_vjl = vjl_c / sum_vj

                Y_vjl = Y[X[:, j] == vjl]

                hs_y_xj += p_vjl * entropie(Y_vjl)

            gain = entropie_classe - hs_y_xj
            if gain_max < gain:
                gain_max = gain
                i_best = j
                Xbest_valeurs = vj

        #############

        if len(LNoms) > 0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best, LNoms[i_best])
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v, construit_AD(
                X[X[:, i_best] == v], Y[X[:, i_best] == v], epsilon, LNoms))
    return noeud


class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe = None       # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.

    def ajoute_feuille(self, classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe = classe
        self.Les_fils = None   # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ', self.nom_attribut,
                  ' -> Valeur inconnue: ', exemple[self.attribut])
            return 0

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe, str(self.classe), shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i = 0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g, prefixe+str(i))
                g.edge(prefixe, prefixe+str(i), valeur)
                i = i+1
        return g


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """

    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None

    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.racine = construit_AD(
            desc_set, label_set, self.epsilon, self.LNoms)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass

    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok = 0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i, :]) == label_set[i]:
                nb_ok = nb_ok+1
        acc = nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self, GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:, num_col])

    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([], []))

    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')

    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []

    nb_exemples = len(m_class)

    for v in l_valeurs:
        cl_inf = m_class[m_desc[:, num_col] <= v]
        cl_sup = m_class[m_desc[:, num_col] > v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)

        # calcul de l'entropie de la coupure
        # entropie de l'ensemble des inf
        val_entropie_inf = entropie(cl_inf)
        # entropie de l'ensemble des sup
        val_entropie_sup = entropie(cl_sup)

        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
            + (nb_sup / float(nb_exemples)) * val_entropie_sup

        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)

        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v

    return (best_seuil, best_entropie), (liste_coupures, liste_entropies)


def partitionne(mdesc, mclass, num_col, s):

    inf = mdesc[:, num_col] <= s
    sup = mdesc[:, num_col] > s

    desc_inf, label_inf = mdesc[inf], mclass[inf]
    desc_sup, label_sup = mdesc[sup], mclass[sup]
    return ((desc_inf, label_inf), (desc_sup, label_sup))


class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe = None       # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup

    def ajoute_feuille(self, classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe = classe
        self.Les_fils = None   # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe

        if exemple[self.attribut] > self.seuil:
            return self.Les_fils['sup'].classifie(exemple)

        return self.Les_fils['inf'].classifie(exemple)

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe, str(self.classe), shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g, prefixe+"g")
            self.Les_fils['sup'].to_graph(g, prefixe+"d")
            g.edge(prefixe, prefixe+"g", '<=' + str(self.seuil))
            g.edge(prefixe, prefixe+"d", '>' + str(self.seuil))
        return g


def construit_AD_num(X, Y, epsilon, LNoms=[]):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """

    # dimensions de X:
    (nb_lig, nb_col) = X.shape

    entropie_classe = entropie(Y)

    if (entropie_classe <= epsilon) or (nb_lig <= 1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        # numéro du meilleur attribut (init à -1 (aucun))
        i_best = -1
        Xbest_seuil = None

        #############

        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))

        def refresh_x(x, s): return "inf" if x <= s else "sup"
        refresh_x = np.vectorize(refresh_x)

        for j in range(nb_col):

            (seuil, _), (_, _) = discretise(X, Y, j)

            if seuil is None:
                continue

            X_cat = refresh_x(X[:, j], seuil)
            vj, vj_counts = np.unique(X_cat, return_counts=True)
            sum_vj = sum(vj_counts)

            hs_y_xj = 0

            for vjl, vjl_c in zip(vj, vj_counts):
                p_vjl = vjl_c / sum_vj

                Y_vjl = Y[X_cat == vjl]

                hs_y_xj += p_vjl * entropie(Y_vjl)

            gain = entropie_classe - hs_y_xj
            if gain_max < gain:
                gain_max = gain
                i_best = j
                Xbest_seuil = seuil

        Xbest_tuple = ((X, Y), (None, None))
        if Xbest_seuil is not None:
            Xbest_tuple = partitionne(X, Y, i_best, Xbest_seuil)

        ############
        if (gain_max != float('-Inf')):
            if len(LNoms) > 0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best, LNoms[i_best])
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data, left_class), (right_data, right_class)) = Xbest_tuple
            noeud.ajoute_fils(Xbest_seuil,
                              construit_AD_num(
                                  left_data, left_class, epsilon, LNoms),
                              construit_AD_num(right_data, right_class, epsilon, LNoms))
        else:  # aucun attribut n'a pu améliorer le gain d'information
            # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1, "Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))

    return noeud


class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """

    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None

    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.racine = construit_AD_num(
            desc_set, label_set, self.epsilon, self.LNoms)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass

    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok = 0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i, :]) == label_set[i]:
                nb_ok = nb_ok+1
        acc = nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self, GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


def tirage(VX, m, r):
    if not r:
        return random.sample(list(VX), m)

    L = []
    for i in range(0, m):
        L.append(random.choice(VX))

    return L


def echantillonLS(labeledSet, m, r):
    elems = tirage(np.arange(len(labeledSet[0])), m, r)

    return (labeledSet[0][elems, :], labeledSet[1][elems])


class ClassifierBaggingTree(Classifier):

    def __init__(self, B, per, seuil, r):
        self.B = B
        self.per = per
        self.seuil = seuil
        self.r = r
        self.trees = []
        self.labels = []

    def train(self, labeledSet):

        dim = labeledSet[0].shape[1]
        m = math.floor(self.per * labeledSet[0].shape[0])
        for _ in range(self.B):
            labeledSet = echantillonLS(labeledSet, m, self.r)
            tree = ClassifierArbreNumerique(dim, self.seuil)
            tree.train(labeledSet[0], labeledSet[1])
            self.trees.append(tree)

    def score(self, x):
        pass

    def predict(self, x):
        predictions = [tr.predict(x) for tr in self.trees]
        classes, counts = np.unique(predictions, return_counts=True)
        return classes[np.argmax(counts)]


def construit_AD_aleatoire(LS, epsilon, nb_att):
    X = LS[0]
    Y = LS[1]

    (nb_lig, nb_col) = X.shape

    entropie_classe = entropie(Y)

    if (entropie_classe <= epsilon) or (nb_lig <= 1):
        noeud = NoeudNumerique(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')
        i_best = -1
        Xbest_seuil = None

        def refresh_x(x, s): return "inf" if x <= s else "sup"
        refresh_x = np.vectorize(refresh_x)

        atts = np.arange(nb_col)
        atts = random.sample(list(atts), nb_att)

        for j in atts:

            (seuil, _), (_, _) = discretise(X, Y, j)

            if seuil is None:
                continue

            X_cat = refresh_x(X[:, j], seuil)
            vj, vj_counts = np.unique(X_cat, return_counts=True)
            sum_vj = sum(vj_counts)

            hs_y_xj = 0

            for vjl, vjl_c in zip(vj, vj_counts):
                p_vjl = vjl_c / sum_vj

                Y_vjl = Y[X_cat == vjl]

                hs_y_xj += p_vjl * entropie(Y_vjl)

            gain = entropie_classe - hs_y_xj
            if gain_max < gain:
                gain_max = gain
                i_best = j
                Xbest_seuil = seuil

        Xbest_tuple = ((X, Y), (None, None))
        if Xbest_seuil is not None:
            Xbest_tuple = partitionne(X, Y, i_best, Xbest_seuil)

        if (gain_max != float('-Inf')):
            noeud = NoeudNumerique(i_best)
            ((left_data, left_class), (right_data, right_class)) = Xbest_tuple
            noeud.ajoute_fils(Xbest_seuil,
                              construit_AD_aleatoire(
                                  (left_data, left_class), epsilon, nb_att),
                              construit_AD_aleatoire((right_data, right_class), epsilon, nb_att))
        else:
            noeud = NoeudNumerique(-1, "Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))

    return noeud


class ClassifierRandomForest(ClassifierArbreDecision):

    def __init__(self, B, nb_att, epsilon):
        super().__init__(nb_att, epsilon)
        self.nb_att = nb_att
        self.B = B

    def train(self, labeledSet):
        self.racine = []

        for _ in range(self.B):
            self.racine.append(construit_AD_aleatoire(
                labeledSet, self.epsilon, self.nb_att))

    def predict(self, x):
        predictions = [tr.classifie(x) for tr in self.racine]
        classes, counts = np.unique(predictions, return_counts=True)
        return classes[np.argmax(counts)]

    def affiche(self, GTree):
        pass
