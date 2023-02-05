# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2022

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


def plot2DSet(desc, labels):
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    negatifs = desc[labels == -1]
    positifs = desc[labels == +1]
    plt.scatter(negatifs[:, 0], negatifs[:, 1], marker='o', color="red")
    plt.scatter(positifs[:, 0], positifs[:, 1], marker='x', color="blue")

# ------------------------


def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax = desc_set.max(0)
    mmin = desc_set.min(0)
    x1grid, x2grid = np.meshgrid(np.linspace(
        mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))
    grid = np.hstack((x1grid.reshape(x1grid.size, 1),
                     x2grid.reshape(x2grid.size, 1)))

    # calcul de la prediction pour chaque point de la grille
    res = np.array([classifier.predict(grid[i, :]) for i in range(len(grid))])
    res = res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid, x2grid, res, colors=[
                 "darksalmon", "skyblue"], levels=[-1000, 0, 1000])

# ------------------------


def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    desc = np.random.uniform(binf, bsup, (2*n, p))
    label = np.asarray([-1] * n + [1] * n)
    return (desc, label)

# ------------------------


def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    desc = []

    for _ in range(nb_points):
        desc.append(np.random.multivariate_normal(
            negative_center, negative_sigma))

    for _ in range(nb_points):
        desc.append(np.random.multivariate_normal(
            positive_center, positive_sigma))

    desc = np.asarray(desc)
    label = np.asarray([-1] * nb_points + [1] * nb_points)
    return (desc, label)

# ------------------------


def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    one = var * np.random.randn(n, 2) + [0, 0]
    two = var * np.random.randn(n, 2) + [1, 1]
    three = var * np.random.randn(n, 2) + [0, 1]
    four = var * np.random.randn(n, 2) + [1, 0]

    desc = np.vstack((one, two, three, four))
    label = np.hstack(-1 * np.ones(2*n), np.ones(2*n))

    return (desc, label)

# ------------------------


def plot_frontiere_V3(desc_set, label_set, w, kernel, step=30, forme=1, fname="out/tmp.pdf"):
    """ desc_set * label_set * array * function * int * int * str -> NoneType
        Note: le classifieur linéaire est donné sous la forme d'un vecteur de poids pour plus de flexibilité
    """
    # -----------
    # ETAPE 1: construction d'une grille de points sur tout l'espace défini par les points du jeu de données
    mmax = desc_set.max(0)
    mmin = desc_set.min(0)
    x1grid, x2grid = np.meshgrid(np.linspace(
        mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))
    grid = np.hstack((x1grid.reshape(x1grid.size, 1),
                     x2grid.reshape(x2grid.size, 1)))

    # -----------
    # ETAPE 2: calcul de la prediction pour chaque point de la grille
    res = np.array([kernel(grid[i, :])@w for i in range(len(grid))])
    # pour les affichages avancés, chaque dimension est présentée sous la forme d'une matrice
    res = res.reshape(x1grid.shape)

    # -----------
    # ETAPE 3: le tracé
    #
    # CHOIX A TESTER en décommentant:
    # 1. lignes de contours + niveaux

    if forme <= 0 or forme >= 4:
        print("Forme must be between 1 and 4")
        return

    if forme <= 2:
        fig, ax = plt.subplots()  # pour 1 et 2
        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
    if forme == 1:
        CS = ax.contour(x1grid, x2grid, res)
        ax.clabel(CS, inline=1, fontsize=10)
    #
    # 2. lignes de contour 0 = frontière
    if forme == 2:
        CS = ax.contour(x1grid, x2grid, res, levels=[0], colors='k')
    #
    # 3. fonction de décision 3D
    if forme == 3 or forme == 4:
        fig = plt.gcf()
        ax = fig.gca(projection='3d')  # pour 3 et 4
        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
        ax.set_zlabel('f(X)')
    #
    if forme == 3:
        surf = ax.plot_surface(x1grid, x2grid, res, cmap=cm.coolwarm)
    #
    # 4. fonction de décision 3D contour grid + transparence
    if forme == 4:
        norm = plt.Normalize(res.min(), res.max())
        colors = cm.coolwarm(norm(res))
        rcount, ccount, _ = colors.shape
        surf = ax.plot_surface(x1grid, x2grid, res, rcount=rcount,
                               ccount=ccount, facecolors=colors, shade=False)
        surf.set_facecolor((0, 0, 0, 0))

    # -----------
    # ETAPE 4: ajout des points
    # Ensemble des exemples de classe -1
    negatifs = desc_set[label_set == -1]
    positifs = desc_set[label_set == +1]     # +1
    # Affichage de l'ensemble des exemples en 2D:
    if forme <= 2:
        ax.scatter(negatifs[:, 0], negatifs[:, 1],
                   marker='o', c='b')  # 'o' pour la classe -1
        ax.scatter(positifs[:, 0], positifs[:, 1],
                   marker='x', c='r')  # 'x' pour la classe +1
    else:
        # on peut ajouter une 3ème dimension si on veut pour 3 et 4
        ax.scatter(negatifs[:, 0], negatifs[:, 1], -1,
                   marker='o', c='b')  # 'o' pour la classe -1
        ax.scatter(positifs[:, 0], positifs[:, 1], 1,
                   marker='x', c='r')  # 'x' pour la classe +1

    # -----------
    # ETAPE 5 en 3D: régler le point de vue caméra:
    if forme == 3 or forme == 4:
        ax.view_init(20, 70)  # a régler en fonction des données

    # -----------
    # ETAPE 6: sauvegarde (le nom du fichier a été fourni en argument)
    if fname != None:
        # avec les options pour réduires les marges et mettre le fond transprent
        plt.savefig(fname, bbox_inches='tight', transparent=True, pad_inches=0)
