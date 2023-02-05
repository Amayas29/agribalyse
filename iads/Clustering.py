import random as rd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import numpy as np
import pandas as pd


def normalisation(df):

    def norm_col(x, xmax, xmin): return (x - xmin) / \
        (xmax - xmin) if xmax != xmin else 0

    norm_col = np.vectorize(norm_col)
    normalized_data = np.zeros(len(df))

    for col_name in df.columns:
        col = df[col_name]
        col_normalized = norm_col(col, np.max(col), np.min(col))
        normalized_data = np.column_stack((normalized_data, col_normalized))

    normalized_data = normalized_data[:, 1:]

    return pd.DataFrame(normalized_data, columns=df.columns)


def dist_euclidienne(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def dist_manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2))


def dist_vect_type(dist_type, x1, x2):
    if dist_type == "euclidienne":
        return dist_euclidienne(x1, x2)
    return dist_manhattan(x1, x2)


def centroide(data):
    return np.mean(data, axis=0)


def dist_centroides(g1, g2, dist_type="euclidienne"):
    return dist_vect_type(dist_type, centroide(g1), centroide(g2))


def dist_complete(g1, g2, dist_type="euclidienne"):
    if type(g1) == pd.DataFrame:
        g1 = g1.to_numpy()

    if type(g2) == pd.DataFrame:
        g2 = g2.to_numpy()

    return np.max([dist_vect_type(dist_type, p1, p2) for p1 in g1 for p2 in g2])


def dist_simple(g1, g2, dist_type="euclidienne"):

    if type(g1) == pd.DataFrame:
        g1 = g1.to_numpy()

    if type(g2) == pd.DataFrame:
        g2 = g2.to_numpy()

    return np.min([dist_vect_type(dist_type, p1, p2) for p1 in g1 for p2 in g2])


def dist_average(g1, g2, dist_type="euclidienne"):

    if type(g1) == pd.DataFrame:
        g1 = g1.to_numpy()

    if type(g2) == pd.DataFrame:
        g2 = g2.to_numpy()

    return np.mean([dist_vect_type(dist_type, p1, p2) for p1 in g1 for p2 in g2])


def initialise(df):
    df_dict = {}

    for i in range(df.shape[0]):
        df_dict[i] = [i]

    return df_dict


def fusionne(df, P0, verbose=False, linkageType="centroid", dist_type="euclidienne"):

    P1 = {}
    k1 = -1
    k2 = -1
    d = float("inf")

    for k_a in P0:
        for k_b in P0:

            if k_a == k_b:
                continue

            g1 = df.iloc[P0[k_a]]
            g2 = df.iloc[P0[k_b]]

            dist = 0

            if linkageType == "centroid":
                dist = dist_centroides(g1, g2, dist_type)
            elif linkageType == "complete":
                dist = dist_complete(g1, g2, dist_type)
            elif linkageType == "simple":
                dist = dist_simple(g1, g2, dist_type)
            else:
                dist = dist_average(g1, g2, dist_type)

            if dist < d:
                d = dist
                k1 = k_a
                k2 = k_b

    for k in P0:
        if k == k1 or k == k2:
            continue

        P1[k] = P0[k]

    P1[max(P0) + 1] = P0[k1] + P0[k2]

    if verbose:
        print(f"Distance mininimale trouvée entre  [{k1}, {k2}]  =  {d}")

    return(P1, k1, k2, d)


def clustering_hierarchique(df, verbose=False, dendrogramme=False, linkageType="centroid", dist_type="euclidienne"):
    result = []

    par = initialise(df)

    while len(par) > 1:
        (n_par, k1, k2, d) = fusionne(df, par, verbose, linkageType, dist_type)
        s = len(par[k1]) + len(par[k2])
        par = n_par
        result.append([k1, k2, d, s])

    if dendrogramme:
        plt.figure(figsize=(30, 15))
        plt.title('Dendrogramme', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        scipy.cluster.hierarchy.dendrogram(
            result,
            leaf_font_size=24.,
        )
        plt.show()

    return result


def clustering_hierarchique_complete(df, verbose=False, dendrogramme=False, dist_type="euclidienne"):
    return clustering_hierarchique(df, verbose, dendrogramme, "complete", dist_type)


def clustering_hierarchique_simple(df, verbose=False, dendrogramme=False, dist_type="euclidienne"):
    return clustering_hierarchique(df, verbose, dendrogramme, "simple", dist_type)


def clustering_hierarchique_average(df, verbose=False, dendrogramme=False, dist_type="euclidienne"):
    return clustering_hierarchique(df, verbose, dendrogramme, "average", dist_type)


def dist_vect(v1, v2):
    return dist_euclidienne(v1, v2)


def inertie_cluster(Ens):
    Ens = np.array(Ens)
    c = centroide(Ens)
    return np.sum([dist_vect(c, p) ** 2 for p in Ens])


def init_kmeans(K, Ens):
    Ens = np.array(Ens)
    indices = np.arange(len(Ens))
    indices = np.random.choice(indices, K, replace=False)
    return Ens[indices]


def plus_proche(Exe, Centres):
    return np.argmin([dist_vect(Exe, centre) for centre in Centres])


def affecte_cluster(Base, Centres):
    matrice_affecation = {}

    for i in range(len(Centres)):
        matrice_affecation[i] = []

    for i in range(len(Base)):
        plus_proche_centre = plus_proche(Base.iloc[i], Centres)
        matrice_affecation[plus_proche_centre].append(i)

    return matrice_affecation


def nouveaux_centroides(Base, U):
    cen = []
    for _, v in U.items():
        ens_exemples = Base.iloc[v].to_numpy()
        cen.append(centroide(ens_exemples))

    return np.array(cen)


def inertie_globale(Base, U):
    return np.sum([inertie_cluster(Base.iloc[U[k]]) for k in U])


def kmoyennes(K, Base, epsilon, iter_max, verbose=True):

    centres = init_kmeans(K, Base)
    old_iner = 0

    for i in range(iter_max):
        mat = affecte_cluster(Base, centres)
        iner = inertie_globale(Base, mat)

        diff = np.abs(old_iner - iner)

        if verbose:
            print(
                f" iteration {i + 1} Inertie : {iner:1.4f} Difference : {diff:1.4f}")

        if diff < epsilon:
            break

        old_iner = iner
        centres = nouveaux_centroides(Base, mat)

    return centres, mat


def affiche_resultat(Base, Centres, Affect):
    couleurs = ['g', 'b', 'y', 'c', 'm']

    plt.scatter(Centres[:, 0], Centres[:, 1], color='r', marker='x')

    if len(Centres) > len(couleurs):
        print("Nombre de couleurs insuffisant pour representer tous les clusters")
        return

    i = 0
    for k in Affect:
        cluster = Base.iloc[Affect[k]]
        plt.scatter(cluster['X'], cluster['Y'], color=couleurs[i])
        i += 1


def distance_globale(Base, clust):
    distance = []

    len_clust = len(clust)

    if len_clust == 1:
        return [0]

    for i in range(len_clust):
        for j in range(i+1, len_clust):
            distance.append(
                dist_vect(Base.iloc[clust[i]], Base.iloc[clust[j]]))

    return distance


def distance_clusters(centres):
    distance = []

    for i in range(len(centres)):
        for j in range(i+1, len(centres)):
            distance.append(dist_vect(centres[i], centres[j]))

    return distance


def index_dunn(Base, affect):
    codist = 0

    iner = inertie_globale(Base, affect)

    if iner == 0:
        return 0

    for _, value in affect.items():

        distance = distance_globale(Base, value)
        codist += max(distance)

    return codist / iner


def index_XieBeni(Base, centres, affect):

    iner = inertie_globale(Base, affect)
    if iner == 0:
        return 0

    sep = min(distance_clusters(centres))
    return sep / iner


def affiche_resultat(Base, Centres, Affect):
    clust = []
    for key, value in Affect.items():
        clust.append(pd.DataFrame([Base.iloc[elem]
                     for elem in value], columns=Base.columns))

    for i in range(len(Centres)):
        indice = rd.randint(1, len(couleurs()))
        vec1 = reduire_dim(clust[i])[2]
        vec2 = reduire_dim(clust[i])[3]
        plt.scatter(np.dot(clust[i], vec1), np.dot(
            clust[i], vec1), color=couleurs()[indice])
        plt.scatter(np.dot(Centres[i], vec2), np.dot(
            Centres[i], vec2), color='r', marker='x')
    plt.show()
