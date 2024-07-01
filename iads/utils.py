

# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2024

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.utils as sk

# ------------------------ 

def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data1_desc = np.random.uniform(binf,bsup,(2*n,p))

    data1_label = np.asarray([-1 for i in range(0,n)] + [+1 for i in range(0,n)])

    return (data1_desc,data1_label)
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    n_a=np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
    p_a=np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
    data_a=np.concatenate([n_a,p_a])
    data_label = np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])

    return data_a,data_label
    
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    #TODO: A Compléter    
    # Extraction des exemples de classe -1:
    data_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data_positifs = desc[labels == +1]
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1
    
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
    
  
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    desc1,label1 = genere_dataset_gaussian(np.array([1,1]),np.array([[var,0],[0,var]]),np.array([0,0]),np.array([[var,0],[0,var]]),n)
    desc2,label2 = genere_dataset_gaussian(np.array([0,1]),np.array([[var,0],[0,var]]),np.array([1,0]),np.array([[var,0],[0,var]]),n)
    return np.concatenate((desc1,desc2),axis=0),np.asarray([-1]*len(desc1)+[1]*len(desc2))
    
def calculCout(w):
  c=0
  for i in range(len(X)):
    fx = np.dot(X[i],w)
    if (1-fx*Y[i]) > 0 :
      c=c+(1-fx*Y[i])
  return c


def evolCout(allw):
  cout=[]
  for i in range(len(allw)):
    w=allw[i].copy()
    couti = calculCout(w)
    cout.append(couti)
  return cout
  
  
def crossval(X, Y, n_iterations, iteration):
    #############
    # A COMPLETER
    Xtest = X[iteration * len(X) // n_iterations : ((iteration + 1)*len(X)// n_iterations)]
    Ytest = Y[iteration * len(Y) // n_iterations : ((iteration + 1)*len(Y)// n_iterations)]

    Xapp1 = X[:iteration * len(X) // n_iterations ] 
    Xapp2=  X[((iteration + 1)*len(X)// n_iterations):]

    Xapp = np.concatenate(( np.array(Xapp1) , np.array(Xapp2)))

    Yapp1 = Y[:iteration * len(X) // n_iterations ] 
    Yapp2=  Y[((iteration + 1)*len(X)// n_iterations):]


    Yapp = np.concatenate(( np.array(Yapp1) , np.array(Yapp2)))

       
    #############    
    return Xapp, Yapp,Xtest, Ytest
    

def crossval_strat(X, Y, n_iterations, iteration):
    #############
    # A COMPLETER
    #############  
    XtrainP , YtrainP , XtestP , YtestP = crossval(X[Y==1],Y[Y==1],n_iterations,iteration)
    XtrainN , YtrainN , XtestN , YtestN = crossval(X[Y==-1],Y[Y==-1],n_iterations,iteration)
    Xapp = np.concatenate((XtrainN,XtrainP))
    Yapp = np.concatenate((YtrainN,YtrainP))
    Xtest = np.concatenate((XtestN,XtestP))
    Ytest = np.concatenate((YtestN,YtestP))
    
    # Xapp,Yapp = np.delete(X,np.s_[debut:fin]), np.delete(Y,np.s_[debut:fin])

    return Xapp, Yapp, Xtest, Ytest
    
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L),np.var(L)    


def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """ permet de générer une base d'apprentissage et une base de test
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
        n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
        Hypothèses: 
           - desc_set et label_set ont le même nombre de lignes)
           - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """
    pos = [desc_set[i] for i in range(len(desc_set)) if label_set[i]>=0]
    neg = [desc_set[i] for i in range(len(desc_set)) if label_set[i]<0]
    pos1 = random.sample(pos, n_pos)
    neg1 = random.sample(neg, n_neg)
    for i in pos1:
        for j in range(len(pos)):
            if (i==pos[j]).all():
                pos.pop(j)
                break
    for i in neg1:
        for j in range(len(neg)):
            if (i==neg[j]).all():
                neg.pop(j)
                break
    desc_train = np.asarray(pos1+neg1)
    label_train = np.asarray([1]*n_pos+[-1]*n_neg)
    desc_test = np.asarray(pos+neg)
    label_test = np.asarray([1]*len(pos)+[-1]*len(neg))
    return (desc_train,label_train),(desc_test,label_test)
    
