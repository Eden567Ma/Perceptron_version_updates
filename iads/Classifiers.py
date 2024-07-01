# -*- coding: utf-8 -*-


# Classfieurs implémentés en LU3IN026

# Import de packages externes

import numpy as np
import pandas as pd

# ---------------------------

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
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        # COMPLETER CETTE FONCTION ICI : 
        label_predict = []
        for i in range(len(desc_set)):
          label_predict.append(self.predict(desc_set[i]))
        
        accuracy= np.mean(np.array(label_predict)==np.array(label_set))
        return accuracy
        
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension=input_dimension
        self.learning_rate=learning_rate
        if(init):
          self.w=np.zeros(input_dimension)
        else:
          v=np.random.uniform(0,1,input_dimension)
          v=(2*v-1)*0.001
          self.w=v.copy()
        self.allw = [self.w.copy()]
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        index_list =[i for i in range(len(desc_set))]
        np.random.shuffle(index_list)

        for i in index_list :
          xi = desc_set[i]
          yHut = np.dot(xi,self.w)
          yi = label_set[i]
          if not (yHut<0 and label_set[i]==-1) and not(yHut>0 and label_set[i]==1) :
            self.w =self.w + np.dot(self.learning_rate*yi,xi)
            self.allw.append(self.w.copy())
        
     
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """    
        diff_list = []
        i = 0 

        w0= self.w.copy()
        self.train_step(desc_set , label_set ) 
        norme = np.linalg.norm(w0-self.w)
        diff_list.append(norme)
        
        while ( i != nb_max and norme > seuil) :
          i += 1
          w0 = self.w.copy()
          self.train_step(desc_set , label_set ) 
          norme = np.linalg.norm(w0-self.w)
          diff_list.append(norme)

        return diff_list  
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        y_chap=self.score(x)
        if(y_chap<=0):
          return -1
        else:
          return 1

    def get_allw(self):
      return self.allw
          

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
        super().__init__(input_dimension)
        self.k = k 
       
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dst=[]
        for i in range(len(self.desc_set)):
          dst.append(distance.euclidean(self.desc_set[i],x))
        dst_arranger=np.argsort(dst)
        cpt=0
        for i in range(self.k):
          if(self.label_set[dst_arranger[i]]==1):
            cpt+=1
        p=cpt/self.k
        return 2*(p-0.5)
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if(self.score(x)>0):
          return 1
        else:
          return -1
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """    
        self.desc_set=desc_set
        self.label_set=label_set   
        
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.poids = np.random.uniform(-1, 1, input_dimension)
        self.poids /= np.linalg.norm(self.poids)

        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.poids)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        return 1 if score >= 0 else -1
        
class ClassifierKNN_MC(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension, k,nb_class):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.k = k
        self.nb_class = nb_class
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        
        dst = np.asarray([distance.euclidean(x,y) for y in self.desc_set])
        dstarranger = np.argsort(dst) 
        
        classes = self.label_set[dstarranger[:self.k]]   
        uniques, counts = np.unique(classes, return_counts=True)    
        
        return uniques[np.argmax(counts)]/self.nb_class
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return self.score(x)*self.nb_class

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """       
        self.desc_set=desc_set
        self.label_set=label_set
        
class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        ### A COMPLETER !
        index_list =[i for i in range(len(desc_set))]
        np.random.shuffle(index_list)

        for i in index_list :
            xi = desc_set[i]
            yi = label_set[i]
            if ((self.score(xi) * yi ) < 1 ):
                self.w =self.w + ((yi - self.score(xi)) * xi) * self.learning_rate
                self.allw.append(self.w.copy())