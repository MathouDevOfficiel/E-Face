# E-Face

E-Face est une application de reconnaissance faciale locale en Python, basée sur :

- **Tkinter** pour l’interface graphique
- **OpenCV (opencv-contrib)** pour la détection et la reconnaissance de visages
- **LBPH** (Local Binary Patterns Histograms) pour reconnaître les personnes
- Une **base de données locale** de visages (par personne)

## ✨ Fonctionnalités

- 🏠 **Menu principal**
  - ➕ Ajouter un visage
  - 🧠 Entraîner le modèle
  - 👁️ Reconnaissance faciale

- 👤 **Ajouter un visage**
  - Saisie du **nom** (accents autorisés : Éléa, Mathïs, etc.)
  - Liste des personnes déjà enregistrées
  - Caméra en direct avec détection du visage
  - Bouton **Capturer** (enregistrement manuel d’images)
  - Bouton **📡 Scan auto** :
    - Capture automatiquement plusieurs images
    - Guide l’utilisateur : face, gauche, droite, haut, bas
  - Bouton **📂 Importer photos** pour ajouter des visages depuis des fichiers
  - Liste des images pour chaque personne
  - Prévisualisation d’une image + bouton **Supprimer**

- 👁️ **Reconnaissance faciale**
  - Miroir de la caméra
  - Détection de plusieurs visages
  - Affichage du **nom au-dessus de chaque visage**
  - Mode **très strict** (limite les erreurs : beaucoup d’“Inconnu” plutôt qu’une mauvaise personne)
  - Thème clair / sombre qui change en douceur selon la luminosité

## 🧰 Prérequis

- Python 3.12 (conseillé)
- Windows (testé dessus)

### Modules Python

Tout est résumé dans `requirements.txt`, mais en gros :

- `opencv-contrib-python==4.7.0.72`
- `numpy<2.0`
- `Pillow`

## ⚙️ Installation

1. Cloner ou télécharger ce projet.
2. (Optionnel mais recommandé) Créer un environnement virtuel :

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
