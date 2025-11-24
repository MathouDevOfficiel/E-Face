
---

## 3. PROJECT_LOG.md – suivi du projet

Dans `E-Face/PROJECT_LOG.md` :

```markdown
# E-Face – Journal de bord / Suivi du projet

## 📅 0.1.0 – Version de base (date à compléter)
- Interface Tkinter avec :
  - Menu principal
  - Ajout de visage
  - Entraînement du modèle
  - Reconnaissance faciale
- Détection de visages avec Haar Cascade.
- Reconnaissance avec LBPH (opencv-contrib).
- Sauvegarde du modèle dans `models/trainer.yml`.

---

## 📅 0.2.0 – Ajout gestion des visages
- Écran “Ajouter un visage” amélioré :
  - Liste des personnes existantes.
  - Liste des images par personne.
  - Prévisualisation d’une image.
  - Suppression d’images.

---

## 📅 0.3.0 – Scan automatique façon téléphone
- Ajout du bouton **📡 Scan auto** :
  - Capture automatique d’une série d’images.
  - Messages guidant l’utilisateur :
    - tête bien droite
    - tourner légèrement à gauche, à droite
    - regarder en haut, en bas.
- Sensibilité renforcée pour limiter les images de mauvaise qualité.

---

## 📅 0.4.0 – Reconnaissance stricte + thème dynamique
- Reconnaissance “très stricte” (peu ou pas de faux positifs).
- Seuil LBPH ajusté (préférence pour “Inconnu” en cas de doute).
- Thème clair/sombre en fonction de la luminosité de la caméra.

---

## 🧭 Roadmap (idées futures)

- [ ] Ajouter un système de rôles / droits (admin, user).
- [ ] Logger les reconnaissances (qui, heure, résultat) dans un fichier `logs/`.
- [ ] Ajouter un écran de réglages (sensibilité, seuils, chemin dataset, etc.).
- [ ] Support multi-caméra (choix de l’index caméra).
- [ ] Export / import de la base de visages (zip du dossier `dataset` + `models`).

---

## 🧪 Notes techniques / problèmes rencontrés

- NumPy 2.x casse certaines versions d’OpenCV : utiliser `numpy<2.0`.
- `cv2.face` est uniquement dans `opencv-contrib-python`, pas `opencv-python`.
- Sur certains PC, il faut forcer `cv2.CAP_DSHOW` pour que la caméra s’ouvre correctement (Windows).
