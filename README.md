# Projet de Réduction des Étoiles dans les Images Astronomiques

## Auteurs
- Rampelberghe Rémy
- Gaillard Noa
- Nave Axel

## Description du Projet

Ce projet vise à réduire la visibilité des étoiles dans les images de nébuleuses pour mieux faire ressortir les structures du gaz interstellaire. On utilise la morphologie mathématique et la détection d'étoiles pour appliquer une érosion sélective.

## Méthodologie

### Phase 1 : Érosion Globale
On a testé différentes configurations d'érosion morphologique :
- Kernels de tailles variées (3×3, 5×5, 7×7)
- Plusieurs itérations (1 et 3 iterations)
- Résultat : l'érosion globale réduit les étoiles mais floute aussi la nébuleuse

### Phase 2 : Réduction Sélective

**Étape A - Détection des étoiles :**
- Utilisation de `DAOStarFinder` (photutils) pour localiser les étoiles
- Calcul des statistiques du fond de ciel (sigma clipping)
- Seuil de détection : 3×σ du bruit de fond
- Fallback sur seuillage adaptatif si nécessaire

**Étape B - Application du masque :**
1. Création d'un masque binaire (cercles de rayon 5px autour de chaque étoile)
2. Lissage du masque par flou gaussien (7×7, σ=2) pour éviter les transitions brutales
3. Érosion douce de l'image (kernel 3×3, 1 itération)
4. Interpolation : `I_final = M × I_erode + (1-M) × I_original`

**Résultat :** Les étoiles sont réduites sans affecter la nébuleuse

### Phase 3 : Interface Interactive

**Prolongements implémentés :**

1. **Interface Utilisateur**
   - Chargement de fichiers FITS
   - 6 paramètres ajustables en temps réel :
     - Seuil de détection (1-5×σ)
     - Rayon des étoiles (2-12 px)
     - Flou gaussien (taille et sigma)
     - Kernel d'érosion (3-9)
     - Itérations (1-5)

2. **Comparateur Avant/Après**
   - Mode clignotement (500ms) pour comparaison visuelle
   - Slider de fondu (0-100%) pour doser l'effet
   - Mode différence (carte de chaleur) pour voir les zones modifiées

**Technologies :**
- Tkinter pour l'interface graphique
- Matplotlib pour la visualisation
- Support des images couleur (RGB)

## Difficultés Rencontrées

1. **Préservation de la nébuleuse**
   - Problème : L'érosion débordait sur la nébuleuse proche des étoiles
   - Solution : Réduction du rayon (8→5px) et du flou gaussien (5×5→3×3)

2. **Précision des calculs**
   - Problème : Perte de précision avec uint8
   - Solution : Passage à float64 pour tous les traitements

3. **Détection robuste**
   - Problème : DAOStarFinder échoue sur certaines images
   - Solution : Système de fallback avec seuillage par percentile

4. **Interface réactive**
   - Problème : Recalcul complet à chaque ajustement
   - Solution : Optimisation du pipeline (détection séparée du traitement)

## Résultats

### Images Test
- **test_M31_linear.fits** : 974 étoiles détectées
- **HorseHead.fits** : Tête de Cheval bien préservée

