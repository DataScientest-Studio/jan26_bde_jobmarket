## API Recherche des offres


`range` : Pagination des données => La plage de résultats est limitée à 150.

Avec le format : `p-d`, où :

- `p est l'index (débutant à 0)` du premier élément demandé ne devant pas dépasser  3000
- `d est l'index de dernier élément` demandé ne devant pas dépasser 3149

=> On récupère des fenêtres de 150 éléments avec au maximum avec des bornes max **p ≤ 3000 / d ≤ 3149**


**Conséquence :** 
une collecte complète doit être partitionnée (sinon une requête peut dépasser ~3150 résultats récupérables via range).

Il faut partitionné ou se restreindre à des sous-ensemble de 3150 élements.

**Approches possibles**

- Incrémental par fenêtre de temps
 avec `minCreationDate` / `maxCreationDate`

- Partitionnement fonctionnel : utiliser des axes qui limitent le volume: `departement` (jusqu'à 5 valeurs) `codeRome` (jusqu'à 200 valeurs).


API ROME
https://francetravail.io/produits-partages/catalogue/rome-4-0-metiers

Le référentiel des appellations d’emploi est composé de:

**1584 métiers** (et plus de 13120 appellations) ​
110 domaines professionnels ​
14 grands domaines. ​
17752 compétences structurées ​
6 domaines de compétences
32 enjeux, 84 objectifs
507 macro-compétences