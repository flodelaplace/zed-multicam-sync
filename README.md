# Pipeline de synchronisation multi-caméras ZED

Pipeline Python pour traiter des enregistrements SVO (Stereolabs ZED) filmés avec
**plusieurs caméras en parallèle** et produire des MP4 **parfaitement synchronisés**
frame-à-frame.

Utile quand :
- tes N caméras ZED ne démarrent pas exactement en même temps,
- leurs horloges dérivent dans le temps,
- certaines images ont été perdues (drops) à l'enregistrement.

Le pipeline détecte ces problèmes, les corrige, et aligne toutes les vidéos sur un
événement visuel commun que tu choisis manuellement.

---

## TL;DR

```bash
git clone https://github.com/flodelaplace/zed-multicam-sync.git
cd zed-multicam-sync

conda env create -f environment.yml
conda activate zed_env

# Installer le binding ZED SDK (voir section "Installation" ci-dessous)

python pipeline_sync.py -i "chemin/vers/dossier/contenant/les/svo"
```

Le pipeline produit un dossier `MP4_synced/` avec toutes les vidéos alignées.

---

## Prérequis

- **Python 3.10** (testé). 3.8+ devrait marcher.
- **ffmpeg** accessible en ligne de commande (`ffmpeg -version` doit répondre).
- **ZED SDK** installé + binding Python `pyzed` — obligatoire pour lire les `.svo`.
- **OS** : Windows (principalement testé), Linux/macOS devraient fonctionner.

---

## Installation

### 1. Cloner le repo

```bash
git clone https://github.com/flodelaplace/zed-multicam-sync.git
cd zed-multicam-sync
```

### 2. Créer l'environnement

**Option A — conda (recommandée)** :

```bash
conda env create -f environment.yml
conda activate zed_env
```

**Option B — venv + pip** :

```bash
python -m venv .venv
# Windows :
.\.venv\Scripts\activate
# Linux / macOS :
source .venv/bin/activate

pip install -r requirements.txt
```

Dans ce cas, installer `ffmpeg` séparément (via le site officiel, `winget`, `apt`, `brew`…).

### 3. Installer le binding ZED Python (`pyzed`)

`pyzed` n'est **pas** disponible via pip/conda standard. Il faut :

1. Télécharger et installer le **ZED SDK** depuis
   [stereolabs.com/developers/release](https://www.stereolabs.com/developers/release/).
2. Exécuter le script `get_python_api.py` fourni par l'installeur du SDK
   (généralement dans `C:\Program Files (x86)\ZED SDK\` sur Windows).
3. Vérifier :
   ```bash
   python -c "import pyzed.sl as sl; print(sl.Camera)"
   ```

Si cette commande n'affiche pas d'erreur, le binding est opérationnel.

---

## Utilisation

### Structure attendue en entrée

Un dossier contenant **un ou plusieurs fichiers `.svo`**, un par caméra :

```
mon_enregistrement/
├── cam1_22516499.svo
├── cam2_22513567.svo
└── cam3_22519876.svo
```

Les noms des fichiers identifient chaque caméra dans les outputs.

### Lancer le pipeline complet

```bash
python pipeline_sync.py -i "chemin/vers/mon_enregistrement"
```

### Arguments utiles

| Argument | Description |
|---|---|
| `-i, --input-dir` | (requis) dossier contenant les `.svo` |
| `-o, --output-dir` | dossier de sortie (défaut : `input-dir`) |
| `--fps N` | force le FPS cible (sinon détection auto depuis les timestamps) |
| `--start-frame N` | traite uniquement à partir de la frame N |
| `--end-frame N` | s'arrête à la frame N (inclus) |
| `--workers N` | nombre de processus parallèles (défaut : min(n_caméras, n_cpu)) |
| `--rotate SPEC` | rotation par caméra (voir section ci-dessous) |
| `--overwrite` | force la ré-exécution de toutes les étapes |
| `--rerun-repair` | refait seulement les étapes 4 + 6 (ré-utilisation avec nouvelle rotation) |
| `--rerun-sidecars` | régénère uniquement les sidecars `*.dropped.json` sans retoucher aux MP4 |
| `--log-file PATH` | fichier log (défaut : `<output>/pipeline.log`) |
| `--debug` | logs DEBUG |

### Reprise après interruption

Chaque étape vérifie si sa sortie existe déjà et la saute. Relance la même
commande pour reprendre où tu t'étais arrêté.e. Utilise `--overwrite` si tu
veux tout recalculer.

### Rotation de caméras à l'envers (`--rotate`)

Certaines caméras peuvent être montées tête-en-bas ou en portrait. Le flag
`--rotate` applique une rotation **pendant l'étape de réparation** (rapide),
ce qui veut dire que tu peux itérer sans re-lire les SVO.

| Format | Effet |
|---|---|
| `--rotate 180` | toutes les caméras à 180° |
| `--rotate "22516499=180"` | seule la caméra dont le nom contient `22516499` |
| `--rotate "22516499=180,23859316=90"` | une rotation par caméra |
| `--rotate "all=180,24710321=0"` | toutes à 180° **sauf** celle qui matche `24710321` |

Angles supportés : `0`, `90`, `180`, `270` (degrés dans le sens horaire).

**Workflow typique** :
1. Lance le pipeline une première fois sans `--rotate`.
2. Ouvre les vidéos dans `MP4_repares/` pour repérer celles à l'envers.
3. Relance avec `--rotate "xxx=180,..." --rerun-repair` — seules les étapes 4
   et 6 sont refaites, l'extraction SVO (qui est la plus lente) est sautée.

Le flag `--rerun-repair` force uniquement les étapes 4 (réparation) et 6
(découpage). Le flag `--overwrite` refait tout (y compris l'extraction SVO).

---

## Contrôles de l'interface GUI (étape 5)

À l'étape de **sélection des frames de référence**, une fenêtre OpenCV s'ouvre
pour chaque caméra. Tu dois choisir une même frame-événement (flash, clap,
lumière qui s'allume…) visible sur toutes les caméras.

| Touche | Action |
|---|---|
| `D` ou `→` | avancer d'une frame |
| `A` ou `←` | reculer d'une frame |
| `X` | avancer de 10 frames |
| `Z` | reculer de 10 frames |
| `W` ou `Espace` | marquer la frame courante comme référence |
| `S` ou `Entrée` | valider et passer à la caméra suivante |
| `Q` | annuler et quitter |

---

## Architecture du pipeline

```
SVO (×N caméras)
    │
    │  Étape 1+3 (parallèle) : lecture SVO + extraction timestamps + écriture MP4
    ▼
MP4 bruts + CSV global des timestamps
    │
    │  Étape 2 : détection auto du FPS (médiane des deltas)
    │
    │  Étape 4 (parallèle) : insertion de frames noires pour combler les drops
    ▼
MP4_repares/  (toutes vidéos au même tempo)
    │
    │  Étape 5 : sélection manuelle d'une frame de référence par caméra (GUI)
    ▼
reference_frames.csv
    │
    │  Étape 6 : découpage frame-accurate pour aligner les vidéos
    ▼
MP4_synced/   (toutes vidéos alignées, même longueur)
    │
    │  Étape 7 : écriture des sidecars *.dropped.json (indices des frames noires)
    ▼
MP4_synced/<cam>_repaired_synced.dropped.json
```

### Pourquoi deux passes ?

- **Passe 1** (étapes 1+3) : chaque `.svo` est ouvert **une seule fois**. On lit
  chaque image, on enregistre son timestamp, et on l'écrit dans un MP4.
- **Passe 2** (étape 4) : connaissant maintenant le FPS global, on rouvre chaque
  MP4 et on insère des frames noires là où les timestamps indiquent des drops.

Les deux passes sont parallélisées par caméra (`ProcessPoolExecutor`).

---

## Sorties produites

Dans `<output-dir>` (= `<input-dir>` par défaut) :

```
<output-dir>/
├── Analyse_Timestamps_Global.csv   # timestamps bruts par caméra
├── Graphiques_Analyse.png          # delta-t + dérive temporelle
├── <cam>.mp4                       # MP4 bruts (une par caméra)
├── reference_frames.csv            # frame de référence choisie dans la GUI
├── pipeline.log                    # log complet de l'exécution
├── MP4_repares/
│   └── <cam>_repaired.mp4                        # MP4 avec frames noires insérées
└── MP4_synced/
    ├── <cam>_repaired_synced.mp4                 # MP4 finaux alignés
    └── <cam>_repaired_synced.dropped.json        # sidecar : indices des frames noires
```

**Ce qu'il faut utiliser** : les fichiers dans `MP4_synced/`. Les autres sont
des étapes intermédiaires / outils de diagnostic.

### Sidecars `*.dropped.json`

À côté de chaque MP4 synced, un sidecar JSON liste les **indices 0-based des
frames noires** insérées pour compenser les drops SVO. Format :

```json
{
  "fps": 30,
  "total_frames": 1800,
  "dropped_frame_indices": [42, 87, 512]
}
```

Les consommateurs aval (calibration multi-cam, export Pose2Sim, tracking…)
peuvent ainsi **ignorer ces frames sans refaire de détection pixel**. Les
indices sont calculés de façon déterministe depuis les timestamps SVO et la
sélection GUI — ils décrivent exactement les frames insérées à l'étape 4,
retransposées dans l'espace du MP4 synced final (après découpage).

Pour régénérer les sidecars sur un run déjà calculé, sans rejouer tout le
pipeline :

```bash
python tools/build_dropped_sidecars.py --input-dir <output-dir>
```

---

## Dépannage

### « ffmpeg not found »

Vérifier l'install :
```bash
ffmpeg -version
```
Si rien ne répond, l'ajouter au PATH ou utiliser l'env conda (qui inclut ffmpeg).

### « No module named pyzed »

Le binding ZED n'est pas installé. Voir section [Installation — étape 3](#3-installer-le-binding-zed-python-pyzed).

### Une caméra extraite donne 0 frames

- Vérifier que le `.svo` n'est pas corrompu (l'ouvrir dans ZED Explorer).
- Si le chemin contient des accents / espaces : déplacer dans un chemin ASCII.

### Les vidéos finales ne sont pas alignées

- Relancer avec `--overwrite` après avoir supprimé `reference_frames.csv` et
  refaire la sélection GUI **avec attention** sur un événement net et commun.
- Vérifier le fichier `Graphiques_Analyse.png` : si une caméra a une dérive
  aberrante, l'alignement sur un seul point de référence sera imprécis en fin
  de séquence.

### Multiprocessing qui plante au démarrage

Certaines versions du ZED SDK n'aiment pas le spawn de processus multiples.
Forcer un seul worker :
```bash
python pipeline_sync.py -i <dir> --workers 1
```

---

## Outils annexes (`tools/`)

Scripts autonomes non requis par le pipeline principal, mais utiles en dépannage :

| Script | Usage |
|---|---|
| `tools/convert_svo.py` | Conversion SVO → MP4 seule (ffmpeg avec fallback ZED SDK) |
| `tools/svo_converter.py` | Conversion SVO → MP4 avec choix de vue (`left`/`right`/`depth`/`side_by_side`) |
| `tools/utils_rotate_ffmpeg.py` | Rotation batch de vidéos (90°/270°) via ffmpeg |
| `tools/bench_convert.py` | Benchmark ffmpeg vs ZED SDK sur un segment court |
| `tools/build_dropped_sidecars.py` | (Re)génère les sidecars `*.dropped.json` sur un run existant, sans rejouer les étapes 1-6 |

Chaque script est exécutable avec `python tools/<nom>.py --help`.

---

## Structure du repo

```
Codes_ZED/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── pipeline_sync.py           # Orchestrateur principal
├── cut_sync.py                # Étape 6 : découpage frame-accurate
├── select_reference_gui.py    # Étape 5 : GUI de sélection
│
└── tools/                     # Utilitaires standalone (optionnels)
    ├── convert_svo.py
    ├── svo_converter.py
    ├── utils_rotate_ffmpeg.py
    ├── bench_convert.py
    └── build_dropped_sidecars.py
```

---

## Notes techniques

- **Frame-accurate cut** : l'étape 6 utilise `ffmpeg -ss` **après** `-i` avec
  ré-encodage H.264 (et fallback OpenCV). L'ancienne version utilisait
  `-c copy` qui seek au keyframe le plus proche — jusqu'à ±2s d'erreur sur
  des GOP longs.
- **Insertion de frames noires** : choix volontaire pour matérialiser les drops.
  Les indices des frames insérées sont exportés dans les sidecars
  `*.dropped.json` (voir section *Sidecars*), ce qui permet au traitement aval
  (calibration, Pose2Sim, pose estimation, tracking…) de les ignorer sans
  refaire de détection pixel. Si malgré tout elles perturbent un pipeline
  existant, on peut remplacer par la dernière frame valide : dans
  `pipeline_sync.py > _repair_mp4_worker`, remplacer `out.write(black)` par
  `out.write(frame)`.
- **FPS détecté** : médiane globale des deltas entre timestamps, sur toutes
  caméras confondues. Si les caméras tournent à des FPS différents (cas rare
  mais possible), forcer avec `--fps N`.

---

## Licence / contact

Code à usage interne (stage M2). Pour toute question, contacter l'auteur·e du
projet.
