# Résumé éco-conçu (FR/EN) — API Flask

Application Flask de **résumé de texte** (10–15 mots) utilisant **Transformers** (Hugging Face) et **PyTorch CPU**, avec **métriques de performance et d’énergie** (latence, mémoire, CO₂, Wh).
Interface web minimaliste et **accessible** (Tailwind), permettant de choisir la **langue** (FR/EN) et le **mode optimisé** (pruning + low‑rank + quantif INT8).

---

### 1. Prérequis

- Python 3.10 ou supérieur  
- Connexion Internet (pour le téléchargement initial du modèle Hugging Face)  
- CPU uniquement (aucune carte graphique requise)  
- Configuration recommandée : 2 à 4 vCPU, minimum 2 Go de RAM

---

### 2. Installation

#### a) Création de l’environnement virtuel

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

### b) Dépendances

**Option 1 :** avec `requirements.txt` (si présent)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Option 2 :** direct (fallback)

```bash
pip install --upgrade pip
pip install torch transformers flask psutil codecarbon
# (optionnel) torchao pour INT8 dynamique (si dispo sur votre plateforme)
pip install torchao --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 3) Arborescence

```
project/
├─ app.py                      # Point d’entrée (API Flask)
├─ templates/
│  └─ index1.html              # Interface web (dark dashboard)
└─ static/                     # (optionnel)
```

---

## 4) Variables d’environnement (optionnelles)

| Variable            | Rôle                                      | Défaut                            |
|---------------------|-------------------------------------------|-----------------------------------|
| `MODEL_NAME`        | Modèle HF                                 | `EleutherAI/pythia-70m-deduped`   |
| `MAX_NEW_TOKENS`    | Longueur max de génération                | `48`                              |
| `CPU_THREADS`       | Threads CPU utilisés                      | moitié des cœurs dispo            |
| `LR_MAX_LAYERS`     | Nb de Linear factorisées (SVD)            | `6`                               |
| `LR_RANK_CAP`       | Rang max                                  | `32`                              |
| `LR_RANK_FRAC`      | Ratio de rang                             | `0.10`                            |
| `PRUNE_AMOUNT`      | Pourcentage de pruning                    | `0.10`                            |
| `PRELOAD_BASELINE`  | Précharger le modèle baseline             | `1`                               |
| `PRELOAD_OPTIMIZED` | Précharger le modèle optimisé             | `1`                               |

Exemple (Linux/macOS) :

```bash
export MODEL_NAME="EleutherAI/pythia-70m-deduped"
export PRELOAD_OPTIMIZED=1
```

---

## 5) Lancer en développement

```bash
python app.py
```
Ouvrir : **http://127.0.0.1:5000**

### Test rapide (cURL)

```bash
curl -X POST "http://127.0.0.1:5000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Flask est un framework Python léger.", "optimized": true, "lang": "fr"}'
```

Réponse type (les champs correspondent exactement au frontend) :

```json
{
  "summary": "Flask est un micro‑framework Python léger pour API rapides",
  "energy_wh": 0.0578,
  "co2_g": 0.0032,
  "latency_ms": 1600,
  "memory_mb": 39.91,
  "throughput_tok_s": 30.01,
  "ppl": 60.66,
  "model_name": "EleutherAI/pythia-70m-deduped - Optimized",
  "lang_used": "fr"
}
```

---

## 6) Lancer en production

###  A. Linux / macOS — **Gunicorn**

```bash
pip install gunicorn
gunicorn "app:app" --bind 0.0.0.0:8000 --workers 1 --threads 2 --timeout 120 --preload
```

### B. Windows — **Waitress**

```powershell
pip install waitress
waitress-serve --listen=0.0.0.0:8000 app:app
```

---

## 7) Endpoints

| Endpoint     | Méthode | Description                                   |
|--------------|---------|-----------------------------------------------|
| `/`          | GET     | Interface web (`templates/index1.html`)       |
| `/health`    | GET     | Statut & modèles préchargés                   |
| `/summarize` | POST    | Résumé + métriques (latence, énergie, etc.)   |

**Body attendu :**

```json
{
  "text": "Mon texte à résumer…",
  "optimized": true,
  "lang": "fr"
}
```

---

## 8) Détails techniques

- **Baseline** : modèle Hugging Face tel quel.  
- **Optimisé (CPU)** : pruning léger + factorisation **SVD low‑rank** sur grosses `Linear` + **INT8 dynamique** (TorchAO si dispo, sinon `torch.ao.quantization`).  
- **Métriques** :  
  - Latence (ms)  
  - Énergie (Wh) & **CO₂ (g)** via **CodeCarbon**  
  - Mémoire (Δ MB) via **psutil**  
  - Débit (tokens/s)  
  - **Perplexité proxy** (ppl) via loss du modèle

---

## 9) Dépannage rapide

- `ModuleNotFoundError: transformers / torch` → exécuter `pip install -r requirements.txt` (ou voir Option 2 ci‑dessus).  
- CodeCarbon indisponible → l’API renverra énergie/CO₂ à `0`/`null` sans planter.  
- Lancement mais page blanche → vérifier que `templates/index1.html` existe et que `app.py` sert bien `render_template("index1.html")`.

---

## Licence

Projet éducatif — usage libre pour expérimentations (ajoute ta licence si besoin).


