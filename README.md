# 🤖 SCRIBE AI

**Générateur Intelligent de Notes de Cadrage Bancaire**

SCRIBE AI est une application web complète qui utilise l'intelligence artificielle pour générer automatiquement des notes de cadrage de projets bancaires conformes aux standards du secteur. Propulsée par un modèle Mistral-7B fine-tuné spécifiquement pour le secteur bancaire, l'application permet de gagner jusqu'à 70% du temps de rédaction tout en maintenant une qualité professionnelle.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)

---

## ✨ Fonctionnalités Principales

### 🎯 Génération IA Intelligente
- **Génération contextuelle** : L'IA analyse le contexte du projet pour générer du contenu pertinent
- **Stratégies multiples** : Minimal, Contextuel, Smart, Full - adaptées à vos besoins
- **Champs supportés** : 20+ champs de texte et 7 types de tableaux
- **Apprentissage continu** : Système de feedback et correction pour améliorer la qualité

### 📊 Gestion Complète de Projet
- **Sections structurées** : 10 sections couvrant tous les aspects d'une note de cadrage
- **Tableaux dynamiques** : Contraintes, risques, lots, livrables, jalons, coûts
- **Calculs automatiques** : Totaux financiers, contingence, TCO 3 ans
- **Validation temps réel** : Vérification de cohérence et suggestions

### 📈 Visualisation et Planning
- **Diagramme de Gantt** : Génération automatique avec Canvas natif
- **Export multi-formats** : PNG haute définition, Excel CSV
- **Modes d'affichage** : Vue par jour, semaine ou mois
- **Zoom interactif** : Navigation fluide dans le planning

### 💼 Export Professionnel
- **Export Word** : Documents formatés selon les standards Groupe
- **Thèmes personnalisables** : 6 palettes de couleurs (Default, Green, Corporate, Modern, Elegant, DarkGreen)
- **Mise en page optimisée** : Headers, footers, table des matières automatique
- **Qualité A4** : Prêt pour impression et diffusion

### 💾 Sauvegarde et Templates
- **Sauvegarde JSON** : Export/import de projets complets
- **Autosave** : Sauvegarde automatique toutes les 30 secondes
- **Templates** : Création et réutilisation de modèles de projets
- **Récupération** : Restauration depuis sauvegarde automatique

### 📊 Insights et Analytics
- **Statistiques de génération** : Temps économisé, qualité moyenne
- **Feedback utilisateur** : Notation et correction des générations
- **Tableau de bord** : Vue d'ensemble de la qualité par champ
- **Apprentissage continu** : L'IA s'améliore avec vos retours

---

## 🏗️ Architecture Technique

### Backend (Python + FastAPI)
```
Backend/
├── final_api.py              # API FastAPI avec modèle fusionné
├── train_mistral_fixed7.py   # Script d'entraînement du modèle
└── mistral-banking-merged/   # Modèle Mistral-7B fine-tuné
```

**Technologies Backend :**
- **FastAPI** : API REST haute performance
- **PyTorch** : Framework deep learning
- **Transformers (Hugging Face)** : Gestion des modèles LLM
- **PEFT/LoRA** : Fine-tuning efficient
- **BitsAndBytes** : Quantization 4-bit

### Frontend (JavaScript Vanilla)
```
Frontend/
├── index.html                 # Point d'entrée
├── js/
│   ├── app.js                 # Application principale
│   ├── config.js              # Configuration globale
│   ├── core/
│   │   ├── Api.js            # Client API
│   │   ├── Router.js         # Routage SPA
│   │   └── Store.js          # State management
│   ├── components/
│   │   ├── Section.js        # Composant section
│   │   ├── ArrayField.js     # Champs tableaux
│   │   ├── Sidebar.js        # Menu navigation
│   │   └── Modal.js          # Modales
│   └── services/
│       ├── GenerationService.js    # Génération IA
│       ├── ExportService.js        # Export Word
│       ├── GanttService.js         # Diagrammes Gantt
│       ├── ProjectService.js       # Gestion projets
│       └── ValidationService.js    # Validation données
└── css/
    ├── main.css
    ├── components.css
    └── responsive.css
```

**Technologies Frontend :**
- **JavaScript ES6+** : Sans framework lourd
- **docx.js** : Génération de documents Word
- **Canvas API** : Diagrammes de Gantt
- **LocalStorage** : Persistance locale

---

## 🚀 Installation et Configuration

### Prérequis

#### Backend
- Python 3.8 ou supérieur
- CUDA 11.8+ (pour GPU NVIDIA)
- 16 GB RAM minimum (32 GB recommandé)
- 20 GB espace disque pour le modèle

#### Frontend
- Navigateur moderne (Chrome, Firefox, Safari, Edge)
- Serveur web (nginx, Apache, ou serveur de développement)

### Installation Backend

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/scribe-ai.git
cd scribe-ai/backend
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer le modèle**
```bash
# Le modèle fusionné doit être dans :
# /home/votre-user/mistral-banking-merged/
# Ou modifiez le chemin dans final_api.py ligne 44
```

5. **Lancer l'API**
```bash
python final_api.py
```

L'API sera disponible sur `http://localhost:5000`

### Installation Frontend

1. **Configurer le serveur web**

**Option 1 : Serveur Python simple**
```bash
cd scribe-ai/frontend
python -m http.server 8000
```

**Option 2 : Node.js avec http-server**
```bash
npm install -g http-server
cd scribe-ai/frontend
http-server -p 8000
```

**Option 3 : Production avec nginx**
```nginx
server {
    listen 80;
    server_name scribe-ai.local;
    root /path/to/scribe-ai/frontend;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:5000;
    }
}
```

2. **Accéder à l'application**
```
http://localhost:8000
```

### Configuration

**Backend (final_api.py)**
```python
class Settings:
    MODEL_PATH = "/path/to/mistral-banking-merged"
    HOST = "0.0.0.0"
    PORT = 5000
    
    # Génération
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    REPETITION_PENALTY = 1.2
    
    # Quantization (True = 4-bit, False = FP16)
    USE_QUANTIZATION = True
```

**Frontend (js/config.js)**
```javascript
API: {
    BASE_URL: 'http://localhost:5000',
    TIMEOUT: 30000
}
```

---

## 📖 Utilisation

### 1. Créer un Nouveau Projet

1. Ouvrez SCRIBE AI dans votre navigateur
2. Remplissez les **Informations Générales** :
   - Client
   - Secteur d'activité
   - Type de projet
   - Complexité
   - Libellé du projet
   - Année, numéro de projet, trigramme

### 2. Générer du Contenu avec l'IA

**Génération de champs texte :**
```
1. Naviguez vers la section souhaitée
2. Cliquez sur le bouton "🤖 Générer" à côté du champ
3. L'IA analyse le contexte et génère le contenu
4. Éditez le résultat si nécessaire
5. Notez la qualité (😍 Parfait → 😕 À améliorer)
```

**Génération de tableaux :**
```
1. Accédez à la section (Contraintes, Risques, Lots, etc.)
2. Cliquez sur "🤖 Générer les [éléments]"
3. Le tableau est rempli automatiquement
4. Modifiez les lignes individuellement
5. Ajoutez/supprimez des lignes au besoin
```

**Stratégies de génération :**
- **IA Minimale** : Rapide, contexte réduit
- **IA Contextuelle** : Équilibrée (recommandée)
- **IA Intelligente** : Analyse approfondie
- **IA Variée** : Maximum de contexte

### 3. Gestion Financière

**Coûts de Construction :**
```
1. Ajoutez des lignes pour chaque profil
2. Renseignez : Charge (j.h), TJM (€), Code équipe
3. Le total se calcule automatiquement
4. La contingence est appliquée selon le taux défini
```

**Coûts de Fonctionnement :**
```
1. Créez des postes de coût annuels
2. Saisissez : Quantité, Prix unitaire/an, Code UO
3. Pour le décommissionnement, utilisez des valeurs négatives
4. Le TCO 3 ans est calculé automatiquement
```

### 4. Planning et Visualisation

**Diagramme de Gantt :**
```
1. Définissez vos lots avec dates de début/fin
2. Ajoutez des jalons avec leurs dates
3. Cliquez sur "📊 Générer le diagramme de Gantt"
4. Explorez avec zoom et modes d'affichage
5. Exportez en PNG ou Excel
```

### 5. Export du Document

**Export Word :**
```
1. Cliquez sur "📄 Exporter Word" dans la barre d'actions
2. Le document est généré avec tous les contenus
3. Format professionnel avec table des matières
4. Headers/footers personnalisés
5. Prêt pour impression et diffusion
```

**Changer de thème de couleurs :**
```javascript
// Dans ExportService.js, ligne 15
this.currentTheme = 'default';  // ou 'green', 'corporate', etc.
```

### 6. Sauvegarde et Récupération

**Sauvegarder un projet :**
```
1. Cliquez sur "💾 Sauvegarder"
2. Un fichier JSON est téléchargé
3. Nommé automatiquement avec client et date
```

**Charger un projet :**
```
1. Cliquez sur "📂 Charger"
2. Sélectionnez un fichier JSON de projet
3. Les données sont restaurées instantanément
```

**Récupération automatique :**
```
- Autosave toutes les 30 secondes
- Récupération en cas de crash/fermeture
- Accessible depuis le menu projet
```

---

## 🧠 Entraînement du Modèle

### Préparation des Données

**Format d'entraînement (JSONL) :**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "[INST] <TASK>contexte_proj</TASK>\n<CONTEXT>\nclient: BNP Paribas\nsecteur: Banque\n...</CONTEXT> [/INST]"
    },
    {
      "role": "assistant",
      "content": "<START>Le projet s'inscrit dans...<END>"
    }
  ]
}
```

### Lancer l'Entraînement

**Configuration de base :**
```bash
python train_mistral_fixed7.py \
    --train-file train_dataset.jsonl \
    --val-file val_dataset.jsonl
```

**Reprendre depuis un checkpoint :**
```bash
python train_mistral_fixed7.py --resume
```

**Paramètres d'entraînement :**
```python
ModelConfig:
    lora_r = 64
    lora_alpha = 128
    lora_dropout = 0.15
    max_prompt_length = 1024
    max_response_length = 512

TrainingConfig:
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 5e-5
    num_train_epochs = 2
```

**Monitoring :**
```bash
# TensorBoard
tensorboard --logdir ./runs/mistral-banking-perf

# Logs de génération
tail -f generation_logs.jsonl
```

### Fusion et Déploiement

Après l'entraînement, fusionner le modèle :
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = PeftModel.from_pretrained(base_model, "./mistral-banking-perf/checkpoint-XXX")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./mistral-banking-merged")
```

---

## 📊 API Reference

### Endpoints Principaux

#### `POST /generate`
Génère du contenu pour un champ unique.

**Request :**
```json
{
  "field": "contexte",
  "context": {
    "client": "BNP Paribas",
    "sector": "Banque",
    "project_type": "Migration Cloud",
    "complexity": "Complexe"
  },
  "strategy": "related"
}
```

**Response :**
```json
{
  "success": true,
  "field": "contexte",
  "value": "Le projet s'inscrit dans...",
  "validation": {
    "valid": true,
    "warnings": []
  },
  "metadata": {
    "task": "contexte_proj",
    "strategy": "related",
    "token_count": 245
  }
}
```

#### `POST /generate_multiple`
Génère plusieurs champs en une requête.

#### `POST /feedback/correction`
Soumet une correction pour améliorer le modèle.

#### `POST /feedback/rating`
Note la qualité d'une génération (1-5 étoiles).

#### `GET /feedback/summary`
Récupère les statistiques de feedback.

---

## 🎨 Personnalisation

### Thèmes de Couleurs

Le système de thèmes permet de personnaliser l'apparence des exports Word :

```javascript
// ExportService.js - Changer de thème
exportService.setTheme('darkGreen');
// ou
exportService.exportToWordWithTheme('corporate');

// Thèmes disponibles :
// - default    : Bleu-gris (style document)
// - green      : Vert moderne
// - corporate  : Bleu professionnel
// - modern     : Violet innovant
// - elegant    : Noir sophistiqué
// - darkGreen  : Vert forêt élégant
```

### Personnalisation Avancée

**Couleurs personnalisées :**
```javascript
exportService.setCustomColors({
    primary: "1A73E8",      // Bleu Google
    primaryDark: "1557B0",
    accent: "34A853"        // Vert Google
});
```

**Ajout de champs personnalisés :**
```javascript
// config.js - Ajouter un nouveau champ
FIELDS: {
    TEXT_LONG: [
        // ... champs existants
        'monNouveauChamp'
    ]
}
```

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment participer :

1. **Fork** le projet
2. Créez une **branche** pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. **Commit** vos changements (`git commit -m 'Add: Amazing feature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une **Pull Request**

### Guidelines

- Code propre et documenté
- Tests pour les nouvelles fonctionnalités
- Respect des conventions de nommage
- Messages de commit descriptifs

---

## 📝 Changelog

### Version 4.0.0 (Actuelle)
- ✨ Modèle Mistral-7B fusionné optimisé
- 🎨 6 thèmes de couleurs pour exports
- 📊 Nouveaux champs : impactCO2, modalitesPartage
- 🔧 Support des coûts négatifs (décommissionnement)
- 📈 Amélioration du diagramme de Gantt
- 💾 Autosave toutes les 30 secondes
- 🐛 Corrections multiples de bugs

### Version 3.1.0
- 🤖 Système de feedback et correction
- 📊 Tableau de bord insights
- 🔄 Génération multiple de champs
- ✅ Validation temps réel

### Version 3.0.0
- 🚀 Migration vers Mistral-7B
- 🎯 Fine-tuning avec LoRA
- 📝 Support complet des tableaux
- 💼 Export Word professionnel

---

## 🔒 Sécurité

- ✅ Pas de données sensibles envoyées au cloud
- ✅ Modèle IA déployé localement
- ✅ Sauvegarde chiffrée recommandée
- ✅ CORS configuré pour environnement de production
- ⚠️ Ne jamais exposer l'API directement sur Internet sans authentification

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 👥 Auteurs

**Équipe SCRIBE AI**
- Développement initial et architecture
- Fine-tuning du modèle Mistral-7B
- Interface utilisateur et UX

---

## 🙏 Remerciements

- **Mistral AI** pour le modèle de base Mistral-7B
- **Hugging Face** pour la bibliothèque Transformers
- **FastAPI** pour le framework API
- **La communauté open-source** pour les nombreuses librairies utilisées

---

## 📧 Support

Pour toute question ou problème :
- 🐛 **Issues GitHub** : [Ouvrir un ticket](https://github.com/votre-username/scribe-ai/issues)
- 💬 **Discussions** : [Forum de discussion](https://github.com/votre-username/scribe-ai/discussions)
- 📧 **Email** : support@scribe-ai.local

---

## 🔗 Liens Utiles

- [Documentation complète](docs/)
- [Guide d'entraînement](docs/training.md)
- [API Reference](docs/api.md)
- [FAQ](docs/faq.md)

---

<div align="center">
  
**Fait avec ❤️ pour la transformation digitale bancaire**

[⬆ Retour en haut](#-scribe-ai)

</div>