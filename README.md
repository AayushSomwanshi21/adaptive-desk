# AdaptiveDesk

AdaptiveDesk is a continual-learning platform for support ticket classification, human-in-the-loop review, and feedback-driven model updates. It helps teams route incoming support issues into categories, auto-label tickets when the model is confident, and retrain incrementally as new labelled examples arrive.

The project combines a FastAPI backend, a React + Vite dashboard, and a machine-learning training pipeline built around incremental updates and category promotion.

## Why this project exists

Support teams often deal with changing ticket vocabularies and shifting category mixes over time. AdaptiveDesk is designed to:

- classify incoming customer support tickets into categories;
- send low-confidence predictions to a human review queue;
- gradually approve new categories once they gather enough confirmed examples;
- retrain models incrementally instead of fully retraining from scratch;
- keep track of model versions and training jobs in a simple web dashboard.

## Core features

- Review queue for tickets needing confirmation
- Auto-labeling based on calibrated confidence thresholds
- Category approval workflow for proposed labels
- Incremental model training and version promotion
- Model registry for versioned deployment and rollback
- Dashboard for queue health, model status, and training activity
- Demo mode for feeding held-back categories to simulate continual learning

## Project structure

- `backend/` — FastAPI app, database models, training jobs, and APIs
- `frontend/` — React + TypeScript dashboard and UI
- `ml/` — continual-learning training code, model registry, evaluation, and datasets
- `tests/` — focused ML and validation tests
- `docs/` — project documentation folder

## Tech stack

- Backend: Python, FastAPI, SQLAlchemy, SQLite
- Frontend: React, TypeScript, Vite, React Router
- ML: PyTorch, scikit-learn, transformers, pandas, numpy
- Data: CSV-based training data and SQLite-backed application state

## Prerequisites

- Python 3.12
- Node.js 20+ and npm
- Windows PowerShell or a Unix shell for running commands

## Quick start

### 1) Create the backend virtual environment

PowerShell:

```powershell
cd C:\<path-to-folder>\adaptive-desk
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
```

### 2) Install backend dependencies

```powershell
pip install -r backend\requirements.txt
```

### 3) Install frontend dependencies

```powershell
cd frontend
npm install
```

### 4) Bootstrap the first model

From the repo root:

```powershell
python -m backend.app.bootstrap
```

This trains the initial classifier and promotes the first active model version. It also writes held-back demo tickets to `backend/data/demo_tickets.csv` when some categories are left out.

## Run the application

### Start the backend

```powershell
cd C:\<path-to-folder>\adaptive-desk
.\backend\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs will be available at:

- http://localhost:8000/docs
- http://localhost:8000/redoc

### Start the frontend

```powershell
cd C:\<path-to-folder>\adaptive-desk\frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open the app in a browser at:

- http://localhost:5173

## Demo workflow

The project includes a small demo flow for simulating a new category being added over time:

1. Bootstrap the app with some categories held back.
2. Inspect available demo categories:

```powershell
python -m backend.app.demo_feed --list
```

3. Send sample support tickets through the running API:

```powershell
python -m backend.app.demo_feed --category "<category_name>" --count 45 --approve
```

4. Review model status in the UI or via the `/training/plan` endpoint.
5. Start a training job from the dashboard or API.

You can also check whether the newly seen category is performing as expected with:

```powershell
python -m backend.app.demo_check --category "<category_name>" --skip 45
```

## Configuration

The app reads a few environment variables and defaults from `backend/app/config.py`:

- `ADAPTIVEDESK_DB` — SQLite database path
- `ADAPTIVEDESK_REGISTRY` — model registry directory
- `ADAPTIVEDESK_MODEL_NAME` — base model name used for fine-tuning
- `ADAPTIVEDESK_SPOT_CHECK_RATE` — fraction of auto-labels sent to review

## Default data paths

- SQLite DB: `backend/data/adaptivedesk.db`
- Demo tickets: `backend/data/demo_tickets.csv`
- Training data: `ml/datasets/train.csv`
- Model registry: `ml/models/registry`

## Development notes

- The backend exposes endpoints under `/tickets`, `/categories`, `/training`, `/models`, and `/stats`.
- `backend/app.bootstrap` initializes the first versioned model.
- `backend/app.services.JobRunner` launches background training jobs without blocking the web server.
- The active model version is tracked and stored in the model registry for deployment and evaluation workflows.

## Testing

This repo contains a focused test suite for the continual-learning memory components and related ML logic:

```powershell
cd C:\<path-to-folder>\adaptive-desk
.\backend\.venv\Scripts\Activate.ps1
pytest
```

## License

This project is currently distributed without a formal license file in the repository root. If this is intended for public use, add a license before publishing or distribution.

## Summary

AdaptiveDesk is a practical playground for building and observing a human-feedback workflow around support ticket classification. It is especially useful for experimenting with continual learning, category onboarding, and review-driven model updates in a realistic support-ticket context.
