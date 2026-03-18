# Energy Grid Classifier Application

A containerized full-stack application for geospatial/energy infrastructure analysis and machine learning inference, built with a **FastAPI backend**, **React/Vite frontend**, and **PostgreSQL**.

> **Quick start:** Clone the repo, download the ML model files (link to be added), place them in `backend/app/ml_models/` (create directory first), then run `docker compose up --build -d`.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. Download and place ML models](#2-download-and-place-ml-models)
    - [3. Build and start the application](#3-build-and-start-the-application)
    - [4. Configure API keys and database connection](#4-configure-api-keys-and-database-connection)
- [Database Requirements](#database-requirements)
    - [Required table layout (`spatial_data`)](#required-table-layout-spatial_data)
- [Accessing the Application](#accessing-the-application)
- [Docker Services and Ports](#docker-services-and-ports)
- [How Docker Build/Startup Works](#how-docker-buildstartup-works)
- [Scripts Folder (Training / Prototyping)](#scripts-folder-training--prototyping)
- [Useful Commands](#useful-commands)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [Future Improvements](#future-improvements)

---

## Overview

This project is a thesis application that combines:

- a **backend API** (FastAPI / Python) for machine learning inference and processing,
- a **frontend UI** (React + Vite + TypeScript) for user interaction and visualization,
- a **PostgreSQL database** for data storage,
- and pretrained **ML model files** placed locally in the backend.

The application is designed to be started entirely with Docker Compose, including building the frontend and installing backend dependencies.

---

## Architecture

The application runs as three containers:

1. **`db`** — PostgreSQL 15 (persistent storage via Docker volume)
2. **`backend`** — Python 3.11 / FastAPI (ML inference + API endpoints)
3. **`frontend`** — React/Vite app built and served via Nginx

The frontend communicates with the backend, and the backend connects to PostgreSQL.

---

## Tech Stack

### Backend
- Python 3.11
- FastAPI (served with `uvicorn`)
- ML inference code and model loading from `backend/app/`
- Dependencies installed from `backend/requirements.txt`

### Frontend
- React
- TypeScript
- Vite
- Nginx (production static hosting in container)

### Infrastructure
- Docker
- Docker Compose
- PostgreSQL 15 (Alpine)

---

## Project Structure

```text
Thesis/
├── backend/
│   ├── app/
│   │   ├── heat_radiation.py
│   │   ├── main.py
│   │   ├── ml_engine.py
│   │   ├── ml_models/              # Place downloaded ML model files here
│   │   └── models.py
│   ├── Dockerfile
│   └── requirements.txt
├── datasets/
├── docker-compose.yml
├── frontend/
│   ├── components/
│   │   ├── FeatureAnalysisTab.tsx
│   │   ├── icons.tsx
│   │   ├── LeftPanel.tsx
│   │   ├── MainViewTab.tsx
│   │   ├── PredictionSummaryCard.tsx
│   │   ├── RawDataTab.tsx
│   │   └── RightPanel.tsx
│   ├── App.tsx
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── vite.config.ts
│   └── ...
├── scripts/
│   ├── data/
│   ├── notebooks/
│   ├── preprocess.py
│   ├── test_cuda.py
│   └── train.py
└── README.md
```

### Notes
- `backend/app/ml_models/` is where the **pretrained model files must be placed** before starting the app.
- `scripts/` contains **training scripts** and **notebooks/templates** used during development and for future app functionality prototyping.

---

## Prerequisites

Before starting, make sure you have:

- **Git**
- **Docker**
- **Docker Compose** (Docker Compose v2 via `docker compose` command)
- Access to the required ML model files (download from S3)
- API keys for:
  - **Mapbox**
  - **Google Gemini API**
  - **AWS database access** (if using the hosted database)

Alternatively, you can connect **your own database** as long as it contains the required `spatial_data` table layout described below.

---

## Setup

## 1. Clone the repository

```bash
git clone https://github.com/gorzalniksamuel/Thesis_Node_Classification.git
cd Thesis
```
---

## 2. Download and place ML models

Download the pretrained ML model files from https://ml-models-2026.s3.eu-north-1.amazonaws.com/ml_models.zip and place them in:

```text
backend/app/ml_models/
```

Example model files currently expected in that folder:

- `best_convnext.pt`
- `best_effnet.pt`
- `best_model.pt`
- `best_resnet.pt`
- `best_swin.pt`
- `best_yolo11_old.pt`
- `best_yolo11.pt`
- `best_yolo26.pt`

> If the app expects specific filenames, do **not rename** the model files unless you also update the backend code accordingly.

---

## 3. Build and start the application

From the project root (`Thesis/`), run:

```bash
docker compose up --build -d
```

This command will:
- start the PostgreSQL database container,
- build the backend image and install Python dependencies from `backend/requirements.txt`,
- build the frontend production bundle with Vite,
- start the frontend via Nginx,
- run all services in detached mode.

Verify containers are running:

~~~bash
docker compose ps
~~~

View logs:

~~~bash
docker compose logs -f
~~~

---

## 4. Configure API keys and database connection

After the application has started, you must provide/configure the following credentials:

- **Mapbox API key**
- **Gemini API key**
- **AWS database API key / credentials** (if using the hosted AWS database)

### Important
If you are **not** using the provided AWS database, you must connect your **own database** and ensure it follows the required table layout for `spatial_data` (see [Database Requirements](#database-requirements)).

> Recommended approach: store secrets in environment variables or a `.env` file and **do not** commit them to Git.

---

## Database Requirements

The database contains a central table, `spatial_data`, which stores point-based geospatial records representing industrial and energy-related infrastructure. Each record consists of a unique identifier, geographic coordinates (latitude and longitude), a categorical classification, the data source, a timestamp, and additional metadata stored as semi-structured text. This design enables efficient spatial filtering while retaining heterogeneous contextual information from multiple external datasets.

The database can handle read-heavy workloads, as it primarily supports proximity-based lookup queries rather than transactional operations. By restricting the dataset to relevant geographic regions and categories, query performance is improved, enabling scalable spatial analysis across several hundred thousand records.

### Required table layout (`spatial_data`)

If you connect your own database, it should provide a table equivalent to:

- **Unique identifier** (primary key / unique ID)
- **Latitude**
- **Longitude**
- **Category / classification**
- **Data source**
- **Timestamp**
- **Additional metadata** (semi-structured text / JSON-like text)

A practical PostgreSQL example schema (adapt as needed to match backend expectations):

~~~sql
CREATE TABLE spatial_data (
    id TEXT PRIMARY KEY,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    category TEXT,
    source TEXT,
    record_timestamp TIMESTAMP,
    metadata TEXT
);
~~~

> If your backend expects different column names or types, update either the schema or backend query/model definitions accordingly.

---

## Accessing the Application

After startup, the services are exposed on the following host ports:

- **Frontend (UI):** `http://localhost:3001`
- **Backend API:** `http://localhost:8001`
- **PostgreSQL:** `localhost:5432`

Open the frontend in your browser:

~~~text
http://localhost:3001
~~~

---

## Docker Services and Ports

### Database (`db`)
- Image: `postgres:15-alpine`
- Container name: `postgres_db`
- Port mapping: `5432:5432`
- Persistent volume: `postgres_data`
- Database name: `energygridclassifier`

### Backend (`backend`)
- Built from: `./backend`
- Container name: `backend`
- Port mapping: `8001:8000`
- Mounts local backend app code:
    - `./backend/app:/app`
- Depends on: `db`
- Connects via:
    - `DATABASE_URL=postgresql://adminuser:ultrasafepassword@db:5432/energygridclassifier`

### Frontend (`frontend`)
- Built from: `./frontend`
- Container name: `frontend`
- Port mapping: `3001:80`
- Depends on: `backend`

### Network
- Docker network: `hydrogen_gas_net`

---

## How Docker Build/Startup Works

### Backend Dockerfile behavior
- Base: `python:3.11-slim`
- Installs system libs required by common ML/CV deps
- Installs Python deps from `backend/requirements.txt`
- Copies backend source from `backend/app/`
- Runs FastAPI with Uvicorn on port `8000`

### Frontend Dockerfile behavior (multi-stage)
- Build stage: `node:22-alpine`
    - `npm install`
    - `npm run build` (Vite -> `/app/dist`)
- Runtime stage: `nginx:1.27-alpine`
    - serves `/usr/share/nginx/html` on port `80`

---

## Scripts Folder (Training / Prototyping)

The `scripts/` directory contains training and preprocessing code used during model development and experimentation.

Includes:
- `train.py` — training workflow
- `preprocess.py` — data preprocessing
- `test_cuda.py` — CUDA environment testing
- `notebooks/` — exploratory notebooks and templates for future app functionality
- `data/` — script-related data assets

---

## Useful Commands

Start (build if needed):

~~~bash
docker compose up --build -d
~~~

Check status:

~~~bash
docker compose ps
~~~

Logs:

~~~bash
docker compose logs -f
~~~

Stop:

~~~bash
docker compose down
~~~

Stop + remove volumes (deletes DB data):

~~~bash
docker compose down -v
~~~

Rebuild a single service:

~~~bash
docker compose build backend
docker compose build frontend
~~~

---

## Troubleshooting

### Frontend loads but backend calls fail
- Check backend logs:
  ~~~bash
  docker compose logs -f backend
  ~~~
- Confirm model files exist in `backend/app/ml_models/`.

### Backend crashes on startup
Common causes:
- missing model files,
- missing/invalid API keys (Mapbox/Gemini/AWS DB),
- dependency issues,
- DB connectivity issues.

Check logs:
~~~bash
docker compose logs -f backend
~~~

### Database connection errors
- Check DB logs:
  ~~~bash
  docker compose logs -f db
  ~~~
- If using your own DB, ensure credentials/network access are correct and `spatial_data` exists with the required layout.

### Port already in use
If `3001`, `8001`, or `5432` are taken, stop the conflicting service or change port mappings in `docker-compose.yml`.

---

## Security Notes

- Do **not** commit API keys or DB credentials to GitHub.
- Use environment variables / `.env` files for secrets.
- The included Postgres credentials are for local development; change them for anything shared or production-like.
