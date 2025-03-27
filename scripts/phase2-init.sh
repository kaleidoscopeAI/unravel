# Unravel AI - Phase Migration Guide

This document provides instructions for migrating your work from this conversation to the next one, ensuring a seamless continuation of the Unravel AI project development.

## Phase 1 Summary

We've successfully created the core infrastructure for Unravel AI:

1. Project structure with modular components
2. Database schema and initialization scripts
3. Configuration management system 
4. Sandbox environment with Docker
5. FastAPI-based API foundation
6. Deployment configuration for Render

## Migration Instructions

1. **Download the setup script**:
   Save the `unravel-setup.sh` script to your local machine.

2. **Run the setup script**:
   ```bash
   chmod +x unravel-setup.sh
   ./unravel-setup.sh
   ```

3. **Configure environment**:
   ```bash
   cd unravel-ai
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Initialize the project**:
   ```bash
   ./setup.sh
   ```

5. **Initialize the database**:
   ```bash
   ./launch.sh init-db
   ```

## Continuing in the Next Conversation

In the next conversation, start by providing the Phase Transition Token:

```
UNRAVEL_PHASE_TOKEN=<token from NEXT_PHASE.md file>
```

This will signal to continue development from Phase 1 to Phase 2, where we'll implement:

1. Core engine migration from Kaleidoscope
2. Software ingestion pipeline
3. LLM integration for analysis
4. Database interaction layer

## Folder Structure Reference

```
unravel-ai/
├── config/
│   ├── dev/
│   └── prod/
├── data/
├── docker/
│   ├── app/
│   └── sandbox/
├── scripts/
│   ├── db_init.py
│   ├── deploy_render.py
│   └── migrate_session.py
├── src/
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   └── detection.py
│   ├── data/
│   ├── sandbox/
│   │   └── manager.py
│   └── utils/
│       └── config.py
├── tests/
├── .env.example
├── launch.sh
├── render.yaml
├── requirements.txt
└── setup.sh
```

## Database Connection

The database is already set up on Render with the following connection string:

```
postgresql://unravel_admin:aTY6nh1Pj8SWxtyrFX0S1uydNRIfOxIN@dpg-cvajmjaj1k6c738ta6ug-a.ohio-postgres.render.com/unravel_ai
```

## System Architecture

The Unravel AI system follows a modular architecture:

1. **Core Engine**: Handles software ingestion, analysis, and generation
2. **API Layer**: FastAPI-based endpoints for interacting with the system
3. **Sandbox**: Docker-based secure execution environment
4. **Database**: PostgreSQL for storing metadata, specs, and tracking
5. **LLM Integration**: Connection to language models for code analysis

## Next Steps

In Phase 2, you'll need to:

1. Implement the core engine components
2. Create the ingestion pipeline
3. Add LLM integration
4. Build database models and repositories

## Deployment

To deploy the application to Render:

```bash
./launch.sh deploy <your_render_api_key>
```

This will use the pre-configured `render.yaml` file to set up the service.