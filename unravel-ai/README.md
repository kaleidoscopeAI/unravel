# Unravel AI

An intelligent software ingestion and mimicry system capable of analyzing, decompiling, and reconstructing software with enhanced capabilities.

## Features

- **Software Ingestion**: Decompile and analyze binaries and obfuscated code
- **Specification Generation**: Create detailed specifications from analyzed code
- **Software Reconstruction**: Generate enhanced versions of ingested software
- **Language Mimicry**: Create new software in different programming languages
- **Secure Sandbox**: Test generated applications in a secure environment
- **Licensing System**: Manage licenses and monetization

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL database
- API keys for LLM services (OpenAI, Anthropic, etc.)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/unravel-ai.git
   cd unravel-ai
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create configuration:
   ```
   cp .env.example .env
   # Edit .env with your settings
   ```

5. Initialize the database:
   ```
   python scripts/db_init.py --db-url <your-database-url>
   ```

6. Start the API server:
   ```
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Deployment

### Deploying to Render

1. Fork this repository to your GitHub account

2. Create a new Web Service on Render, linking to your forked repo

3. Set the required environment variables in the Render dashboard

4. Alternatively, use our deployment script:
   ```
   python scripts/deploy_render.py --api-key YOUR_RENDER_API_KEY --repo-url YOUR_GITHUB_REPO
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
