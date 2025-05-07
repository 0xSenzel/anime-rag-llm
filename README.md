# Anime Knowledge Base Backend

A FastAPI-based backend service for the Anime Knowledge Base project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-anime-backend
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Environment Variables:

Create a `.env` file in the project root and set the following environment variables:

```
SUPABASE_DB_HOST=<your_supabase_host>
SUPABASE_DB_PORT=<your_supabase_port> # Usually 5432
SUPABASE_DB_NAME=<your_supabase_name>
SUPABASE_DB_USER=<your_supabase_user>
SUPABASE_DB_PASSWORD=<your_supabase_password>
SUPABASE_DB_SSL_MODE=require # Or disable if not using SSL
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`

2. Access the API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## psql command

- docker exec -it <your_postgres_container> psql -U <your_db_user> -d <your_db_name>

## Database Migrations (Alembic)

This project uses Alembic to manage database schema migrations.

**1. Initial Setup (Run Once):**

*   Ensure your `.env` file is configured correctly with database credentials.
*   Apply all existing migrations to create the initial database schema:

    ```bash
    alembic upgrade head
    ```

**2. Generating New Migrations (After Model Changes):**

*   After making changes to your SQLAlchemy models (e.g., in `app/models/`):
    *   **If you created a *new* model file (e.g., `app/models/new_table.py`):** You **must** manually add an import for the new model class in `app/models/__init__.py`. For example:
        ```python
        # app/models/__init__.py
        # ... other imports ...
        from .new_table import NewTable # Add this line
        ```
        This ensures Alembic can detect the new table.
    *   Generate a new migration script:
        ```bash
        alembic revision --autogenerate -m "Describe your changes here"
        ```
        *   Replace `"Describe your changes here"` with a concise message summarizing the model changes (e.g., `"Add message_id to ConversationHistory"` or `"Create new_table"`).
        *   Alembic will compare your models (including any newly imported ones) to the current database state and generate a script in `alembic/versions/`.
        *   **Important:** Always review the generated migration script before applying it.

**3. Applying Migrations:**

*   Apply the latest migration script (or all pending migrations):

    ```bash
    alembic upgrade head
    ```

*   Apply migrations up to a specific revision ID:

    ```bash
    alembic upgrade <revision_id>
    ```

**4. Checking Migration Status:**

*   View the migration history and see which revisions have been applied:

    ```bash
    alembic history
    ```

*   Show the current revision applied to the database:

    ```bash
    alembic current
    ```

**5. Reverting Migrations (Downgrading):**

*   Revert the most recent migration:

    ```bash
    alembic downgrade -1
    ```

*   Revert migrations down to a specific revision ID:

    ```bash
    alembic downgrade <revision_id>
    ```

*   Revert all migrations (use with caution!):

    ```bash
    alembic downgrade base
    ```

    *   **Warning:** Downgrading can potentially lead to data loss. Be sure you understand the implications before reverting migrations, especially in production.

## Project Structure

```
rag-anime-backend/
├── main.py           # Main FastAPI application
├── requirements.txt  # Project dependencies
└── venv/            # Virtual environment (not tracked in git)
```

## Development

- The server runs in development mode with hot-reload enabled
- Any changes to the code will automatically restart the server
- Use the interactive API documentation at `/docs` to test endpoints

## Dependencies

- FastAPI: Web framework for building APIs
- Uvicorn: ASGI server for running FastAPI applications
- Pydantic: Data validation and settings management 