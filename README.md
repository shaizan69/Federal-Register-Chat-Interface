# Federal Register RAG Agentic System

This is a RAG (Retrieval-Augmented Generation) agentic system that allows users to query information from the Federal Register through a chat interface. The system uses Ollama for local LLM inference, MySQL for data storage, and FastAPI for the backend.

## Features

- **Data Pipeline**: Fetches data from the Federal Register API, processes it, and stores it in MySQL
- **LLM Agent**: Uses Ollama with the qwen model to understand user queries and call appropriate tools
- **Database Tools**: Safely query the MySQL database using predefined functions
- **FastAPI Backend**: Handles user requests asynchronously
- **Simple Web Interface**: Chat-style frontend for interacting with the system

## Architecture

```
app/
├── agent/             # LLM agent and tool functions
├── api/               # FastAPI endpoints
├── data_pipeline/     # Federal Register data fetching and processing
├── database/          # MySQL connection and queries
├── frontend/          # Web interface
├── utils/             # Helper utilities
├── config.py          # Configuration variables
├── main.py            # Main application entry point
└── run_data_pipeline.py  # Script to run the data pipeline
```

## Prerequisites

- Python 3.8+
- MySQL 8.0+
- [Ollama](https://ollama.ai/) installed and running locally
- The `qwen:0.5b` or `qwen:1b` model pulled in Ollama

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd federal-register-rag
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up MySQL:
   - Create a new database: `CREATE DATABASE rag_agent_db;`
   - The application will create the necessary tables on startup

4. Configure environment variables (or edit `config.py`):
   - Database settings: host, port, username, password, database name
   - Ollama settings: base URL and model name
   - Application settings: host, port, etc.

5. Run the data pipeline to fetch initial data:
   ```
   python -m app.run_data_pipeline
   ```

6. Start the application:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. Access the web interface at http://localhost:8000

## Usage

1. Open the web interface in your browser
2. Type your question about Federal Register documents in the chat box
3. The system will:
   - Process your query using the LLM
   - Determine the appropriate database query to execute
   - Retrieve relevant information
   - Return a summarized response

Example queries:
- "What are the most recent EPA regulations?"
- "Show me documents related to climate change"
- "Find notices published by the Department of Transportation"
- "What documents were published yesterday?"

## Data Pipeline

The data pipeline fetches documents from the [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1) and stores them in the MySQL database. It also saves the raw and processed data as JSON and CSV files for backup.

To run the data pipeline manually:
```
python -m app.run_data_pipeline
```

You can also set up a scheduled task (cron job) to run this daily.

## Limitations

- The system only has access to data that has been fetched from the Federal Register API
- The quality of responses depends on the LLM model being used (qwen:0.5b or qwen:1b)
- The tool functions provide a limited set of query capabilities

## License

[MIT License](LICENSE) 