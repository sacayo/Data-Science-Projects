# Streamlit Frontend

> Interactive web interface for searching and filtering county ordinances across multiple states — deployed on AWS Elastic Beanstalk with Nginx proxy.

This is **Component 4** in the RAG system. It provides a user-facing interface that connects to the [RAG Query API](../rag-query/README.md), supporting multi-state search, advanced filtering by legal classification and readability metrics, and CSV export of results.

---

## Features

- **Multi-state search**: Query ordinances across CA, FL, GA, and TX
- **Legal classification filters**: Penalty, obligation, permission, prohibition
- **Readability filters**: Flesch-Kincaid grade, reading ease, word count, complexity
- **Interactive results**: Chunk-level scores and metadata display
- **CSV export**: Download search results for offline analysis
- **Sticky search bar**: Persistent UI controls for iterative querying

---

## Quick Start

### Local Development

```bash
cd rag-pipeline/streamlit-app
pip install -r requirements.txt

# Configure API endpoint
echo 'UNBARRED_API="http://localhost:8000/query"' > .env

# Run
python run.py
# or: streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### AWS Elastic Beanstalk Deployment

```bash
# Package for deployment (excludes dev artifacts)
zip -r -X streamlit-app.zip . \
  -x ".git" -x "__pycache__" -x "*.pyc" -x ".env" \
  -x "run.py" -x "uv.lock" -x ".python-version" \
  -x ".DS_Store" -x ".venv/" -x "*.egg-info/"

# Upload via Elastic Beanstalk console → Upload and deploy
```

The `Procfile` runs Streamlit headless on port 8000 behind Nginx:

```
web: streamlit run app.py --server.address 0.0.0.0 --server.port 8000 --server.headless true
```

---

## Configuration

### Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `UNBARRED_API` | `http://3.234.136.27:8000/query` | RAG Query API endpoint |
| `UNBARRED_API_KEY` | `""` | Optional API key (leave empty if no auth) |
| `UNBARRED_PASSCODES` | `demo123` | Application access passcode (optional) |

### Elastic Beanstalk Settings

- **Platform**: Python 3.11 on 64-bit Amazon Linux 2023
- **Environment type**: Single instance
- **Proxy**: Nginx forwarding to Streamlit on port 8000

Set environment variables under **Configuration > Software > Environment properties** in the EB console.

---

## Project Structure

```
streamlit-app/
├── app.py              # Main Streamlit application
├── run.py              # Local launcher (loads .env)
├── Procfile            # Elastic Beanstalk process definition
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata
└── README.md
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend connection errors | Verify `UNBARRED_API` is reachable; check EC2 security group allows port 8000 |
| App fails on Beanstalk | Check `/var/log/web.stdout.log` for Python tracebacks |
| Slow responses | Expected — LLM generation takes 2-10s depending on search mode |

## Dependencies

Requires the [RAG Query API](../rag-query/README.md) to be running and accessible at the configured endpoint.
