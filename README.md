# Django LLM Fact Verification API

## Project Overview

This project uses Django and Django REST Framework to create a fact verification system that integrates with a Large Language Model (LLM) service. It features a REST API for submitting facts, which are then verified, stored in a PostgreSQL database, and forwarded to an external LLM service for further processing.

## Key Features

- Fact Verification Endpoint: Provides a RESTful API endpoint (/api/verify/) for submitting and processing facts.
- Docker Integration: Utilizes Docker Compose for easy setup and deployment.
- PostgreSQL Database: Employs PostgreSQL for data storage.
- External LLM Service Integration: Configures interactions with an external LLM service for advanced processing.
- API Documentation: Generates API documentation using Swagger UI and Redoc (available at /api/swagger/ and /api/redoc/ respectively).

## Technologies Used

- Django & Django REST Framework: For building the web application and REST API.
- Poetry: For Python dependency management.
- Docker & Docker Compose: For containerization and orchestration.
- PostgreSQL: As the database system.
- Swagger UI & Redoc: For API documentation.


## Getting Started

Prerequisites
- Docker and Docker Compose installed on your machine.
- Poetry installed for Python package and dependency management.
- Setup Instructions

### Clone the Repository

Clone the project repository to your local machine and navigate to the project directory.
```bash
git clone <repository-url>
cd <project-directory>
```
### Docker Compose
Before running the Django application, you may want to start the Docker containers for services like the database.

1. Build and Start Containers 
```bash
docker compose up -d
``` 

### Poetry setup // Python dependencies
After setting up the Docker containers, set up the Python environment and install dependencies using Poetry.
Within the project root directory (where pyproject.toml is located):
```bash
poetry install
```
This command installs all dependencies defined in pyproject.toml.

### Running the Django Application

To run the Django development server using Poetry, ensure you are in the directory containing manage.py:
```bash
poetry run python verifai/manage.py runserver
```
This starts the Django development server, making the application accessible at http://localhost:8000/.

### Accessing the Application
With the containers running in the background, access the Django application at http://localhost:8000/ or the configured port.

### API Interaction
To verify facts, send POST requests to /api/verify/.
The system verifies, stores, and forwards the facts to the LLM service as configured.

### Development

Use Poetry for managing Python dependencies and for running Django management commands within the project's environment.
