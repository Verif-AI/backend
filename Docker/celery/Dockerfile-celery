# Use an official Python runtime as a parent image
FROM python:3.10-alpine

RUN apk update && apk add python3-dev gcc libc-dev libffi-dev
RUN pip install --upgrade pip
# Install Poetry
RUN pip install poetry

# Set the working directory in the container
WORKDIR /code


# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock* /code/


# Install the project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of the project files
COPY . /code/
RUN chmod +x /code/Docker/celery/worker-entrypoint.sh