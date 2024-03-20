#!/bin/sh

until cd /code/verifai
do
    echo "Waiting for server volume..."
done

# run a worker :)
poetry run celery -A verifai worker --loglevel=info --concurrency 1 -E