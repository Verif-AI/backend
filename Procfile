web: celery -A verifai worker --loglevel=info & python manage.py migrate && gunicorn verifai.wsgi  --bind 0.0.0.0:$PORT