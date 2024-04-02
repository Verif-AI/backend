from celery import shared_task
from .models import Fact
import requests
from django.conf import settings
from .pipeline import run_claim_judge_pipeline
import time

@shared_task(bind=True)
def process_fact(self, fact_id):
    try:
        time.sleep(5)
        fact = Fact.objects.get(id=fact_id)

        if settings.ENV_LOCATION == "local":
            llm_response = run_claim_judge_pipeline(
                fact.claim,
            )

        else:
            llm_service_url = f"{settings.LLM_ENDPOINT}/api/generate"
            response = requests.post(llm_service_url, json={'model': 'llama2', 'prompt': fact.claim, 'stream': False})
            llm_response = response.json()

        # Update the fact instance with the LLM response data
        # Fact.objects.filter(id=fact.id).update(**llm_response)
        return llm_response
    except Exception as e:
        self.update_state(state='FAILURE', meta={'exc': str(e)})
        raise e
