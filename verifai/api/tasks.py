from celery import shared_task
from .models import Fact
import requests
from django.conf import settings
from .pipeline import run_claim_judge_pipeline
import time

@shared_task(bind=True)
def process_fact(self, fact_id):
    try:
        print("Processing fact:")
        fact = Fact.objects.get(id=fact_id)
        print("Processing fact: ", fact.claim)

        llm_response = run_claim_judge_pipeline(
                fact.claim,
            )

        # Update the fact instance with the LLM response data
        # Fact.objects.filter(id=fact.id).update(**llm_response)
        print(llm_response)
        return llm_response.get('final_output')
    except Exception as e:
        self.update_state(state='FAILURE', meta={'exc': str(e)})
        raise e
