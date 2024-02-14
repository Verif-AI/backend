from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Fact
from .serializers import FactSerializer
from django.conf import settings


@api_view(['POST'])
def verify_fact(request):
    print(request.data)
    serializer = FactSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save(verified=True)  # Assuming verification logic is handled elsewhere
        import requests
        # Here, send a request to your LLM service
        # You can use requests library for HTTP requests (make sure to install it)
        if settings.ENV_LOCATION == "production":
            llm_service_url = f"{settings.LLM_ENDPOINT}/generate"

            response = requests.post(llm_service_url, json={'fact_id': serializer.data['id']})
        else:
            llm_service_url = f"{settings.LLM_ENDPOINT}/api/generate"
            print(llm_service_url)
            print(serializer.data['content'])
            response = requests.post(llm_service_url, json={'model': 'llama2', 'prompt': serializer.data['content'], 'stream': False})
            print(response.json())
        return Response({'status': 'Fact verified and LLM request sent', 'llm_response': response.json()},
                        status=status.HTTP_200_OK)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# Create your views here.
