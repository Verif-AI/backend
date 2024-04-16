from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Fact
from .serializers import FactSerializer, ResultSerializer
from django.conf import settings
from django.http import JsonResponse
from celery.result import AsyncResult
from .tasks import process_fact
import requests


@api_view(['POST'])
def verify_fact(request):
    serializer = FactSerializer(data=request.data)
    if serializer.is_valid():
        fact = serializer.save()  # Save the initial fact with the claim
        task = process_fact.delay(fact.id)  # Start the Celery task
        return Response({'task_id': task.id}, status=status.HTTP_202_ACCEPTED)  # Return the task ID
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_task_status(request, task_id):
    task = AsyncResult(task_id)
    if task.state == 'SUCCESS':
        print(task.state)

        results = ResultSerializer(task.result, many=True).data
        print(results)
        return Response({'status': 'SUCCESS', 'result': results}, status=status.HTTP_200_OK)
    elif task.state == 'FAILURE':
        print(task.state)
        return Response({'status': 'FAILURE', 'error': "PROBLEM"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        print(task.state)
        return Response({'status': task.state}, status=status.HTTP_200_OK)
