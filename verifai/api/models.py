from django.db import models


# Create your models here.
class Fact(models.Model):
    claim = models.TextField()
    justification = models.TextField(null=True, blank=True)
    veracity = models.BooleanField(null=True, blank=True)
    sources = models.JSONField(null=True, blank=True)

