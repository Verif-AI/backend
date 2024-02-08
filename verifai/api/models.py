from django.db import models


# Create your models here.
class Fact(models.Model):
    content = models.TextField()
    verified = models.BooleanField(default=False)
