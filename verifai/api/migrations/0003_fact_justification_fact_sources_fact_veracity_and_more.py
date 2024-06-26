# Generated by Django 5.0.3 on 2024-03-22 18:04

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0002_rename_content_fact_claim_remove_fact_verified"),
    ]

    operations = [
        migrations.AddField(
            model_name="fact",
            name="justification",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="fact",
            name="sources",
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="fact",
            name="veracity",
            field=models.BooleanField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="fact",
            name="veracityScore",
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
