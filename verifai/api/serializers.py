from rest_framework import serializers
from .models import Fact
import json


class FactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Fact
        fields = "__all__"


class InformationItemSerializer(serializers.Serializer):
    title = serializers.CharField()
    link = serializers.URLField()
    snippet = serializers.CharField()


class ResultSerializer(serializers.Serializer):
    statement = serializers.CharField()
    judgement = serializers.CharField()
    justification = serializers.CharField()
    process_time = serializers.FloatField()
    information = serializers.CharField()  # Change this to a CharField
    message = serializers.CharField()

    def to_representation(self, instance):
        """
        Convert the 'information' field to the expected format if it contains a single 'Result' key.
        """
        ret = super().to_representation(instance)
        try:
            # Attempt to parse the 'information' field as JSON
            information = eval(ret['information'])
            # Check if it's a list with a single dictionary containing a 'Result' key
            if len(information) == 1 and information[0].get("Result", None) is not None:
                # Convert to the expected format
                ret['information'] = [{'title': information[0]['Result'], 'link': '', 'snippet': ''}]
            else:
                # Use the original format if it doesn't match the special case
                ret['information'] = information
        except (json.JSONDecodeError, TypeError):
            # If parsing fails or the format is unexpected, use an empty list
            ret['information'] = []
        return ret


class ResultListSerializer(serializers.Serializer):
    results = ResultSerializer(many=True)
