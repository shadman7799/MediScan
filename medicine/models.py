from django.db import models

# Create your models here.

class prescription(models.Model):
    cropped_image = models.ImageField(
        upload_to='images', 
        null = True, 
        default=None
    )

