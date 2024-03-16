import os
import cv2
import numpy as np

from . import ocr
from MediScan import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect

image_rgb = None
is_cropped = False
is_captured = False


def home(request):
    path = settings.MEDIA_ROOT

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            pass

    context = {
        'submitted': False
    }

    return render(request, 'index.html', context)


def upload_form(request):
    if request.method == 'POST':
        global image_path, is_cropped, is_captured

        print(request.FILES)
        if 'img_file' in request.FILES:
            img_file = request.FILES['img_file']

            if 'cropped' in img_file.name:
                # print('in cropped')
                is_cropped = True

            if 'capture' in img_file.name:
                # print('in captured')
                is_captured = True

            image_path = save_uploaded_image(img_file)
            print(image_path)

        return redirect('show_meds')

    return render(request, 'index.html', {})


def save_uploaded_image(img_file):
    img = cv2.imdecode(np.frombuffer(
        img_file.read(), np.uint8), cv2.IMREAD_COLOR)

    image_dir = settings.MEDIA_ROOT
    saved_image_path = os.path.join(image_dir, 'image.png')

    cv2.imwrite(saved_image_path, img)

    global image_rgb
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return saved_image_path


def show_meds(request):
    global image_rgb, is_cropped, is_captured

    if is_captured:
        med_intakes = ocr.recognize(image_rgb, True, True)
    elif is_cropped:
        med_intakes = ocr.recognize(image_rgb, False, True)
    else:
        med_intakes = ocr.recognize(image_rgb, False, False)

    context = {
        'med_data': med_intakes,
        'submitted': True,
    }

    return render(request, 'index.html', context)
