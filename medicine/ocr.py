import os
import sys
import cv2
import numpy as np
import pandas as pd
from thefuzz import fuzz

from MediScan import settings

from tensorflow.keras.models import load_model

base_dir = os.path.join(settings.BASE_DIR, 'medicine')

model_eng_path = os.path.join(base_dir, 'models', 'model_eng_weights.h5')
model_bng_path = os.path.join(base_dir, 'models', 'model_bng_weights.h5')
model_eng = load_model(model_eng_path)
model_bng = load_model(model_bng_path)

map_eng_path = os.path.join(base_dir, 'models', 'char_map_eng.csv')
map_bng_path = os.path.join(base_dir, 'models', 'char_map_bng.csv')
en_df = pd.read_csv(map_eng_path)
bn_df = pd.read_csv(map_bng_path)
char_map_eng = en_df.set_index('Label')['Char'].to_dict()
char_map_bng = bn_df.set_index('Label')['Char'].to_dict()

type_db = ['tab', 'cap', 'syp']
time_db = ['০+০+১', '০+১+০', '০+১+১', '১+০+০', '১+০+১', '১+১+০', '১+১+১']

period_path = os.path.join(base_dir, 'assets', 'periods.csv')
period_db = pd.read_csv(period_path)
period_db = period_db['Period'].values

names_path = os.path.join(base_dir, 'assets', 'names.csv')
name_db = pd.read_csv(names_path)
name_db = [name.lower() for name in set(name_db['names'].values.astype('str'))]
name_db = list(set(name_db))

medicine_path = os.path.join(base_dir, 'assets', 'medicine.csv')
med_db = pd.read_csv(medicine_path)
med_db['brand name'] = med_db['brand name'].apply(lambda x: x.lower())
power_db = med_db['strength'].values
power_db = [str(p) for p in power_db if 'mg' in str(
    p) and '+' not in str(p) and '/' not in str(p)]
power_db = list(set(power_db))


def segment_lines_by_contours(image, dilation_kernel_size=(10, 100), avg_height_factor=0.5, avg_width_factor=0.5):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones(dilation_kernel_size, np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=3)

    contours, _ = cv2.findContours(
        dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    avg_height = np.mean([cv2.boundingRect(contour)[3]
                         for contour in contours])
    avg_width = np.mean([cv2.boundingRect(contour)[2] for contour in contours])
    contours = [contour for contour in contours if
                cv2.boundingRect(contour)[3] >= avg_height * avg_height_factor
                and cv2.boundingRect(contour)[2] >= avg_width * avg_width_factor]

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    print('contours', len(contours))

    segmented_line_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        line_segment = image[y:y+h, :]
        segmented_line_images.append(line_segment)

    return segmented_line_images


def find_jump_points(blank_rows, threshold=5):
    jump_points = []
    for i in range(1, len(blank_rows)):
        if blank_rows[i] - blank_rows[i-1] > threshold:
            jump_points.append((blank_rows[i-1], blank_rows[i]))
    return jump_points


def segment_lines_by_gaps(image, threshold=5):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)

    blank_rows = np.where(np.all(binary_img <= 10, axis=1))[0]
    print('rows:', len(blank_rows))
    jump_points = find_jump_points(blank_rows, threshold)
    print(jump_points)

    segmented_line_images = []
    for start_row, end_row in jump_points:
        line_segment = image[start_row:end_row, :]
        segmented_line_images.append(line_segment)

    return segmented_line_images


def segment_words(line_image, min_word_length=10, word_dilation_kernel=(5, 25), avg_h_factor=0.5, avg_w_factor=0.5):
    if len(line_image.shape) == 2:
        line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)

    gray_line = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    _, binary_line = cv2.threshold(
        gray_line, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    word_dilation_kernel = np.ones(word_dilation_kernel, np.uint8)
    dilated_words = cv2.dilate(binary_line, word_dilation_kernel, iterations=3)

    (contours, _) = cv2.findContours(dilated_words,
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    avg_height = np.mean([cv2.boundingRect(contour)[3]
                         for contour in contours])
    avg_width = np.mean([cv2.boundingRect(contour)[2] for contour in contours])

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    word_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h < avg_height * avg_h_factor and w < avg_width * avg_w_factor:
            continue

        if w > min_word_length:
            word = line_image[y:y + h, x:x + w]
            word_list.append(word)

    return word_list


def segment_characters(word_image, min_char_width=10, space_threshold=2, min_contour_area=100):
    image_gray = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    characters = []
    current_char = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w > min_char_width and cv2.contourArea(contour) > min_contour_area:
            if current_char:
                max_height = max(char.shape[0] for char in current_char)
                current_char_resized = [cv2.resize(
                    char, (char.shape[1], max_height)) for char in current_char]
                characters.append(np.hstack(current_char_resized))
                current_char = []

            characters.append(image_gray[y:y+h, x:x+w])
        elif w <= min_char_width and w > space_threshold:
            current_char.append(image_gray[y:y+h, x:x+w])

    if current_char:
        max_height = max(char.shape[0] for char in current_char)
        current_char_resized = [cv2.resize(
            char, (char.shape[1], max_height)) for char in current_char]
        characters.append(np.hstack(current_char_resized))

    eroded_characters = [cv2.erode(char, (2, 2), iterations=1)
                         for char in characters]

    print('Chars:', len(eroded_characters))

    return eroded_characters


def fix_char_err(term, is_eng=True):
    # print('before:', term)
    if is_eng:
        term = term.replace('8', 'e')
        term = term.replace('5', 's')
        term = term.replace('0', 'o')
        term = term.replace('9', 'a')
        term = term.replace('4', 't')
        term = term.replace('6', 't')
        term = term.replace('+', 't')
        term = term.replace('+', 'l')
        term = term.replace('(', 'l')
        term = term.replace(')', 'l')
        term = term.replace('1', 'l')
        term = term.replace('oi', 'i')
        term = term.replace('io', 'i')
        term = term.replace('ol', 'i')
        term = term.replace('lo', 'i')
        term = term.replace('0l', 'i')
        term = term.replace('l0', 'i')
        term = term.replace('oh', 'i')
        term = term.replace('ho', 'i')
        term = term.replace('ql', 'i')
        term = term.replace('lq', 'i')
        term = term.replace('lg', 'i')
        term = term.replace('gl', 'i')
    else:
        term = term.replace('ত', '৩')
        term = term.replace('অ', '৩')
        term = term.replace('হ্ম', 'মা')
        term = term.replace('ম্ম', 'মা')
        term = term.replace('শ', 'মা')
        term = term.replace('আ', 'মা')
        term = term.replace('উ', 'দি')
        term = term.replace('ছ', 'দি')
        term = term.replace('ট', 'দি')
        term = term.replace('ল', 'ন')
        term = term.replace('ড', '৫')

    # print('after:', term)

    return term.lower()


def fix_digit_err(term):
    # print('before:', term)

    term = term.replace('l', '1')
    term = term.replace('y', '1')
    term = term.replace('i', '1')
    term = term.replace('(', '1')
    term = term.replace(')', '1')
    term = term.replace('z', '2')
    term = term.replace('o', '0')
    term = term.replace('u', '0')
    term = term.replace('d', '0')

    # print('after:', term)

    return term.lower()


def predict_medicine(characters):
    global model_eng, char_map_eng
    predicted_chars = []

    for char in characters:
        max_dim = max(char.shape[0], char.shape[1])
        scale_factor = 52.0 / max_dim
        resized_char = cv2.resize(char, None, fx=scale_factor, fy=scale_factor)

        pad_top = (64 - resized_char.shape[0]) // 2
        pad_bottom = 64 - resized_char.shape[0] - pad_top
        pad_left = (64 - resized_char.shape[1]) // 2
        pad_right = 64 - resized_char.shape[1] - pad_left

        resized_char = cv2.copyMakeBorder(
            resized_char, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)

        resized_char = resized_char.astype('float32') / 255.0
        resized_char = np.expand_dims(resized_char, axis=0)

        prediction = model_eng.predict(resized_char)
        predicted_class = np.argmax(prediction)
        predicted_char = char_map_eng[predicted_class]

        predicted_chars.append(predicted_char)

    return ''.join(predicted_chars)


def predict_schedule(characters):
    global model_eng, model_bng, char_map_eng, char_map_bng
    bengali_digits = ['০', '১', '২', '৩', '৪']
    predicted_chars = []

    for i, char in enumerate(characters):
        resized_char = cv2.resize(char, (64, 64))
        resized_char = resized_char.astype('float32') / 255.0
        resized_char = np.expand_dims(resized_char, axis=0)

        prediction = model_eng.predict(resized_char)
        predicted_class = np.argmax(prediction)
        predicted_char = char_map_eng[predicted_class]

        if predicted_char in ['+', 'T', '4', '7']:
            continue

        prediction = model_bng.predict(resized_char)
        predicted_class = np.argmax(prediction)
        predicted_char = char_map_bng[predicted_class]

        predicted_chars.append(predicted_char)

    predicted_chars = [
        char for char in predicted_chars if char in bengali_digits]

    return '+'.join(predicted_chars)


def predict_period(characters):
    global model_bng, char_map_bng
    predicted_chars = []

    for char in characters:
        resized_char = cv2.resize(char, (64, 64))
        resized_char = resized_char.astype('float32') / 255.0
        resized_char = np.expand_dims(resized_char, axis=0)

        prediction = model_bng.predict(resized_char)
        predicted_class = np.argmax(prediction)
        predicted_char = char_map_bng[predicted_class]

        predicted_chars.append(predicted_char)

    return ' '.join(predicted_chars)


def pred_optimize(intake):
    global type_db, name_db, power_db, time_db, period_db

    threshold = 25

    med_type = intake[0]
    max_type_ratio = max(type_db, key=lambda x: fuzz.ratio(med_type, x))
    prob_type = max_type_ratio.capitalize() if fuzz.ratio(
        med_type, max_type_ratio) >= 0 else med_type

    med_name = intake[1]
    max_name_ratio = max(name_db, key=lambda x: fuzz.ratio(med_name, x))
    prob_med_name = max_name_ratio.title() if fuzz.ratio(
        med_name, max_name_ratio) >= threshold else med_name

    prob_power = None
    if len(intake) > 4:
        power = intake[-3]
        if power != 'N/A':
            max_power_ratio = max(power_db, key=lambda x: fuzz.ratio(power, x))
            prob_power = max_power_ratio.capitalize() if fuzz.ratio(
                power, max_power_ratio) >= threshold else power
        else:
            prob_power = power

    schedule = intake[-2]
    max_schedule_ratio = max(time_db, key=lambda x: fuzz.ratio(schedule, x))
    prob_schedule = max_schedule_ratio if fuzz.ratio(
        schedule, max_schedule_ratio) >= 80 else schedule

    ben_digits = ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']
    periods = ['দিন', 'সপ্তাহ', 'মাস']

    period = intake[-1]
    num = ''.join([p for p in period if p in ben_digits])
    rest = ''.join([p for p in period if p not in ben_digits])
    rest = max(periods, key=lambda x: fuzz.ratio(rest, x))
    period = ' '.join([num, rest])
    max_period_ratio = max(period_db, key=lambda x: fuzz.ratio(period, x))
    prob_period = max_period_ratio if fuzz.ratio(
        period, max_period_ratio) >= threshold else period

    return (prob_type, prob_med_name, prob_power, prob_schedule, prob_period)


def threshold_n_resize(image_rgb, target_width=1920):
    img_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img_inverted = cv2.bitwise_not(img_binary)

    aspect_ratio = img_inverted.shape[1] / img_inverted.shape[0]
    target_height = int(target_width / aspect_ratio)
    resized_img = cv2.resize(img_inverted, (target_width, target_height))

    return cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


def get_price(name):
    rows = med_db.loc[med_db['brand name'] == name.lower()]
    rows = rows.dropna()
    
    if not rows.empty:
        row = rows.iloc[-1]
        price_info = row['Package Size']

        try:
            last_part = price_info.split(',')[-1].strip()
            last_part = last_part.replace('(', '')
            last_part = last_part.replace(')', '')
            price = last_part.split('.')[0].strip()
            return price.split(':')
        except Exception:
            pass

    return 'Not Available', 'Not Available'


def recognize(image_rgb, is_captured, is_cropped):
    if is_cropped or is_captured:
        if is_captured:
            image_rgb = threshold_n_resize(image_rgb)

        lines = segment_lines_by_gaps(image_rgb)
        if not lines:
            lines = segment_lines_by_contours(
                image_rgb, dilation_kernel_size=(2, 100))
    else:
        lines = segment_lines_by_contours(
            image_rgb, dilation_kernel_size=(10, 200))

    print('Lines:', len(lines))

    med_list = [segment_words(line) for line in lines]
    for i, med in enumerate(med_list, start=1):
        print(i, len(med))

    med_intakes = []
    for i, image_row in enumerate(med_list):
        med_type = image_row[0]
        med_name = image_row[1]
        schedule = image_row[-2]
        period = image_row[-1]

        # original_stdout = sys.stdout
        # sys.stdout = open('/dev/null', 'w')

        med_type = predict_medicine(segment_characters(med_type)).lower()
        if len(med_type) > 3:
            med_type = med_type[:-1]

        med_name = predict_medicine(segment_characters(med_name)).lower()
        med_name = fix_char_err(med_name)

        if len(image_row) > 4:
            power = predict_medicine(segment_characters(image_row[2])).lower()
            power = fix_digit_err(power)

        schedule = predict_schedule(segment_characters(schedule))

        period = predict_period(segment_characters(period))
        period = fix_char_err(period, False)

        # sys.stdout = original_stdout

        if len(image_row) > 4:
            intake = (med_type, med_name, power, schedule, period)
        else:
            intake = (med_type, med_name, 'N/A', schedule, period)

        print(f'{i+1}.', intake)
        intake = pred_optimize(intake)
        print(f'{i+1}.', intake, '(prediction optimzer)')

        size, price = get_price(intake[1])

        intake = {
            'Type': intake[0],
            'Name': intake[1],
            'Power': intake[2],
            'Schedule': intake[3],
            'Duration': intake[4],
            'Price': price,
            'Size': size.replace('p', 'P')
        }

        med_intakes.append(intake)

    return med_intakes
