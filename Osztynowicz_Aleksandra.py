import argparse
import json
from pathlib import Path
import cv2
from processing.utils import perform_processing
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}
    template = []
    sign = []
    #start = time.perf_counter()
    templete_dir = Path("./szablon/")
    template_paths = sorted([image_path for image_path in templete_dir.iterdir() if image_path.name.endswith('.jpg')])
    for template_path in template_paths:
        temp = cv2.imread(str(template_path), 0)
        l = len(str(template_path))
        s = str(template_path)[l - 5:l - 4]
        sign.append(s)
        template.append(temp)
    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        results[image_path.name] = perform_processing(image, template, sign)
   # end = time.perf_counter()
   # print("Czas: " + str(round(end-start,2)) + "s.")
    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
