import re
from glob import glob
import os, sys
import shutil

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
tasks = os.path.dirname(parent)
main = os.path.dirname(tasks)
sys.path.append(main)

import src.configs.convers.configs as configs


def parse_textgrid(file_path):
    with open(file_path, 'r', encoding="utf8", errors='ignore') as file:
        content = file.read()

    # Regular expression to find intervals
    interval_pattern = re.compile(r'intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', re.DOTALL)
    intervals = interval_pattern.findall(content)

    return intervals



def extract_text_by_intervals(intervals, interval_length=12, lagged = True):
    """
    Extract text between each consecutive intervals of specified length.
    """
    interval_length = float(interval_length)
    extracted_texts = []
    current_start = 0
    current_end = interval_length
    current_text = []

    # if lagged:
    #     lag = 0
    # else:
    #     lag = 3
    for start, end, text in intervals:
        start = float(start)
        end = float(end)

        # If the current interval is within the current time window, append text

        if start < current_end:
            current_text.append(text)

        else:
            # Append the collected text for the previous interval
            extracted_texts.append(' '.join(current_text).strip())
            current_text = [text]
            current_start = current_end
            current_end += interval_length

            # Handle the case where multiple intervals might be skipped
            while start >= current_end:
                extracted_texts.append('')
                current_start = current_end
                current_end += interval_length

    # Append the last collected text
    extracted_texts.append(' '.join(current_text).strip())

    return extracted_texts


def extract_text_from_textgrid (text_grid_files, out_dir, interval_length=12, lagged = False):
    for file_path in sorted (text_grid_files):

        intervals = parse_textgrid(file_path)
        extracted_texts = extract_text_by_intervals(intervals, 12, lagged)

        texts = []
        text_lagged = ""
        for i, text in enumerate(extracted_texts):
            text = text.replace ("$", "")
            text = text.replace ("@", "")
            text = text.replace ("#", " ")
            text = text.replace ("***", "(rire)")
            #text = re.sub("\s\s+" , " ", text)

            if text_lagged == "":
                text_lagged = text
            else:
                text_lagged = text_lagged + " # " + text

            if lagged:
                texts.append (text_lagged.strip())
            else:
                texts.append (text)


        #elements = file_path.split ('/')[-1].split('.')[0].split ('_')
        subject, sess, conv_type, conv_numb = file_path.split ('/')[-1].split('.')[0].split ('_')

        subject = "sub-%s"%subject[1:]
        sess = "convers-TestBlocks%s"%sess[-1]
        conv_numb = conv_numb[:3]

        file_name = '_'.join ([subject, sess, conv_type, conv_numb])
        #file_name = '_'.join(file_path.split('_')[:-1])
        file_name = out_dir + file_name + ".txt"


        with open(file_name, 'w') as f:
            for text in texts:
                f.write(text + '\n')




if __name__ == '__main__':

    if not os.path.exists("%s/processed_data"%configs.DATA_PATH):
        os.makedirs('%s/processed_data'%configs.DATA_PATH)

    if os.path.exists("%s/processed_data/interlocutor_text_data"%configs.DATA_PATH):
        shutil.rmtree('%s/processed_data/interlocutor_text_data'%configs.DATA_PATH)

    os.makedirs('%s/processed_data/interlocutor_text_data'%configs.DATA_PATH)

    if os.path.exists("%s/processed_data/participant_text_data"%configs.DATA_PATH):
        shutil.rmtree('%s/processed_data/participant_text_data'%configs.DATA_PATH)

    os.makedirs('%s/processed_data/participant_text_data'%configs.DATA_PATH)

    text_grid_files = glob ("%s/raw_data/transcriptions/*conversant.TextGrid"%configs.DATA_PATH, recursive=True)
    extract_text_from_textgrid (text_grid_files, "%s/processed_data/interlocutor_text_data/"%configs.DATA_PATH, interval_length=12, lagged = True)
    text_grid_files = glob ("%s/raw_data/transcriptions/*-participant.TextGrid"%configs.DATA_PATH, recursive=True)
    extract_text_from_textgrid (text_grid_files, "%s/processed_data/participant_text_data/"%configs.DATA_PATH, interval_length=12, lagged = False)
