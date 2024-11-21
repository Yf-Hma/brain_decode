import re
from glob import glob
import os
import shutil

def parse_textgrid(file_path):
    with open(file_path, 'r') as file:
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
            text = re.sub("\s\s+" , " ", text)

            if text_lagged == "":
                text_lagged = text
            else:
                text_lagged = text_lagged + " # " + text

            if lagged:
                texts.append (text_lagged.strip())
            else:
                texts.append (text)

        file_name = '_'.join(file_path.split('_')[:-1])
        file_name = out_dir + file_name.split ('/')[-3] + "_" + file_name.split ('/')[-2] + ".txt"


        with open(file_name, 'w') as f:
            for text in texts:
                f.write(text + '\n')




if __name__ == '__main__':
    # Usage example

    if not os.path.exists("data/processed_data"):
        os.makedirs('data/processed_data')

    if os.path.exists("data/processed_data/interlocutor_text_data"):
        shutil.rmtree('data/processed_data/interlocutor_text_data')

    os.makedirs('data/processed_data/interlocutor_text_data')


    if os.path.exists("data/processed_data/participant_text_data"):
        shutil.rmtree('data/processed_data/participant_text_data')

    os.makedirs('data/processed_data/participant_text_data')


    text_grid_files = glob ("data/raw_data/transcriptions/**/*_right-filter.TextGrid", recursive=True)

    extract_text_from_textgrid (text_grid_files, "data/processed_data/interlocutor_text_data/", interval_length=12, lagged = True)

    text_grid_files = glob ("data/raw_data/transcriptions/**/*_left-reduc.TextGrid", recursive=True)
    extract_text_from_textgrid (text_grid_files, "data/processed_data/participant_text_data/", interval_length=12, lagged = False)
