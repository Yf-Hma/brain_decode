import os, shutil
import sys
import numpy as np
import json
import argparse
import h5py
from glob import glob
import torch


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
main = os.path.dirname(parent)
sys.path.append(main)


from utils.stimulus_utils import get_story_wordseqs, get_resp, load_transcript
#from utils.utils_eval import load_transcript
import src.configs.perceived.configs as configs

np.random.seed(42)

def extract_array_between_values(array, min_value, max_value):
  """Extracts a subarray from a chronological numpy array based on min and max values.

  Args:
    array: A chronological numpy array.
    min_value: The minimum value to include in the subarray.
    max_value: The maximum value to include in the subarray.

  Returns:
    A subarray containing elements from the original array that fall within the specified min and max values.
  """

  # Find the indices of the first occurrence of min_value and the last occurrence of max_value
  min_index = np.where(array >= min_value)[0][0]
  max_index = np.where(array <= max_value)[0][-1]

  # Extract the subarray between the found indices
  subarray = array[min_index:max_index + 1]

  return subarray



def split_array_by_duration(time_array, k, start_point=None, end_point=None):
  """Splits a time numpy array into k chunks with the same duration and returns the cutoff indices.

  Args:
    time_array: A numpy array containing time values.
    k: The number of chunks to split the array into.
    start_point: The starting point for the first chunk (optional).
    end_point: The ending point for the last chunk (optional).

  Returns:
    A list of subarrays, each representing a chunk of the original array with the same duration, and a list of cutoff indices.
  """

  # Calculate the total duration of the array, taking into account start and end points
  if start_point is None:
    start_point = time_array[0]
  if end_point is None:
    end_point = time_array[-1]
  total_duration = end_point - start_point

  # Calculate the duration of each chunk
  chunk_duration = total_duration / k

  # Split the array into k chunks based on the duration and calculate cutoff indices
  chunks = []
  cutoff_indices = []
  start_time = start_point

  cutoff_indices.append(np.where(time_array >= start_point)[0][0])


  for i in range(k):
    end_time = start_time + chunk_duration
    chunk = time_array[(time_array >= start_time) & (time_array < end_time)]
    chunks.append(chunk)

    #print (np.where(time_array >= end_time)[0])
    #exit ()
    if len (np.where(time_array >= end_time)[0]) > 0:
      cutoff_indices.append(np.where(time_array >= end_time)[0][0])
    else:
      cutoff_indices.append (len (time_array))
    start_time = end_time

  return cutoff_indices

def split_array_by_cutoffs(array, cutoff_indices):
  """Splits a NumPy array into subarrays based on cutoff indices.

  Args:
    array: The NumPy array to split.
    cutoff_indices: A list of indices where the array should be split.

  Returns:
    A list of subarrays.
  """

  # Add the beginning and end indices to the cutoff indices
  #cutoff_indices = [0] + cutoff_indices + [len(array)]

  # Split the array using the cutoff indices
  subarrays = [array[cutoff_indices[i]:cutoff_indices[i+1]] for i in range(len(cutoff_indices) - 1)]

  return subarrays

def get_wordseqs_data(stories):
    word_seqs = get_story_wordseqs(stories, configs)

    new_word_sesq = []

    for story in stories:

        local_wrod_seqs = []
        j = 0

        new_time_index = word_seqs[story].tr_times[5 + configs.TRIM : -configs.TRIM] - (5 + configs.TRIM)

        for i in range (len (word_seqs[story].data)):
            if j >= len (new_time_index):
                break
            if word_seqs[story].data_times[i] <= new_time_index[j]:
                local_wrod_seqs.append (word_seqs[story].data[i])
            else:
                new_word_sesq.append (' '.join(local_wrod_seqs))
                local_wrod_seqs = []
                local_wrod_seqs.append (word_seqs[story].data[i])
                j += 1

    return new_word_sesq



def build_train_dict (args):

    # training stories
    stories = []
    with open(os.path.join(configs.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

   # fMRI responses data
    rresp = get_resp(args.subject, stories, configs, stack = True)
    n_chunks = len (rresp) // configs.CHUNKLEN
    rest_chunk = len (rresp) % configs.CHUNKLEN

    if rest_chunk > 1:
        rresp_chunked = np.vsplit(rresp[:-rest_chunk], n_chunks)
        rresp_chunked.append (rresp[-configs.CHUNKLEN:])
    else:
        rresp_chunked = np.vsplit(rresp[:], n_chunks)


    # Textual data
    rtext_data = get_wordseqs_data (stories)
    if rest_chunk > 1:
        rtext_data_chunked = np.array_split(rtext_data[:-rest_chunk], n_chunks)
        rtext_data_chunked.append (rtext_data[-configs.CHUNKLEN:])
    else:
        rtext_data_chunked = np.array_split(rtext_data[:], n_chunks)


    rtext_data_chunked = [' '.join(a) for a in rtext_data_chunked]

    folder_path = os.path.join(args.data_path, "fMRI_data_train_split", args.subject)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path)

    for num_chunk, data_chunk in enumerate (rresp_chunked):
        filename = "bold_chunk_%d.npy"%(num_chunk+1)
        filename_out = os.path.join(folder_path, filename)
        np.save(filename_out, data_chunk)

    train_dict_data = []
    for num_chunk, data_chunk in enumerate (rtext_data_chunked):
        filename = "bold_chunk_%d.npy"%(num_chunk+1)
        bold_signal_filename = os.path.join(folder_path, filename)
        entity = {"bold_signal":bold_signal_filename, "text-output":data_chunk}
        train_dict_data.append (entity)


    with open('%s/%s/train.json'%(args.data_path,args.subject), 'w') as output_file:
        json.dump(train_dict_data, output_file)


def build_test_data (args):

    with open(os.path.join(configs.DATA_TEST_DIR, "eval_segments.json"), "r") as f:
        eval_segments = json.load(f)

    test_dict_data = []

    experiments = glob(os.path.join(args.data_path, configs.DATA_TEST_DIR, "test_stimulus/*"))
    experiments = [os.path.basename(x) for x in experiments]


    for experiment in experiments:
        tasks = glob(os.path.join(configs.DATA_TEST_DIR, "test_response", args.subject, experiment, "*"))
        tasks = [os.path.basename(x).split ('.')[0] for x in tasks]


        for task in tasks:
          folder_path = os.path.join(args.data_path, "fMRI_data_test_split", args.subject, experiment, task)

          if os.path.exists(folder_path):
              shutil.rmtree(folder_path)
          os.makedirs(folder_path)

          try:
            transcript_data = load_transcript(experiment, task.split('_')[0], configs)
          except:
            continue

          # load test responses
          hf = h5py.File(os.path.join(configs.DATA_TEST_DIR, "test_response", args.subject, experiment, task + ".hf5"), "r")
          rresp = np.nan_to_num(hf["data"][:])
          n_chunks = len (rresp) // configs.CHUNKLEN
          rest_chunk = len (rresp) % configs.CHUNKLEN
          if rest_chunk > 0:
              rresp_chunked = np.vsplit(rresp[:-rest_chunk], n_chunks)
          else:
              rresp_chunked = np.vsplit(rresp[:], n_chunks)

            # load test stimuli
          ref_words, ref_times = transcript_data["words"], transcript_data["times"]

          cuttoffs = split_array_by_duration(ref_times, n_chunks, start_point= eval_segments[task][0], end_point=eval_segments[task][-1])
          text_splitted = split_array_by_cutoffs(ref_words, cuttoffs)

          test_dict_data = []
          for num_chunk, data_chunk in enumerate (rresp_chunked):
              #print (num_chunk, data_chunk.shape)
              filename = "bold_chunk_%d.npy"%(num_chunk+1)
              filename_out = os.path.join(folder_path, filename)
              np.save(filename_out, data_chunk)

              entity = {"bold_signal":filename_out, "text-output":' '.join(text_splitted[num_chunk].tolist()), "chunk_number": num_chunk+1}
              test_dict_data.append (entity)

          with open('%s/%s/test_%s_%s.json'%(args.data_path, args.subject, experiment, task), 'w') as output_file:
              json.dump(test_dict_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", '-s', type = str, required = True)
    parser.add_argument("--experiment", '-exp', type = str)
    parser.add_argument("--sessions", nargs = "+", type = int,
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()


    args.data_path = configs.PROCESSED_DATA_PATH # the path of the output processed data

    if os.path.isdir(f"{args.data_path}/{args.subject}"):
        shutil.rmtree(f"{args.data_path}/{args.subject}")
    os.mkdir(f"{args.data_path}/{args.subject}")

    build_train_dict (args)
    build_test_data (args)
