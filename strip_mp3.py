import os
from tinytag import TinyTag
import librosa
import pandas as pd

directory_path = '../Music Files'

# Function that pulls all header information from an mp3
def get_mp3_header_info(file_path):
   try:
       tag = TinyTag.get(file_path)
       header_info = {
           'artist': tag.artist,
           'album': tag.album,
           'title': tag.title,
           'duration': tag.duration,
           'bitrate': tag.bitrate,
           'samplerate': tag.samplerate,
           'channels': tag.channels,
           'year': tag.year,
           'genre': tag.genre,
           'track': tag.track
       }
       return pd.DataFrame([header_info])
   except FileNotFoundError:
       raise FileNotFoundError(f"File not found: {file_path}")
   except Exception as e:
       raise Exception(f"Error processing MP3 file: {e}")

# Function to open all files in a directory
def open_all_files(directory):
    df_mp3 = pd.DataFrame()
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):  # Ensure it's a file, not a subdirectory
                try:
                    with open(filepath, 'r') as file:
                        # Process the file content here
                        df_mp3_info = get_mp3_header_info(filepath)
                        y, _ = librosa.load(filepath, mono=False)
                        df_arr = pd.DataFrame(columns = ['Ch0','Ch1'])
                        df_arr.at[0, 'Ch0'] = y[0]
                        df_arr.at[0, 'Ch1'] = y[1]
                        df_mp3_row = pd.concat([df_mp3_info, df_arr], axis = 1) # Concatonate the info and array into a single row
                        df_mp3 = pd.concat([df_mp3, df_mp3_row], axis = 0, ignore_index = True)
                except Exception as e:
                    print(f"Error opening {filename}: {e}")
        return df_mp3
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    except Exception as e:
         print(f"An error occurred: {e}")
    

# Example Use:
try:
   df = open_all_files(directory_path)
   print(df)
except (FileNotFoundError, Exception) as e:
   print(e)




