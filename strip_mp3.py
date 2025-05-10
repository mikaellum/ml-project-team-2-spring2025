from tinytag import TinyTag
import librosa
import matplotlib.pyplot as plt

file_name = '04 - Girl on Fire (Instrumental Version).mp3'

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
       return header_info
   except FileNotFoundError:
       raise FileNotFoundError(f"File not found: {file_path}")
   except Exception as e:
       raise Exception(f"Error processing MP3 file: {e}")
# Example usage:
try:
   mp3_info = get_mp3_header_info(file_name)
   print(mp3_info)
   y, sr = librosa.load(file_name, mono=False)
   plt.figure()
   plt.plot(y[0])
   plt.show()
except (FileNotFoundError, Exception) as e:
   print(e)




