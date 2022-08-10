import sounddevice as sd ## recording sound and make a numpy array
from scipy.io.wavfile import write
from sympy import N #save numpy array into a wav file

def record_audio_and_save(save_path,n_times=24):
    input("To start recording press enter : ")

    for i in range(12,n_times):
        fs=44100 ## sample rate of audio, analogus to frames in video
        seconds=2 ## duration of recording
        my_recording=sd.rec(int(fs * seconds),samplerate=fs,channels=2)
        sd.wait()
        write(save_path+str(i)+".wav",fs,my_recording)
        input(f"press to record next sample, to stop press command+C({i+1/n_times})")

def record_background_and_save(save_path,n_times=150):
    input("To start recording press enter : ")

    for i in range(n_times):
        fs=44100 ## sample rate of audio, analogus to frames in video
        seconds=2 ## duration of recording
        my_recording=sd.rec(int(fs*seconds),samplerate=fs,channels=2)
        sd.wait()
        write(save_path+str(i)+".wav",fs,my_recording)
        print(f"Currently on ({i+1/n_times})")

print("recording the trigger word : \n")
record_audio_and_save("audio_data/")

# print("recording the background sound : \n")
# record_background_and_save("background_sound/")
 
