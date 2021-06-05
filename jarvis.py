import datetime
import pyttsx3
import speech_recognition as sr
import os
import webbrowser
import wikipedia



engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices)   #  show 2 voices in terminal when run. these are window inbuilt voices.
# print(voices[1])  # show the name of voice through which you can guess the gender. 2 voice for [1] & [0].
engine.setProperty('voice', voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishme():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good morning sir!")

    elif hour >= 12 and hour < 18:
        speak("Good afternoon sir!")

    else:
        speak("Good night sir!")

    speak("I am jarvis at your service sir!, how may i help you.")

def takecommand():
    ''' it take microphone input from user and return string outputs.
    '''
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listning.....")
        r.pause_threshold = 1
        audio = r.listen(source)
    
    try:
        print("Recronizing.......")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said {query}")
    except Exception as e:
        # print(e)
        print("say that again please....")
        return 'None'
    return query


if __name__ == '__main__':

    wishme()
    while True:
        query = takecommand().lower()

        if 'wikipedia' in query:
            speak('searching wikipedia........')
            query = query.replace('wikipedia', '')
            result = wikipedia.summary(query, sentences=2)
            print(result)
            speak('According to wikipedia..')
            speak(result)

        elif 'open youtube' in query:
            webbrowser.open('youtube.com')

        elif 'open google' in query:
            webbrowser.open('google.com')
        
        elif 'open stackflow' in query:
            webbrowser.open('stackflow.com')

        elif 'play music' in query:
            music_dir = 'D:\\mymusic'
            songs = os.listdir(music_dir)
            speak(songs)
            os.startfile(os.path.join(music_dir, songs[1]))
        elif 'stop' in query:
            exit()