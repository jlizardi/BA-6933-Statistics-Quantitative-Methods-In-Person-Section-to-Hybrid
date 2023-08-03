# -*- coding: utf-8 -*-
"""
*add games
add food stuff add ,
Alarm Clock*
Content Aggregator
SUDOKU GAME
 add camras
# add yoga timer with music
#schduler for wake up and go to bed + all reminders
story teller secirity system
#sentament analysis , face recognition 
#weater alerts/emergeny
Created on Tue Jun 20 10:45:27 2023

@author: Joshua LLizardi
"""

# Import the necessary packages
import random
import os
import time
import requests
import pyttsx3
import speech_recognition
import webbrowser
import datetime
import warnings

warnings.filterwarnings("ignore")
import platform
import bs4
import GPUtil
import psutil
import cv2
import numpy as np
import dlib
import pyautogui
import face_recognition
import ray




#######################################################################
#######################################################################
#######################################################################
#######################################################################
# Speech Recognition
def SeeScreen():

    # Specify resolution
    resolution = (1920, 1080)

    # Specify video codec
    codec = cv2.VideoWriter_fourcc(*"XVID")

    # Specify name of Output file
    filename = "Recording.avi"

    # Specify frames rate. We can choose any
    # value and experiment with it
    fps = 60.0


    # Creating a VideoWriter object
    out = cv2.VideoWriter(filename, codec, fps, resolution)

    # Create an Empty window
    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    # Resize this window
    cv2.resizeWindow("Live", 480, 270)

    while True:
        # Take screenshot using PyAutoGUI
        img = pyautogui.screenshot()

        # Convert the screenshot to a numpy array
        frame = np.array(img)

        # Convert it from BGR(Blue, Green, Red) to
        # RGB(Red, Green, Blue)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the coordinates
        detector = dlib.get_frontal_face_detector()


        # Capture frames continuously

            # Capture frame-by-frame
            #ret, #frame = cap.read()
        frame = cv2.flip(frame, 1)

            # RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

            # Iterator to count faces
        i = 0
        for face in faces:

                # Get the coordinates of faces
                x, y = face.left(), face.top()
                x1, y1 = face.right(), face.bottom()
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                # Increment iterator for each face in faces
                i = i+1

                # Display the box and faces
                cv2.putText(frame, 'face num'+str(i), (x-10, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(face, i)


            # Display the resulting frame
        cv2.imshow('Live', frame)

            # This command let's us quit with the "q" button on a keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


	# Release the capture and destroy the windows

@ray.remote
def SeeWebCam():

    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    Josh_image = face_recognition.load_image_file("C:\\Users\\HP\\Pictures\\Joshua-Lizardi-300x300.jpg",mode='RGB')
    josh_face_encoding = face_recognition.face_encodings(Josh_image)[0]

    # Load a second sample picture and learn how to recognize it.
    Angie_image = face_recognition.load_image_file("C:\\Users\\HP\\Desktop\\angie.jpg",mode='RGB')
    angie_face_encoding = face_recognition.face_encodings(Angie_image)[0]

    Misty_image = face_recognition.load_image_file("C:\\Users\\HP\\Desktop\\misty.jpg",mode='RGB')
    misty_face_encoding = face_recognition.face_encodings(Misty_image)[0]

    #jack_image = face_recognition.load_image_file("C:\\Users\\HP\\Desktop\\jack.jpg",mode='RGB')
    #jack_face_encoding = face_recognition.face_encodings(jack_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        josh_face_encoding,
        angie_face_encoding,
        misty_face_encoding,
        #jack_face_encoding
    ]
    known_face_names = [
        "Joshua",
        "Angie",
        "Misty",
        #"Jack"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            #rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #first_match_index = matches.index(True)
                #name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def takeCommand():
    r = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=2)
        print(
            random.choice(
                (
                    "I am Listening",
                    "Awaiting your command",
                    "what?",
                )
            )
        )
        r.non_speaking_duration = 0.2
        r.pause_threshold = 0.6
        audio = r.listen(source)
        try:
            Query = r.recognize_google(audio, language="en-in")
            print(f"I herd you say {Query}")
            speak("I am Thinking")
        except Exception:
            speak(
                random.choice(
                    (
                        "Say that again...",
                        "I did not hear you.",
                        "i din't get that",
                        "I am not sure I understand you fully",
                        "What?...",
                        "Huh?",
                        "Sorry",
                        "I’m afraid I don’t follow you",
                        "Excuse me, could you repeat the question?",
                        "I’m sorry ",
                        " I don’t understand. ",
                        "Could you say it again?",
                        "I’m sorry?",
                        " I didn’t catch that",
                        "Would you mind speaking more slowly?",
                        "I’m confused.",
                        "Could you tell me that again?",
                        "I’m sorry, I didn’t understand.",
                        "Could you repeat a little louder, please?",
                        "I didn’t hear you.",
                        "Please could you say it again?",
                        "Sorry? ",
                        "Sorry, what? ",
                        "Scuse me?  ",
                        "Huh?  ",
                        "I don’t get it’ or ",
                        "I don’t understand",
                        "I can’t hear you",
                        "What? ",
                        "Eh? ",
                        "Hmm? ",
                        "i guess im deaf",
                        "Come again?",
                        "Say what? ",
                        "Pass that by me again?",
                        "You what? ",
                        "I don’t get it… ",
                        "I can’t make head nor tail of what you’re saying.",
                        "This is all Greek to me.",
                        "Sorry, this is as clear as mud to me.",
                    )
                )
            )
            return "None"
        return Query


# Text To Speech
def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    engine.say(audio)
    engine.runAndWait()


# Bot Intro Sounds
def music():
    import vlc

    os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")
    player = vlc.MediaPlayer("C:\\Users\\HP\\music.mp3")
    player.play()
    player.audio_set_volume(80)
    time.sleep(10)
    player = vlc.MediaPlayer("C:\\Users\\HP\\music2.mp3")
    player.audio_set_volume(35)
    player.play()


# user list of bot commands
def cmands():
    from tabulate import tabulate

    print(
        tabulate(
            [
                [
                    "work",
                    "pray",
                    "we doing",
                    "my schedule",
                    "calendar",
                    "need you",
                    "quote",
                    "what time",
                    "sleep",
                    "be quiet",
                    "my files",
                    "my shows",
                    "lost my phone",
                    "gym",
                    "don't listen",
                    "stop listeng",
                    "open youtube",
                    "open Google",
                    "open stack overflow",
                    "data analysis mode",
                    "play some music",
                    "play a song",
                    "chill party",
                    "play music",
                    "play my song",
                    "dance party",
                    "bored",
                    "entertaine",
                    "who made you",
                    "who created you",
                    "joke",
                    "good morning",
                    "i'm good",
                    "will you be my gf",
                    "will you be my bf",
                    "hows life",
                    "i love you",
                    "mrs. smith",
                    "who am i",
                    "what am i",
                    "yes",
                    "is love",
                    "who are you",
                    "your name",
                    "about yourself",
                    "why are you here",
                    "i need information on",
                    "i want to know about",
                    "weather",
                    "rain",
                    "snow",
                    "question",
                    "what is",
                    "who is",
                    "wikipedia",
                    "where is",
                    "calculate",
                    "search for",
                    "look up",
                    "don't need anything",
                    "go away",
                    "exit",
                    "fuck off",
                    "bye",
                    "later",
                ]
            ],
            headers=["Commands List"],
        )
    )


# bot talking face fix windowsize
def resize():
    import ctypes

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    handle = user32.FindWindowW(None, "Kauna")
    user32.ShowWindow(handle, 10)
    user32.MoveWindow(handle, 1400, 10, 400, 400, True)
    time.sleep(5)

    handle = user32.FindWindowW(None, "PenGUIn BOT")
    user32.ShowWindow(handle, 10)
    user32.MoveWindow(handle, 1400, 10, 100, 100, True)
    time.sleep(5)


#######################################################################
#######################################################################
# math stuff


def vscode():
    os.startfile("C:\\Users\HP\\AppData\\Local\Programs\\Microsoft VS Code\\Code.exe")

def rstudio():
    os.startfile("C:\\Program Files\\RStudio\\rstudio.exe")

def bluestacks():
    from AppOpener import open, close
    open("BlueStacks", match_closest=True)

def mathAch():
    webbrowser.open("https://columbus.mathplus.online/home")
    webbrowser.open("https://www.desmos.com/calculator")
    webbrowser.open("https://appx.wheniwork.com/timesheets")
    webbrowser.open("https://www.coolmathgames.com/0-number-solver#immersiveModal")
    os.startfile("C://Users//HP//AppData//Roaming//Zoom//bin//Zoom.exe")


def math():

    webbrowser.open("https://stackoverflow.com")
    webbrowser.open("https://statskingdom.com")
    webbrowser.open("https://wolframalpha.com")
    webbrowser.open(
        "https://raw.githubusercontent.com/aaronwangy/Data-Science-Cheatsheet/main/images/page1-1.png"
    )
    webbrowser.open(
        "https://raw.githubusercontent.com/aaronwangy/Data-Science-Cheatsheet/main/images/page2-1.png"
    )


# data analysis
def DA():
    webbrowser.open("https://youtu.be/pmxYePDPV6M?t=1")
    time.sleep(30)
    import dtale
    import pandas as pd

    os.startfile("D:\DataSets")
    speak("Opening your datasets folder")
    speak("input a file name when ready")
    try:
        data = input("Enter a filename: ")
        data = pd.read_csv(data)
        d = dtale.show(data)
        d.open_browser()
    except FileNotFoundError as e:
        print(f"File Not Found... try deleting quotation marks\n" f"{e}")
    speak(
        "Before you can dive into the data, there are several questions that need to be answered first. These questions will help you understand if you have right the kind of data for your goals."
    )
    speak(
        "Who collected the data? How was the data collected? Test for sampling bias, outliers & missing values. Can the data measure what we desire to be measure?"
    )
    speak("Opening V-S-Code and R-Studio")
    rstudio()
    vscode()
    speak("I'll let you work...tap me when you need me")
    ZZZ()


# work emails and calenders
def work():
    speak("Opening work ")
    speak(
        " Ok... What are we working on today? Keep in mind you work for Trine University, Math Academy"
    )
    webbrowser.open("https://outlook.office.com/mail/")
    webbrowser.open("https://moodle.trine.edu")
    webbrowser.open("https://mail.google.com/mail/u/0/#inbox")
    webbrowser.open("https://www.youtube.com/watch?v=xMmbyzZ4nHc")
    time.sleep(15)
    webbrowser.open("https://columbus.mathplus.online/home")
    tellDay()
    speak("I'm Pulling up your Calender now...")
    webbrowser.open("https://calendar.google.com/calendar/u/0/r?pli=1")
    weather()


def inventory():
    import pickle

    remaining_items = {}

    # Save and load functions for remaining items items and used_item_groups

    def save_obj(obj, name):
        current_directory_is = os.getcwd()
        final_directory_is = os.path.join(current_directory_is, r"FoodMonitor")
        if not os.path.exists(final_directory_is):
            os.makedirs(final_directory_is)

        with open(f"FoodMonitor/{name}.pkl", "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        with open(f"FoodMonitor/{name}.pkl", "rb") as f:
            return pickle.load(f)

    # Standard multi choice question template
    def multi_choice_question(options: list):
        while True:
            print(
                "Enter the number of your choice -",
                *(f"{number}. {option}" for number, option in enumerate(options, 1)),
                sep="\n",
                end="\n\n",
            )
            """
            The same as - 
            print("\nEnter the number of your choice - ")
            for i, option in enumerate(options, 1):
                print('{0}. {1}'.format(i, option))
            print("\n"")        
            """
            try:
                answer = int(input())
                if 1 <= answer <= len(options):
                    return answer
                print("That option does not exist! Try again!")
            except ValueError:
                print("Doesn't seem like a number! Try again!")

    # For getting an int from the user'
    def get_int():
        while True:
            try:
                return int(input())
            except ValueError:
                print("Doesn't seem like a number! Try again!")

    """ UI methods - For direct interaction with the user"""

    def buy_items():
        if len(remaining_items) != 0:
            print("What item did you buy?\n")
            multi_choice_params = []
            for name, amount in remaining_items.items():
                multi_choice_params.append(name)
            item_name = multi_choice_params[
                multi_choice_question(multi_choice_params) - 1
            ]
            print("How much did you buy?")
            item_amount = get_int()
            remaining_items[item_name] += item_amount
            print("Successfully updated item library")
        else:
            print("No item yet! Use option 4 to add item.")

        time.sleep(1)

    def edit_items():
        def new_item():
            while True:
                name = input("What will the name of the new item be?").lower().strip()
                if name in remaining_items:
                    print("item already exists! Try again!")
                    continue
                break

            print("How much of this do you currently have?")
            amount = get_int()
            remaining_items[name] = amount
            save_obj(remaining_items, "I")
            print(f"Successfully added {name}")

        def edit_item_quantity():
            if len(remaining_items) != 0:
                print("Which item's amount do you want to change?")
                to_change = multi_choice_question(list(remaining_items.keys()))
                print("What will the new amount of the item?")
                new_value = get_int()
                remaining_items[list(remaining_items.keys())[to_change - 1]] = new_value
                print(
                    f"Success! {list(remaining_items.keys())[to_change - 1]} now has value {new_value}"
                )
            else:
                print("No items yet! Use option 4 to add new items.")

        to_do = multi_choice_question(
            ["Add a new item", "Edit the amount of an existing item"]
        )

        if to_do == 1:
            new_item()
        else:
            edit_item_quantity()
        time.sleep(1)

    def get_items():
        print("Current item Inventory:\n")
        if len(remaining_items) == 0:
            print("No items yet!\n")
        else:
            for name, amount in remaining_items.items():
                print(f"{name} : {amount}")
        time.sleep(1)
        print("            ")

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r"FoodMonitor")
    if os.path.exists(final_directory):
        remaining_items = load_obj("I")
    flag = True
    while flag is True:
        choice = multi_choice_question(
            [
                "If you bought items, choose option 1. ",
                "If you used an item or finished an item choose option 2. ",
                "To view current item inventory, choose option 3",
                "To save inventory and quit the application, choose option 4. ",
            ]
        )

        if choice == 1:
            buy_items()

        if choice == 2:
            edit_items()

        if choice == 3:
            get_items()

        if choice == 4:
            speak("closing inventory...")
            print("closing inventory...")
            save_obj(remaining_items, "I")
            import sys

            with open("C:\\Users\\HP\\Documents\\KMSdatabase.txt", "w") as f:
                sys.stdout = f
                print(get_items())
                sys.stdout = sys.__stdout__
                flag = False


def currentinventroy():
    from discord_webhook import DiscordWebhook
    import aspose.words as aw

    doc = aw.Document("KMSdatabase.txt")

    for page in range(doc.page_count):
        extractedPage = doc.extract_pages(page, 1)
        extractedPage.save("C:\\Users\\HP\\Output.png")

    webhook = DiscordWebhook(
        url="https://discord.com/api/webhooks/1126136006433841182/6yBecHJdeW2wadK_2hovARKlUo485TXmtTzObjo6Z3DQ8mBG6FjbdAnPpyyITuBslIT7",
        username="Webhook with files",
    )
    # send two images
    with open("C:\\Users\\HP\\Output.png", "rb") as f:
        webhook.add_file(file=f.read(), filename="example.png")
        webhook.execute()


# morning prayer set up
def pray():
    speak(random.choice(("Tell God I said Hi!")))
    gita()
    webbrowser.open("www.https://www.youtube.com/watch?v=rwxd9Q0qp4o.com")
    webbrowser.open("https://www.youtube.com/@BhaktiMarga/streams")
    webbrowser.open("https://www.youtube.com/watch?v=BfrMutwbciQ&t=1s")
    webbrowser.open("https://www.youtube.com/watch?v=SC05lp6fkek")
    ZZZ()


# google-calenders
def calender():
    speak(random.choice(("I'm Pulling up your Calender now...", "ok no problem")))
    webbrowser.open("https://calendar.google.com/calendar/u/0/r?pli=1")


# daily todo
def tellDay():
    day = datetime.datetime.now().weekday() + 1
    Day_dict = {
        1: "Monday, Take Attendance for trine today,Start Grading for Trine ",
        2: "Tuesday Trine Faculty development meetings around 11am",
        3: "Wednesday send Trine absence notifications You have Office Hours today starting at 11am trine on zoom",
        4: "Thursday, You have an Advisors meeting at 1pm for Trine & You have Math Academy tonight 4:30 to 5:30 ",
        5: "Friday",
        6: "Saturday-You have Math Academy 9:30am to 11:30am",
        7: "Sunday",
    }
    if day in Day_dict.keys():
        day_of_the_week = Day_dict[day]
        print(day_of_the_week)
        speak("It is " + day_of_the_week)
        speak("dont forget to check calenders and to do list")
        speak("fun fact!")
        facts()


# weather
def weather():
    url = "https://www.google.com/search?q=" + "weather" + "fort_wayne"
    html = requests.get(url).content
    soup = bs4.BeautifulSoup(html, "html.parser")
    temp = soup.find("div", attrs={"class": "BNeawe iBp4i AP7Wnd"}).text
    strr = soup.find("div", attrs={"class": "BNeawe tAd8D AP7Wnd"}).text
    data = strr.split("\n")
    times = data[0]
    sky = data[1]
    print(
        "Temperature is "
        + temp
        + " ."
        + " Time: "
        + times
        + "."
        + "  Sky Description: "
        + sky
    )
    speak(
        "Temperature is "
        + temp
        + " ."
        + " Time: "
        + times
        + "."
        + "  Sky Description: "
        + sky
    )
    city = "fort wayne"
    url = "https://wttr.in/{}".format(city)
    res = requests.get(url)
    speak("Displaying Weater report for: fort wayne in window")
    print(res.text)
    speak(" Ok... tap me when you need me")
    ZZZ()


# time
def Time():
    t = time.localtime()
    return time.strftime("%m/%d/%Y, %I:%M %p", t)


# bot workout
def exorcise():
    import workout

    webbrowser.open("https://youtu.be/QY7TyMo9JCM=1")
    speak("are you ready?...")
    speak("start with a stretch")
    workout.announce("ok and ... Stretch")
    speak("ok get ready!")
    speak("five")
    speak("four")
    speak("three")
    speak("two")
    speak("one!")
    speak("get ready for up downs")
    workout.announce("up downs")
    speak("get ready for jumping jacks")
    workout.announce("Jumping jacks")
    workout.announce("Rest!")
    speak("ok get ready for shadow boxing!")
    speak("five")
    speak("four")
    speak("three")
    speak("two")
    speak("one!")
    workout.announce("Shadow boxing")
    workout.announce("ok rest!!")
    speak("five")
    speak("four")
    speak("three")
    speak("two")
    speak("one!")
    speak("...and ... done")
    speak("get ready for push up")
    workout.announce("push ups")
    workout.announce("ok rest!!")
    speak("five")
    speak("four")
    speak("three")
    speak("two")
    speak("one!")
    speak("get ready for jump rope")
    workout.announce("jump rope!!")
    speak("five")
    speak("four")
    speak("three")
    speak("two")
    speak("one!")
    speak("get ready for weights")
    workout.announce("10 pound weight")
    speak("ok get ready!")
    speak("five")
    speak("four")
    speak("three")
    speak("two")
    speak("one!")
    workout.announce("Stretch")
    speak("get ready for jumping jacks")
    workout.announce("Jumping jacks")
    workout.announce("Rest!")
    speak("ok get ready!")
    speak("five")
    speak("four")
    speak("three")
    speak("two")
    speak("one!")
    speak("almost done! get ready for up shadow boxing")
    workout.announce("Shadow boxing")
    speak("ok rest!!")
    speak("five")
    speak("four")
    speak("three")
    speak("two")
    speak("one!")
    speak("done")
    speak("...and ... done")
    speak("great job you finished your workout!")


#######################################################################
#######################################################################
def Sudoku():
    import pygame

    pygame.font.init()
    Window = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("SUDOKU GAME by DataFlair")
    x = 0
    z = 0
    diff = 500 / 9
    value = 0
    defaultgrid = [
        [0, 0, 4, 0, 6, 0, 0, 0, 5],
        [7, 8, 0, 4, 0, 0, 0, 2, 0],
        [0, 0, 2, 6, 0, 1, 0, 7, 8],
        [6, 1, 0, 0, 7, 5, 0, 0, 9],
        [0, 0, 7, 5, 4, 0, 0, 6, 1],
        [0, 0, 1, 7, 5, 0, 9, 3, 0],
        [0, 7, 0, 3, 0, 0, 0, 1, 0],
        [0, 4, 0, 2, 0, 6, 0, 0, 7],
        [0, 2, 0, 0, 0, 7, 4, 0, 0],
    ]

    font = pygame.font.SysFont("comicsans", 40)
    font1 = pygame.font.SysFont("comicsans", 20)

    def cord(pos):
        global x
        x = pos[0] // diff
        global z
        z = pos[1] // diff

    def highlightbox():
        for k in range(2):
            pygame.draw.line(
                Window,
                (0, 0, 0),
                (x * diff - 3, (z + k) * diff),
                (x * diff + diff + 3, (z + k) * diff),
                7,
            )
            pygame.draw.line(
                Window,
                (0, 0, 0),
                ((x + k) * diff, z * diff),
                ((x + k) * diff, z * diff + diff),
                7,
            )

    def drawlines():
        for i in range(9):
            for j in range(9):
                if defaultgrid[i][j] != 0:
                    pygame.draw.rect(
                        Window, (255, 255, 0), (i * diff, j * diff, diff + 1, diff + 1)
                    )
                    text1 = font.render(str(defaultgrid[i][j]), 1, (0, 0, 0))
                    Window.blit(text1, (i * diff + 15, j * diff + 15))
        for l in range(10):
            if l % 3 == 0:
                thick = 7
            else:
                thick = 1
            pygame.draw.line(Window, (0, 0, 0), (0, l * diff), (500, l * diff), thick)
            pygame.draw.line(Window, (0, 0, 0), (l * diff, 0), (l * diff, 500), thick)

    def fillvalue(value):
        text1 = font.render(str(value), 1, (0, 0, 0))
        Window.blit(text1, (x * diff + 15, z * diff + 15))

    def raiseerror():
        text1 = font.render("wrong!", 1, (0, 0, 0))
        Window.blit(text1, (20, 570))

    def raiseerror1():
        text1 = font.render("wrong ! enter a valid key for the game", 1, (0, 0, 0))
        Window.blit(text1, (20, 570))

    def validvalue(m, k, l, value):
        for it in range(9):
            if m[k][it] == value:
                return False
            if m[it][l] == value:
                return False
        it = k // 3
        jt = l // 3
        for k in range(it * 3, it * 3 + 3):
            for l in range(jt * 3, jt * 3 + 3):
                if m[k][l] == value:
                    return False
        return True

    def solvegame(defaultgrid, i, j):

        while defaultgrid[i][j] != 0:
            if i < 8:
                i += 1
            elif i == 8 and j < 8:
                i = 0
                j += 1
            elif i == 8 and j == 8:
                return True
        pygame.event.pump()
        for it in range(1, 10):
            if validvalue(defaultgrid, i, j, it) == True:
                defaultgrid[i][j] = it
                global x, z
                x = i
                z = j
                Window.fill((255, 255, 255))
                drawlines()
                highlightbox()
                pygame.display.update()
                pygame.time.delay(20)
                if solvegame(defaultgrid, i, j) == 1:
                    return True
                else:
                    defaultgrid[i][j] = 0
                Window.fill((0, 0, 0))

                drawlines()
                highlightbox()
                pygame.display.update()
                pygame.time.delay(50)
        return False

    def gameresult():
        text1 = font.render("game finished", 1, (0, 0, 0))
        Window.blit(text1, (20, 570))

    flag = True
    flag1 = 0
    flag2 = 0
    rs = 0
    error = 0
    while flag:
        Window.fill((255, 182, 193))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                flag1 = 1
                pos = pygame.mouse.get_pos()
                cord(pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x -= 1
                    flag1 = 1
                if event.key == pygame.K_RIGHT:
                    x += 1
                    flag1 = 1
                if event.key == pygame.K_UP:
                    y -= 1
                    flag1 = 1
                if event.key == pygame.K_DOWN:
                    y += 1
                    flag1 = 1
                if event.key == pygame.K_1:
                    value = 1
                if event.key == pygame.K_2:
                    value = 2
                if event.key == pygame.K_3:
                    value = 3
                if event.key == pygame.K_4:
                    value = 4
                if event.key == pygame.K_5:
                    value = 5
                if event.key == pygame.K_6:
                    value = 6
                if event.key == pygame.K_7:
                    value = 7
                if event.key == pygame.K_8:
                    value = 8
                if event.key == pygame.K_9:
                    value = 9
                if event.key == pygame.K_RETURN:
                    flag2 = 1
                if event.key == pygame.K_r:
                    rs = 0
                    error = 0
                    flag2 = 0
                    defaultgrid = [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                if event.key == pygame.K_d:
                    rs = 0
                    error = 0
                    flag2 = 0
                    defaultgrid = [
                        [0, 0, 4, 0, 6, 0, 0, 0, 5],
                        [7, 8, 0, 4, 0, 0, 0, 2, 0],
                        [0, 0, 2, 6, 0, 1, 0, 7, 8],
                        [6, 1, 0, 0, 7, 5, 0, 0, 9],
                        [0, 0, 7, 5, 4, 0, 0, 6, 1],
                        [0, 0, 1, 7, 5, 0, 9, 3, 0],
                        [0, 7, 0, 3, 0, 0, 0, 1, 0],
                        [0, 4, 0, 2, 0, 6, 0, 0, 7],
                        [0, 2, 0, 0, 0, 7, 4, 0, 0],
                    ]
        if flag2 == 1:
            if solvegame(defaultgrid, 0, 0) == False:
                error = 1
            else:
                rs = 1
            flag2 = 0
        if value != 0:
            fillvalue(value)
            if validvalue(defaultgrid, int(x), int(z), value) == True:
                defaultgrid[int(x)][int(z)] = value
                flag1 = 0
            else:
                defaultgrid[int(x)][int(z)] = 0
                raiseerror1()
            value = 0

        if error == 1:
            raiseerror()
        if rs == 1:
            gameresult()
        drawlines()
        if flag1 == 1:
            highlightbox()
        pygame.display.update()

    pygame.quit()


#######################################################################
#######################################################################

# holybook text
def gita():
    speak(
        random.choice(
            (
                "As both armies stand ready for the battle the mighty warrior Arjuna on observing the warriors on both sides becomes increasingly sad and depressed due to the fear of losing his relatives and friends and the consequent sins attributed to killing his own relatives.",
                "So he surrenders to Lord Krishna seeking a solution.",
                "Thus follows the wisdom of the Bhagavad Gita.",
                "This is the most important of the Bhagavad Gita as Lord Krishna condenses the teachings of the entire Gita in this .",
                "This is the essence of the entire Gita.",
                "Arjuna completely surrenders himself to Lord Krishna and accepts his position as a disciple and Krishna as his Guru.",
                "He requests Krishna to guide him on how to dismiss his sorrow.",
                "Explanation of the main cause of all grief which is ignorance of the true nature of Self.",
                "Karma Yoga the discipline of selfless action without being attached to its fruits.",
                "Description of a Perfect Man One whose mind is steady and onepointed.",
                "Here Lord Krishna emphasizes the importance of karma in life.",
                "He reveals that it is important for every human being to engage in some sort of activity in this material world.",
                "Further he describes the kinds of actions that lead to bondage and the kinds that lead to liberation.",
                "Those persons who continue to perform their respective duties externally for the pleasure of the Supreme without attachment to its rewards get liberation at the end.",
                "In this Krishna glorifies the Karma Yoga and imparts the Transcendental Knowledge (the knowledge of the soul and the Ultimate Truth) to Arjuna.",
                "He reveals the reason behind his appearance in this material world.",
                "He reveals that even though he is eternal he reincarnates time after time to reestablish dharma and peace on this Earth.",
                "His births and activities are eternal and are never contaminated by material flaws.",
                "Those persons who know and understand this Truth engage in his devotion with full faith and eventually attain Him.",
                "They do not have to take birth in this world again.",
                "In this Krishna compares the paths of renunciation in actions (Karma Sanyas) and actions with detachment (Karma Yoga) and explains that both are means to reach the same goal and we can choose either.",
                "A wise person should perform his/her worldly duties without attachment to the fruits of his/her actions and dedicate them to God.",
                "This way they remain unaffected by sin and eventually attain liberation.",
                "In this Krishna reveals the Yoga of Meditation and how to practise this Yoga.",
                "He discusses the role of action in preparing for Meditation how performing duties in devotion purifies ones mind and heightens ones spiritual consciousness.",
                "He explains in detail the obstacles that one faces when trying to control their mind and the exact methods by which one can conquer their mind.",
                "He reveals how one can focus their mind on Paramatma and unite with the God.",
                "In this Krishna reveals that he is the Supreme Truth the principal cause and the sustaining force of everything.",
                "He reveals his illusionary energy in this material world called Maya which is very difficult to overcome but those who surrender their minds unto Him attain Him easily.",
                "He also describes the four types of people who surrender to Him in devotion and the four kinds that don't.",
                "Krishna confirms that He is the Ultimate Reality and those who realize this Truth reach the pinnacle of spiritual realization and unite with the Lord.",
                "In this Krishna reveals the importance of the last thought before death.",
                "If we can remember Krishna at the time of death we will certainly attain him.",
                "Thus it is very important to be in constant awareness of the Lord at all times thinking of Him and chanting His names at all times.",
                "By perfectly absorbing their mind in Him through constant devotion one can go beyond this material existence to Lords Supreme abode.",
                "In this Krishna explains that He is Supreme and how this material existence is created maintained and destroyed by His Yogmaya and all beings come and go under his supervision.",
                "He reveals the Role and the Importance of Bhakti (transcendental devotional service) towards our Spiritual Awakening.",
                "In such devotion one must live for the God offer everything that he possesses to Him and do everything for Him only.",
                "One who follows such devotion becomes free from the bonds of this material world and unites with the Lord.",
                "In this Krishna reveals Himself as the cause of all causes.",
                "He describes His various manifestations and opulences in order to increase Arjunas Bhakti.",
                "Arjuna is fully convinced of the Lords paramount position and proclaims him to be the Supreme Personality.",
                "He prays to Krishna to describe more of His divine glories which are like nectar to hear.",
                "In this Arjuna requests Krishna to reveal His Universal Cosmic Form that encompasses all the universes the entire existence.",
                "Arjuna is granted divine vision to be able to see the entirety of creation in the body of the Supreme Lord Krishna.",
                "In this Krishna emphasizes the superiority of Bhakti Yoga (the path of devotion) over all other types of spiritual disciplines and reveals various aspects of devotion.",
                "He further explains that the devotees who perform pure devotional service to Him with their consciousness merged in Him and all their actions dedicated to Him are quickly liberated from the cycle of life and death.",
                "He also describes the various qualities of the devotees who are very dear to Him.",
                "The word kshetra means the field and the kshetrajna means the knower of the field.",
                "We can think of our material body as the field and our immortal soul as the knower of the field.",
                "In this Krishna discriminates between the physical body and the immortal soul.",
                "He explains that the physical body is temporary and perishable whereas the soul is permanent and eternal.",
                "The physical body can be destroyed but the soul can never be destroyed.",
                "They then describe God as the Supreme Soul.",
                "All the individual souls have originated from the Supreme Soul.",
                "One who clearly understands the difference between the body the Soul and the Supreme Soul attains the realization of Brahman.",
                "In this Krishna reveals the three gunas (modes) of the material nature goodness passion and ignorance which everything in the material existence is influenced by.",
                "He further explains the essential characteristics of each of these modes their cause and how they influence a living entity affected by them.",
                "He then reveals the various characteristics of the persons who have gone beyond these gunas.",
                "The ends with Krishna reminding us of the power of pure devotion to God and how attachment to God can help us transcend these gunas.",
                "In Sanskrit Purusha means the All pervading God  and Purushottam means the timeless & transcendental aspect of God.",
                "Krishna reveals that the purpose of this Transcendental knowledge of the God is to detach ourselves from the bondage of the material world and to understand Krishna as the Supreme Divine Personality who is the eternal controller and sustainer of the world.",
                "One who understands this Ultimate Truth surrenders to Him and engages in His devotional service.",
                "In this Krishna describes explicitly the two kinds of natures among human beings divine and demoniac.",
                "Those who possess demonaic qualities associate themselves with the modes of passion and ignorance do not follow the regulations of the scriptures and embrace materialistic views.",
                "These people attain lower births and further material bondage.",
                "But people who possess divine qualities follow the instructions of the scriptures associate themselves with the mode of goodness and purify the mind through spiritual practices.",
                "This leads to the enhancement of divine qualities and they eventually attain spiritual realization.",
                "In this Krishna describes the three types of faith corresponding to the three modes of the material nature.",
                "Lord Krishna further reveals that it is the nature of faith that determines the quality of life and the character of living entities.",
                "Those who have faith in passion and ignorance perform actions that yield temporary material results while those who have faith in goodness perform actions in accordance with scriptural instructions and hence their hearts get further purified.",
                "Arjuna requests the Lord to explain the difference between the two types of renunciations sanyaas(renunciation of actions) and tyaag(renunciation of desires).",
                "Krishna explains that a sanyaasi is one who abandons family and society in order to practise spiritual discipline whereas a tyaagi is one who performs their duties without attachment to the rewards of their actions and dedicating them to the God.",
                "Krishna recommends the second kind of renunciation tyaag.",
                "Krishna then gives a detailed analysis of the effects of the three modes of material nature.",
                "He declares that the highest path of spirituality is pure unconditional loving service unto the Supreme Divine Personality Krishna.",
                "If we always remember Him keep chanting His name and dedicate all our actions unto Him take refuge in Him and make Him our Supreme goal then by His grace we will surely overcome all obstacles and difficulties and be freed from this cycle of birth and death.",
                "One thing I ask from the LORD, this only do I seek that I may dwell in the house of the LORD all the days of my life, to gaze on the beauty of the LORD and to seek him in his temple.",
                "Taste and see that the LORD is good; blessed is the one who takes refuge in him.",
                "Do you not know? Have you not heard? The LORD is the everlasting God, the Creator of the ends of the earth. He will not grow tired or weary, and his understanding no one can fathom. He gives strength to the weary and increases the power of the weak.",
                "For I am convinced that neither death nor life, neither angels nor demons, neither the present nor the future, nor any powers, neither height nor depth, nor anything else in all creation, will be able to separate us from the love of God that is in Christ Jesus our Lord.",
                "For now we see only a reflection as in a mirror; then we shall see face to face. Now I know in part; then I shall know fully, even as I am fully known.",
                "Because of the LORD’s great love we are not consumed, for his compassions never fail. They are new every morning; great is your faithfulness.",
                "So that Christ may dwell in your hearts through faith. And I pray that you, being rooted and established in love, may have power, together with all the Lord’s holy people, to grasp how wide and long and high anddeep is the love of Christ, and to know this love that surpasses knowledge—that you may be filled to the measure of all the fullness of God ",
                "Keep his commands and do what pleases him.",
                "Consider it pure joy, my brothers and sisters, whenever you face trials of many kinds, because you know that the testing of your faith produces perseverance. Let perseverance finish its work so that you may be mature and complete, not lacking anything.",
                "Therefore, my dear brothers and sisters, stand firm. Let nothing move you. Always give yourselves fully to the work of the Lord, because you know that your labor in the Lord is not in vain.",
                "The philosophically inclined should for that reason endeavor only for this spiritual fulfillment that is not so much found by searching from high to low for material fulfillment countered by miseries is in the course of the time that operates so subtly found anyhow as a result of one s actions ",
                "Being only five years old I attended the school of the brahmins and lived depending on her without having a clue about time place and direction ",
                "When she once went out at night to milk a cow she was bitten in the leg by a snake on the path and thus my poor mother fell victim of the supreme time ",
                "Krishna together with the munis pacified the shocked and affected family who had lost their friends and members by showing how each is subjected to the Time that cannot be avoided ",
                "I consider You the personification of Eternal Time the Lord without a beginning or an end the All pervasive One who distributes His mercy everywhere equally among the beings who live in mutual dissent ",
                "All the unpleasant that transpired I think is the inescapable effect of Time you just like the rest of the world with its ruling demigods fall under that control the way clouds are carried by the wind ",
                "Why else would there be such misfortune with Yudhishthhira the son of the ruler of religion being present as also Bhîma with his mighty club Arjuna carrying his Gândîva and our well wisher Krishna ",
                "They said We have always bowed down to Your lotus feet oh Lord like one does in the worship of Brahmâ and his sons and the king of heaven You after all are for the ones who desire the supreme welfare in this life the Master of Transcendence upon whom the inevitable time has no grip ",
                "But the insurmountable and imperceptible Time surpasses inimitably those who are inattentive and engrossed in the mind of attachment to family affairs ",
                "Vidura well aware of this said to Dhritarâshthra Oh King dear brother please withdraw yourself without delay just see how fear is ruling your life ",
                "In this material world oh master there is no help from anyone or anything to escape this fear because that fear concerns the Supreme Lord who approaches us all in the form of eternal Time ",
                "Inevitably overtaken by the pull of time a person must just like that give up this life as dear as it is to everyone not to mention the wealth and such he has acquired ",
                "How is this body which is made out of the five elements fire water air earth and ether and is controlled by time by materially motivated action and by the modes of nature kâla karma and the gunas capable of protecting others when it is just as well bitten by that snake ",
                "He the Father of all creation the Supreme Lord has now oh great King descended in this world in the form of death the all devouring Time in order to eliminate everyone inimical to the enlightened souls ",
                "The time had taken an inauspicious turn he observed seasonal irregularities and saw that human beings sinfully turned to anger greed and falsehood in heartening their civil means of livelihood ",
                "The people gradually were acquiring godless habits like wantonness and such The king facing these serious matters and bad omens spoke with his younger brother about it ",
                "Might it be so that as Nârada told us the Supreme Personality has decided it is time to leave this physical manifestation of Himself ",
                "Bearing in mind the words spoken by Govinda I remember how attractive they are and how they imbued with importance and appropriate to time and circumstance put an end to the pain in the heart ",
                "Please inform me oh reservoir of all riches about the reason of your sadness that reduced you to such a weak state Or has oh mother powerful Time that even subdues the most powerful soul stolen away your good fortune extolled by the demigods ",
                " Verily seven days from now the wretched soul of the dynasty who offended my father will because of breaking with the etiquette be bitten by a snake bird ",
                "Thus pondering the message reached him of the curse of death pronounced by the sage s son That curse in the form of the fire of a snake bird he accepted as auspicious because that impending happening would be the consequence of his indifference about worldly affairs ",
                "Oh member of the Kuru family therefore also your life s duration that is limited to seven days should inspire you to perform everything that traditionally belongs to the rituals for a next life ",
                "His personal body is this gross material world in which we experience all that belongs to the past the present and the future of this universe in existence ",
                "His veins are the rivers and the plants and trees are the hairs on the body of the Universal Form oh King The air is His omnipotent breathing the passing of the ages Time is His movement and the constant operation of the modes of material nature is His activity ",
                "Let me tell you that the hairs on the head of the Supreme One are the clouds oh best of the Kurus and that the intelligence of the Almighty One is the prime cause of the material creation so one says His mind the reservoir of all changes is known as the moon ",
                "Whenever one desires to give up one s body oh King one should as a sage without being disturbed comfortably seated and with one s thinking unperturbed by matters of time and place in control of the life air restrain the senses with the help of the mind ",
                "Therein one will not find the supremacy of time that for sure controls the godly who direct the worldly creatures with their demigods nor will one find there mundane goodness passion or ignorance or any material change or causality of nature at large ",
                "Knowing what and what not relates to the divine of the transcendental position they who wish to avoid what is godless completely give up the perplexities of arguing to time and place and place thereto in purely at Him directed good will every moment His worshipable lotus feet in their heart ",
                "In the control of the divinity of fire Vais vânara or with regular sacrifice and meditation one attains by following the path of the sushumnâ the channel of balancing the breath the illuminating pure Spirit of the Absolute whereupon being freed from impurities going upwards one in respect of the cyclic order of the luminaries reaches the galactic cakra order of the Lord oh King called S is umâra meaning dolphin to the form of the Milky Way galactic time ",
                "Passing beyond that navel of the universe the pivot the center of spin of the Maintainer Vishnu only the individual living being who got purified by the realization of his smallness the yogi reaches the place worshiped by those who know the Absolute Spirit The self realized souls enjoy their stay there for the time of a kalpa a day of Brahmâ ",
                "All of the world that I created was created from the effulgence the brahmajyoti of His existence just like it is with the fire the sun the moon the planets and the stars that radiate from His effulgence ",
                "The Lord of Control by the potency of His material energy thus from the independent will of His divine self arrived at many appearances taking upon Himself their karma being subjected to time and their particular natures ",
                "Because of the superintendence of the Original Person the creation of the mahat tattva the greater reality took place from eternal time there was the transformation of the modes and from the modification of the original nature the different activities found their existence ",
                "By transformation of the ether the air found its existence which is characterized by the quality of touch Along with it sound also appeared as a characteristic that was remembered from the ether Air thus acquired also a life of diversity with energy and force Air on its turn again transformed under the influence of time and generated from its nature the element of fire in response to what preceded With its form there was likewise touch and sound as the hereditary burden or the karma of the previous elements Fire transformed or condensed from oxygen and hydrogen into water Thus the element of taste came about which consequently was accompanied by touch sound and form But because of the variegatedness of that transformation of water next the smell of the juice followed that assumed form as the earth element together with the qualities of touch and sound ",
                "universe after countless millennia having been submerged in the causal waters was by the personal soul the Lord who animates the inanimate awakened to its own time of living ",
                "The unseen mover Time of the seas and oceans of the living beings that evolve but also find physical destruction in His belly during Brahmâ s night is by the intelligent ones known as the beating heart that is located in the subtle body ",
                "What is needed for the performance of sacrifices are matters such as flowers and leaves burning material such as straw an altar and also a framework of time a calendar e g for following the qualities of nature like spring ",
                "The first avatâra of the Lord is the Original Person Mahâvishnu or Kâranodakas âyî Vishnu He is the foundation of space time kâla svabhavah the original nature of time cause effect the elements the modes as also the ego the senses and the mind These together constitute the diversity of the gigantic universal independent body also called Garbhodakas âyî Vishnu of all that moves and does not move of the Almighty Supreme ",
                "what about a day of Brahmâ a kalpa and the periods between them vikalpas What can you say about the time we refer to with the words past future and present And how about the lifespan allotted to embodied beings ",
                "Oh purest of the twice born souls when did time begin and what can you say about the indication of time as being short or long as like passing with a certain activity ",
                " The very moment he the witness the soul in his glory of transcending the time of the material energy enjoys it to be free from illusion he in that fulness will forsake these two of I and mine ",
                "Uddhava said What can I say about our wellbeing now the sun of Krishna has set and the house of my family has been swallowed by the great serpent of the past ",
                "Even though You have no desires You engage in all kinds of activities even though You are unborn You still take birth being the controller of eternal Time You nevertheless take shelter of the fortress out of fear for Your enemies and despite enjoying within Yourself You lead a household life in the association of women this bewilders the intelligence of the scholars in this world ",
                "You are never divided and ever fresh yet You in Your eternal intelligence oh Master call upon me for consultation as if You would be bewildered But that is never the case That boggles my mind oh Lord ",
                "I pity all those pitiable poor souls who out of touch with the divinity of Time in their ignorant sinfulness have turned themselves away from the stories about the Lord and waste their lives with useless philosophical exercises imaginary purposes and a diversity of rituals ",
                "With the effect of Eternal Time kâla upon the three modes of this illusory energy the Supreme being in the beyond generated the virility the valor the manliness the power by means of the person or the Purusha as a plenary expansion of Himself ",
                "From the unmanifested then by the interaction of time came about the Mahat tattva the complete of the Supreme the cosmic intelligence This self of discernment seated in the physical self dispells the ignornance and makes the universe clearly visible ",
                "That cosmic intelligence thus being a part of or subjected to guna kāla and jîv âtmâ the material qualities time and the individual self transformed itself within the range of sight of the Personality of Godhead into the individuality of all the different life forms of this universe with their desire to pro create to continue their identification and karma ",
                "Material energy is a partial local mixture of time the time of expanding and contracting The Supreme Lord glancing this over from the ether thus being contacted created the transformation of that touch in the form of air gasses ",
                "The air also transformed by the extremely powerful ether gave in contraction rise to the form of the light of the fire of the sun and the stars and the bio electricity of sense perception by which the world is perceived ",
                "With the interaction of air and light fire there was with the glance of the Lord of the ether mixing time with the external energy a transformation that created water in combination with its taste ",
                "With the partly local uniting of the material energy with eternal time the by the light produced water that was thus created as a consequence of the transformation of the Supreme Spirit of God glancing over the earth led to the creation of the quality of smell ",
                "Oh Unborn One direct us in making our offerings at the right time Thus we can share our meals and can also all other living beings have their sustenance so that we with our offerings of item undisturbed may enjoy our meal ",
                "Eyes appeared in the gigantic body that offered a position to Tvashthâ the director of light and the power of sight by which forms can be seen ",
                "Next the heart of the Universal Being manifested in which Candra the god of the moon took his position with the function of mental activity because of which one is lost in thoughts ",
                "Oh brahmin how are the periodical offerings of S râddha regulated to honor the deceased and to respect what the forefathers have created and how are the times settled in respect of the positions of the luminaries like the planets and the stars ",
                "The way the power of fire is hidden in wood He resided there in His place in the water keeping all living beings in their subtlety within His transcendental body from where He gives life in the form of Time kâla ",
                "For the duration of thousand times four yugas billion years He with His internal potency lay dormant for the sake of the further development by means of His force called kâla time of the worlds of the living beings who depend on fruitive activities That role gave His body a bluish look the blue of the refuge of the vivifying water ",
                "With the Time that roused the karma to activity soon from the original self of Vishnu with that agitation a lotus bud appeared that just like a sun illumined the vast waters with its effulgence ",
                "Groping in the dark oh Vidura with his contemplating this way it thus came to pass that the enormity of the three dimensional reality of time tri kâlika was generated that as a weapon a cakra inspires fear in the embodied unborn soul by limiting his span of life to a hundred years compare ",
                "As long as the people of the world are engaged in unwanted activities and in the activities of their self interest despise the by You as beneficial pronounced devotional activities the struggle for existence of these people will be very tough and with the defiance of Your Vigilant Rule of Time lead straight to a shambles Let there be my obeisances unto You ",
                "He who was born on the lotus then saw how the lotus upon which he was situated and the water surrounding it trembled because of the wind that was propelled by the power of eternal Time ",
                "It is by means of time kâla the hidden impersonal feature that the Lord separated from the Supreme Absolute God or brahma the material phenomenon that was established as the bewildering material potency of Vishnu ",
                "The way it Eternal Time is there in the present it was there in the beginning and will also be there hereafter ",
                "The supreme oneness of that particle being present within material bodies keeps its original form till the end of time it is of a continual unrivaled uniformity ",
                " Time my best one besides being known as the supreme non manifest Almighty Lord who controls all physical action can therefore also be measured by the motion of the minutest and largest forms of combinations of particles ",
                " The time of that infinitesimal particle is the time it takes to occupy or vibrate in a certain atomic space The greatest of time is the time taken by the existence of the complete of all atoms ",
                "The aggregate of such a day and night is called an ancestral traditional or solar month with two of them forming a season There are six of them respectively cold or hemanta dew or s is ira spring or vasanta warm or grîshma rainy or varshâs and autumn or s arad counting from December corresponding to the movement of the sun going through the southern and northern sky ",
                " This movement of the sun is said to form one day of the demigods and is called a vatsara a tropical year of twelve months The duration of life of the human being is estimated to be of a great number a hundred of those years see also the full calendar of order ",
                "The infinitesimal particles and their combinations the planets the heavenly bodies like the moon and the stars all rotate in the universe to complete their orbit in a year of the Almighty cyclic order the command of eternal time ",
                "We speak about an orbit of the sun about an orbit of the other planets the orbit of the stars in our galaxy around Sagittarius A in the sky the orbit of the moon oh Vidura and the orbit of the earth as being a single but differently named year respectively a celestial year a planetary year a galactic year a lunation and a tropical year ",
                "With attention for all His five different types of years one should be of respect for the One Lord of Time who differing from all that was created moves under the name of Eternal Time and who with His energy in different ways invigorates the seeds of creation while during the day dissipating the darkness of the living entities By thus performing sacrifices one develops quality in one s material existence ",
                " The time measured by the two halves of Brahmâ s life takes but a second for the beginningless unchanging and unlimited Soul of the universe ",
                "This eternal time beginning from the atom up to the final duration of two parârdhas is never capable of controlling the Supreme Lord it is the controller of those souls who are identified with their body ",
                "The demigods said You oh mighty one must be knowing about this darkness we are so very afraid of Your supreme divinity is not affected by time and thus nothing is hidden for you ",
                "Oh original father of all the conditioned souls in the grip of desire are all bound by the rope of the words of You the Lord of the living beings I following their example also offer my oblations to You oh light of eternal time ",
                "The wheel of the universe that with a tremendous speed spins around the axle of Your imperishable nature Brahman with the three naves of the sun the moon and the stars the twelve to thirteen spokes of the lunar months the three hundred and sixty joints of the days in a demigod year the six rims of the seasons and the innumerable leaves that are the moments may be cutting short the life span of the universe but not the lives of the devotees ",
                "I surrender myself to You Lord Kapila who are the supreme transcendental personality the origin of the world the full awareness of time and the three basic qualities of nature the Maintainer of All the Worlds and the sovereign power who by His own potency absorbs the manifestations after their dissolution ",
                "Oh mother My devotees will never not by time nor by a weapon of destruction lose Me and My opulence who was chosen by them as their dearest self son friend preceptor benefactor and deity ",
                "Thus with the classification I provided the material qualities of the Absolute Truth of Brahman are summed up called saguna brahman One speaks thereto of time as the twenty fifth element ",
                "Some say that the time factor constitutes the power of the Original Person that is feared by the doer the individual soul who is deluded by the false ego of being in touch with material nature ",
                "The expanding accelerating movement of material nature without the interaction of her modes and their specific qualities oh daughter of Manu is the space time the fourth dimension from which we in our world know Him the Supreme Lord ",
                "He the Lord of All Entities abides as a consequence of His potencies within in the form of the original person purusha and without in the form of time the twenty fifth element ",
                "The characteristic traits of one s reason in this state of Krishna or natural time consciousness thus are similar to those of the natural state of pure water clarity invariability and serenity ",
                "From the ether evolved from the subtlety of sound the subtle element of touch under the transforming impulse of time Thus the air is found as also the sense organ for it and the active perception with that sense of touch ",
                "When in the beginning of creation the seven primary elements the five material elements ego and cosmic intelligence the mahat tattva were not yet mixed the Lord the origin of creation endowed with kâla karma and guna time workload and the modes entered the universe ",
                "one s thinking is purified and controlled by the practice of yoga one should while looking at the tip of one s nose meditate on the goal of the Supreme Lord kâshthhâ His form and measure of time e g represented by a mechanical clock or water clock fixed on the sun s summit with the division of time according to the Bhâgavatam ",
                "And what about the nature of Eternal Time the drive of nature the form of the Supreme Lord of You as the ruler over all the other rulers under the influence of whom the common people act piously ",
                " Natural time known as the divine cause of the different manifestations of the living entities constitutes the reason why all living beings who consider themselves as existing in separation from the greatest on live in fear ",
                "He who from within enters all the living entities constitutes the support of everyone and annihilates them again by means of other living beings is named Vishnu the enjoyer of all sacrifices who is that time factor the master of all masters ",
                "He for whom the wind out of fear blows and this sun is shining for whom out of fear Indra sends his rains and the heavenly bodies are shining He because of whom out of fear the trees creepers and herbs each in their own time bear flowers and produce their fruits He afraid of whom the rivers flow and the oceans do not overflow because of whom fire burns and the earth with her mountains does not submerge He because of whom the sky provides air to those who breathe and under the control of whom the universe expands its body of the complete reality mahat tattva with its seven layers He for whom out of fear the gods of creation and more in charge of the basic qualities of nature within this world carry out their functions according to the yugas see He of whom afraid all the animate and inanimate beings find their control that infinite final operator of beginningless Time is the unchangeable Creator who creates people out of people and by means of death puts an end to the rule of death ",
                "Kapila said Just like a mass of clouds has no knowledge of the powerful wind a person has no knowledge of this time factor even though he is being conditioned by it ",
                "What else but Your divinity that as a partial aspect the Paramâtmâ dwells in both the animate and the inanimate would give us the knowledge of the threefold of Time of past present and future In order to be freed from the threefold misery as caused by oneself nature and others we as individual souls engaged on the path of fruitive activities have to surrender to that divinity ",
                " But remember that even Brahmâ the Creator of the mobile and immobile manifestations who is the source of Vedic wisdom as also the sages and the masters of yoga the Kumâras and the other perfected souls and original thinkers of yoga who attained the original Person of the Absolute Truth the first among all the souls by dint of their detached egoless actions despite their independent vision and all their spiritual qualities again take birth to assume their positions when this manifestation of the Lord is recreated through the operation of time and the three modes And that is also true for all others who enjoyed the divine opulence that resulted from their pious deeds they also return when the interaction of the modes again takes place ",
                "I explained to you the four divisions of identity svarûpa in devotional service three according to the modes and one for their transcendence as also the imperceptible action of time the conditioning that drives the living entities ",
                "Man and woman evolved by the impelling force of Time from the five elements of matter and by their sexual behavior even more men and women came about in this world ",
                "Under the influence of the no doubt hard to fathom potency of the Almighty One in the form of the force of Time the interaction or disturbance of the equilibrium of the modes of nature resulted in this diversity of energies The Supreme Personality exerts His influence upon this diversity even though He is not the one acting and in this diversity He leads to death even though He is not the one who kills ",
                " He to whom there is no end puts in the form of Time everything to an end He who knows no beginning constitutes the beginning of everything He who is inexhaustible gives life to one living being by means of another one and He as death puts an end to everything that kills ",
                "As death entering everyone s life no one is His ally or enemy All the combinations of the elements organic and inorganic helplessly follow His movement like dust particles moved by the wind ",
                "Some oh king explain this karma the work load of fruitive activities as arising from one s particular nature or as brought about by others oh protector of men Some say it is due to time others refer to fate while still others ascribe it to the desire of the living entity ",
                "One s intelligence is of ignorance with the misconceptions of I and you To a person following the bodily concept life appears to be just like in a dream it the physical approach constitutes the cause of bondage and misfortune ",
                "He considered everything created comprising his body his wives children friends his influence riches the pleasure grounds the facilities for his women and the complete of the beauty of the earth with its oceans as something bound to time and for that reason he left for Badarikâs rama the Himalayan forest ",
                "Radiating by its effulgence that place illumines from within all the three worlds everywhere and also makes them radiate It can only be reached by those who constantly engage in welfare activities and not by those who are not merciful with other living beings ",
                "The invincible time by which You in Your prowess and majesty with simply raising Your eyebrows vanquish the entire universe constitutes no threat to a soul of complete surrender ",
                "One may guess about the authority and order of Your reality of Time All we see is how You just like the wind scattering the clouds with Your so very great force of Time in the long run destroy all the planetary systems and how all living beings equally find their end because of external causes ",
                " Intimately making fun she embraced him as he held her in his arms Thus being captivated by the woman he lost his keenness and was not quite aware of how day and night the insurmountable time was passing ",
                "Thus wantonly involved with a heart enslaved by kith and kin one day the time of old age arrived that is not very loved by those who are fond of women Oh King there is a king belonging to the heavenly kingdom Gandharvaloka who is called Candavega the impetuously streaming time He is at the head of three hundred and sixty very powerful other Gandharvas the days in a year ",
                " All of this happened during the time that the daughter of the Almighty Time called Kâlakanyâ traveled the three worlds desiring someone for a husband oh King Prâcînabarhi but there was never anyone who accepted her proposal ",
                "When the king of the Yavanas heard the daughter of Time express herself in these words he wishing to serve the Lord was willing to do his duty in the private sphere and addressed her with a smile ",
                "The year called Candavega stands for the passage of time to which the three hundred and sixty men and women from heaven are to be understood as being the days and nights that by their moving around reduce the lifespan that one has on this earth see The daughter of Time who was welcomed by no one and as the sister in law was accepted by the king of the Yavanas in favor of death and destruction stands for jarâ old age see ",
                "The flowers work just like a woman who with her sweet scent of flowers suggests the safety of a household existence as being the result of an innocent desire for sensual pleasures such as plucking flowers Thus one fulfills one s desires like the deer in always being absorbed in thoughts of sex with the wife and pleasures to the tongue The sound of the different bumblebees that is so very attractive to the ears compares to the most attractive talks of the wife in the first place and also to those of the children that occupy one s mind completely The tigers in front of him are together alike all the moments of the days and nights that in enjoying one s household unnoticed take away one s life span And from behind there is the hunter taking care not to be seen while crouching upon him like the superintendent of death by whose arrow one s heart is pierced in this world You should see yourself in this as the one whose heart is pierced oh King ",
                "With the Fortunate One constantly at one s side abiding by a spirit of pure goodness free from passion and ignorance the world around oneself the so called here and now that with all those impressions can be like with the dark appearance of the new moon also called Rahu with an eclipse thus being connected will manifest itself crystal clear ",
                "According to Vijayadhvaja Tîrtha who belongs to the Madhvâcârya sampradâya the two following verses appear after verse Being of devotion unto Krishna of mercy towards others and of perfect knowledge of the True Self liberation from being bound to a material life will be the consequence The great secret of it all is that the material existence of what we will see and do not see anymore all dissolves just like when one sleeps in other words everything that happened in the past happens in the present and is going to happen in the future is but a dream ",
                "You who in Your compassion by Your expansions and teachers are visible to the humble devotees are by one s devotional service with the necessary respect of time always remembered as such by Your beautiful embodiment and not so much by thousands of mantras oh destroyer of all inauspiciousness ",
                "All of you united in His quality be therefore engaged in the devotional service of directly the Supreme Lord who is the actual cause pradhâna of Time who is the original Person and the One Supreme Soul of the unlimited number of individual souls He who by His spiritual power is aloof from all emanations of the self ",
                " Being of worship unto Vishnu He also in respect of the different gods and purposes and in line with the instructions providing in abundance for everything that was needed performed according to time and circumstance a hundred times over all kinds of ceremonial sacrifices with priests of the proper age and faith ",
                " Alas he thought to himself by the Controller turning the wheel of time this creature was deprived of its family friends and relatives Finding me for its shelter it has only me as its father mother brother and member of the herd Surely having no one else it puts great faith in me as the support to rely upon and thus fully depends on me for its learning sustenance love and protection I have got to admit that it is wrong to neglect someone who has taken shelter and must accordingly act without regrets ",
                "Supposing that his son despite not feeling for it had to be fully instructed by him in all the cleanliness Vedic literature vows principles sacrifice and service to the guru that belongs to the celibate state the brahmacarya âs rama the brahmin who considered his son his life breath in reality acted out of household attachment Therefore he died when he was seized by death not as forgetful as he was as a man full of frustration about the unfit obstinacy of his son ",
                "Oh Vishnudatta protected by Vishnu Parîkchit to those who are not perplexed this is not such a great miracle They who without animosity are of goodness to all are by the Supreme Lord of the invincible Time who carries the best of all weapons the Sudars ana disc personally fully liberated from the very strong and tight knot in the heart that is the consequence of a false physical concept of life Even when threatened by decapitation or by other attacks on their lives those liberated souls and devotees who full of surrender are protected at His lotus feet are never upset by these kinds of emotional conditions they have nothing to fear ",
                "Thus he sarcastically criticized him severely But there was no protest of a false belief of I and mine from him who in silence kept carrying the palanquin As someone on the spiritual platform he happened to be of such a particular disposition concerning the physical matters of having a from ignorance resulting finite vehicle of time a physical body that consists of a mixture of the natural modes the workload and material intentions ",
                "The knower of the field is originally the all pervading omnipresent authentic person the Oldest One who is seen and heard of as existing by His own light He is never born He is the transcendental Nârâyana the Supreme Lord Vâsudeva He is the one who just like the air present within the body by His own potency exists in the soul as the controller of the moving and unmoving living entities He is the Supersoul of expansion who has entered and initiated the creation and thus is of control as the Fortunate One in the beyond He is the shelter and knower of everyone in every field He is the vital force the Mover of Time that appeared in this material world see also B G ",
                "Please understand that being meager fat tiny or big existing as an individual entity inanimate matter or whatever other natural phenomenon of disposition all concerns impermanence in the name of a certain place time and activity a temporary state inherent to the operation of nature s duality ",
                "Sometimes in the dark of night driven by a momentary whirlwind of passion he copulates like mad in a total neglect of the rules Blinded by the strength of that passion he notwithstanding the divinities of the sun and the moon of regularity and order then loses all notion being overcome by a mind full of lust ",
                "Thus it may happen that because of the cakra of the Controller the Supreme Lord Vishnu s disc of Time the influence of which stretches from the first expansion of atoms to the duration of the complete life of Brahmâ one has to suffer the symptoms of its rotating With that rotation in the course of time swiftly before one s eyes in terms of eternity in a moment all lives of the living entities are spent from Brahmâ to the simplest blade of grass Directly of Him the Controller whose personal weapon is the disc of Time one is afraid at heart As a consequence not caring about the Supreme Lord the Original Person of Sacrifice one then accepts as worshipable what lacks foundation with self invented gods who operating like buzzards vultures herons and crows are denied by the scriptures of one s civilization ",
                " The Supreme Lord resides in Ketumâla in the form of Kâmadeva or also Pradyumna see according to His wish to satisfy the Goddess of Fortune as also the sons the days and the daughters the nights of the founding father Samvatsara the deity of the year who rule the land and of whom there are as many as there are days and nights in a human lifetime The fetuses of these daughters whose minds are upset by the radiation of the mighty weapon the cakra of the Supreme Personality are ruined and after one year expelled dead from the womb as miscarriages ",
                "From You with Your countless different names forms and features the scholars derive their notion of numerical proportions enumerations and compositions the truth of which they verify by observation Unto Him who thus discloses Himself in analysis You I offer my obeisances see also Kapila ",
                "Having forsaken one s identification with the body one must at the end of one s time of living with a devotional attitude concentrate one s mind on You who are transcendental to the material qualities This forsaking constitutes the perfection of the practice of yoga as explained by the almighty Lord Brahmâ A person driven by desire thinks in fear about the present and future of his children wife and wealth but anyone who knows about the hopelessness of this deficient vehicle of time considers such endeavors only a waste of time because the body is lost in the end ",
                " With his effulgence he divides the time in the light and dark period of the month s ukla and krishna May he that divinity of the moon and the grain to be distributed to the forefathers and the demigods may that king of all people remain favorably disposed unto us ",
                "That dvîpa has one mountain range named Mânasottara that separates the varshas on the inner and the outer side Measuring a yojanas high and wide it harbors in its four directions the cities of the four demigods ruling there Indra Yama Varuna and Soma The chariot of the sun god Sûrya circumambulating mount Meru on its highest point moves around in an orbit that calculated in terms of the days and nights of the demigods consists of one complete year ",
                "Thus encircling with an orbit before the Mânasottara mountains thereabout of ninety five million one hundred thousand yojanas long so the scholars teach us one on the east of Meru finds Devadhânî the city of King Indra south of it the city named Samyamanî of Yamarâja in the west the city named Nimlocanî of Varuna and in the north the city of the moon named Vibhâvarî At all the four sides of Meru as the energetic pivot thus creating sunrise sunset noontime and midnight it brings about the particular times of the living beings to be active or to cease their activity ",
                "This vehicle has only one wheel with twelve spokes the months six segments the seasons and three pieces to its hub four month periods that in its entirety is known as a solar year a samvatsara Its axle is fixed on the top of Meru with Mânasottara at the other end The wheel of the chariot of the sun being fixed there rotates to the mountain range of Mânasottara like a wheel of an oil press machine ",
                "To that he S uka clearly stated Just as what one sees with the movements of small ants spinning around on a potter s wheel that because of their changing positions experience a different orientation such a difference can also be observed with the movement of the sun and the planets in relation to Meru and Dhruvaloka the central heap of stars and the galaxy center With the stars moving around that center the two are located at their right side but because of the individual movements of the planets led by the sun upon that rotating wheel of time the sun and planets that are observed in different mansions and constellations are evidently of another progress ",
                "He that solar lead of time this supremely powerful Original Person who is Nârâyana Himself the Supersoul of the three Vedic principles who is there for the benefit and karmic purification of all the worlds is the cause sought by all saintly and Vedic knowing He divides the year as He thinks fit in its twelve parts and arranges the six seasons beginning with spring with their different qualities ",
                "He now the Soul of all the worlds who in the form of the sun has entered the wheel of time in a position between heaven and earth passes through the twelve divisions of the year consisting of the months that are named after the signs of the zodiac The scholars teach that they according to the moon are divided in bright and dark halves or fifteen day fortnights and that following their instruction the six portions of its orbit called ritu or season calculated to the stars each cover two and a quarter constellations thus one speaks of twelve or more constellations They also say that the period of time the sun moves through the visible half of outer space is called an ayana ",
                " More than two hundred thousand yojanas behind the moon there are spinning with Meru to the right to the many stars that by the Supreme Controller were attached to the wheel of time the twenty eight lunar mansions including Abhijit ",
                "S rî S uka then said Million yojanas above them the stars of the sages one finds that supreme abode of Lord Vishnu where the great devotee Dhruva the son of Uttânapâda resides whose glory of obedient devotion I described already see It is the source of life of all living entities from now until the end of the kalpa about which Agni the fire god Indra the king of heaven the founding father who is the Prajâpati Kas yapa as also Dharmarâja in unison full of respect move clockwise For all the restless luminaries the planets the stars and the rest that place constitutes the incandescent radiating pivot that is established by the Lord The inconceivable all powerful force of Time is considered the cause of their revolving The luminaries keep their positions just like three bulls that for threshing rice are yoked to a central pole Moving in their orbits they have a fixed position relative to the inner and outer rims of the wheel of time the same way the planets keep their positions around the sun Holding on to Dhruvaloka till the end of creation they revolve in the sky as if they are driven by the wind just like heavy clouds and big birds do that controlled by the air move their bodies around according to their respective positions Thus the luminaries behave consequently by the combined effort of material nature and the Original Person the way they always have and never collide with the earth ",
                "Some imagine this great army of luminaries to be a s is umâra a dolphin and describe it concentrated in yoga as that what can be seen of the Supreme Lord Vâsudeva see also a picture of the celestial sky as factually seen in a telescope ",
                " This form of S is umâra certainly is the form of the Supreme Lord of Lord Vishnu who consists of all the demigods With that form before one s eyes one should each morning noon and evening in all modesty meditate on the following words Our obeisances unto this resting place of all the luminous worlds unto the master of the demigods the Supreme Personality in the form of Time upon whom we meditate namo jyotih lokâya kâlâyanâya animishâm pataye mahâ purushâya abhidhîmahîti see also Those who in respect of that leader of the demigods consisting of all the planets and stars that destroyer of sin practice the mantra as mentioned above by three times a day offering their respects this way or by three times a day meditating as such in silence will by that respect for our sweet Lord in the form of time very soon find all their sins annihilated ",
                "The Supreme Lord who is there for the protection of both these divinities operates by the supreme presence of the wheel of Time the Sudars ana Cakra This disc is deemed the most dear most devoted and favorite weapon that by its power and unbearable heat makes Râhu flee with a mind full of fear and a bewildered heart far away from that position wherein he resides for almost an hour and that by the people is called an eclipse ",
                "There one assuredly is of no concern about divisions of time relative to the changes of day and night as observed with sundials and lunar phases ",
                "No other cause of death than the almighty wheel of Time in the form of His disc weapon is capable of influencing them in any way ",
                "It is practically always out of fear for the Lord s cakra order the compelling natural order of time that the wives of the godless souls lose their fetuses in miscarriages ",
                "Below Sutala in the world of Talâtala the dânava demon king rules named Maya His cities were burned by the almighty Tripurâri S iva the lord of the three cities who desired the welfare of the three worlds But he Maya the master and teacher of all sorcery regained his kingdom by his grace Being protected by Mahâdeva the great god who is S iva he thinks he has nothing to fear from the Sudars ana Cakra the presence of the Lord in the form of Time that in all worlds is worshiped with clocks and calendars ",
                "Someone who takes away the money the wife or children of someone else is sure to be bound with the fetters of time by the most frightening men of Yamarâja and by force to be thrown into the hell of Tâmisra the darkness Having landed in that darkest of all conditions being deprived of item and water beaten with sticks and scolded he sometimes in his desperation loses his consciousness because of the severe punishments received ",
                "The sun the fire the sky the air the gods the moon the evening the day and the night the directions the water and the land are all evidence of the personal dharma the very nature of the embodied living entity see also B G ",
                "Just as the present time carries the characteristics of what was and what will become someone s present birth likewise is indicative of the dharma and adharma of what one did and will be doing ",
                "They the devotees who with an equal vision are of surrender to the Supreme Lord and whose sacred histories are proclaimed by the demigods and perfected souls you should never approach for they are fully protected by the mace of the Lord It is not to us to punish them just as it is not given to time itself to tell right from wrong ",
                "The earth ly affair the body was the field of action the eternal cause engrossing the individual soul that constitutes the basis of his bondage What would the use of time bound labor be when one fails to see the finality of it all ",
                "The so very sharp revolving wheel of Time governs all the world according to its own rule and measure of what use is it to endeavor in desire for results in this world when one does not know about this this order of time ",
                "How can one entangled in the modes of nature see B G undertake anything like begetting children if one does not understand the instructions of the scriptures of the Father that tell one how to put an end to the material way of life ",
                " May Kes ava protect me with His club during the hours after sunrise may Govinda holding His flute protect me early in the morning may Nârâyana the Lord of all potencies protect me late in the morning and may Lord Vishnu the ruler with the disc in His hand protect me during the hours at noon see also ",
                " May Lord Madhusûdana with the fearful bow S ârnga protect me early in the afternoon May Mâdhava the Lord of Brahmâ Vishnu and S iva protect me in the late afternoon and may Lord Hrishîkes a protect me during the hours at dusk May Lord Padmanâbha the Lord from whose navel the universe sprang be the one protector during the entire evening early and late ",
                " May the Lord with the S rîvatsa mark protect me during the hours after midnight may Janârdana the Lord with the sword in His hand protect me late at night and may Lord Dâmodara protect me during the hours before dawn during which there is the brâhmamuhûrta May the Controller of the Universe the Supreme Lord in the form of time protect me as the kâla mûrti also the clock ",
                " Please let the sharp rimmed Sudars ana disc His order of time the cyclic of natural time that wielded by the Lord destructively moves in all directions alike the fire at the end of time burn to ashes the enemy forces the same way a blazing fire with its friend the wind would burn dry grass in an instant ",
                "We pray that whatever is disturbing us and our devotion will find its end as a logical consequence of the fact that it is You the Lord of time alone who decides what the ultimate reality would be of that what is and that what is not like happiness and grief coming and going see B G ",
                "The godly souls said You oh Lord awarding the results of sacrifice we offer our obeisances You using the cakra the disc the cyclic order of time as a weapon are the one to set the limits All our respect for You who are known by so many transcendental names ",
                "Therefore we do not really know whether Your Lordship is there like an ordinary human being bound to actions in the material world like someone who under the influence of the modes thus depends on time space activities and nature and thereby is forced to accept the good and bad results of his own actions or whether You are there as a completely self satisfied âtmârâma and self controlled person who never fails in his spiritual potency and is always a neutral witness ",
                "You on closer scrutiny are the essence of authenticity the controller of all and everything spiritual and material You are there as the cause of all causes of the entire universe who with all qualities are present within all up to the minutest atom You are with the temporality of all manifestations the only one who remains ",
                "Therefore oh Supreme Lord what can we as sparks of the original fire the golden seed that You are tell You You who personally are amused to be engaged in creation destruction and maintenance with Your divine energy You who as the Supersoul and spirit of the absolute Brahman resides in the hearts of all the different living beings and externally are present according to time place and the physical constitution You whom one realizes as the cause of that what constitutes the existence and consciousness of the living being You as the witness of all that is going on as the witnessing itself and the embodiment of the eternal memory of the entire universe the âkâs a record ",
                " Like birds caught in a net all worlds and their rulers sigh powerlessly under the time factor that is the cause out here People not aware of that time factor Him the Lord of Time the strength of our senses mind body life force death ànd immortality consider their indifferent body the cause Oh sir dear Indra please understand that all things thus oh generous one just like a wooden doll a woman made of wood or a cuddly animal of straw and leaves depend on Îs a the Power the Lord and master of Time constituting their life and coherence ",
                " Not knowing the Lord the time factor one considers oneself despite being fully dependent to be the one in control but it is He who creates beings by other living beings and it is He who devours them through others The blessings of longevity opulence fame and power arise when the time is ripe His time just as the opposite is found without having chosen for it ",
                " The way grains of sand wash ashore and drift apart by the force of the waves the embodied souls are brought together and separated by time compare B G ",
                "In the course of time eventually all people become each other s friends family members enemies neuters well wishers indifferent or envious ones compare B G ",
                "Engaged a different way as in demigod worship one lacks in consciousness and arrives in human society thus at the I and mine and the me and you of the false ego In approaches other than Yours one is as a result of the deviating vision impure in one s conduct time bound and full of adharma compare B G ",
                "He the One Supreme Lord by His potencies gives shape to the conditioned existence of all living beings as also to their liberation in devotional service He constitutes the reason for their happiness and distress as also for the position of being elevated with Him above time ",
                " When a person worships the Original Person he immediately gets a grip on his life as for time and measure And so it happened with Diti who for almost a year had worshiped the Lord see In order to compensate for the faults made by the mother the Lord changed the forty nine parts that Indra had created the Maruts into the fifty demigods together with Indra who became soma drinkers priests ",
                " According to the time of their prevalence one with the mode of sattva goodness finds the devas and the rishis the gods and sages with the mode of rajas passion one encounters Asuras the unenlightened ones and with the mode of tamas inertia one is faced with Yakshas and Râkshasas ghosts and demons see also B G ",
                "Oh ruler of man the true cause that is the male principle the original unmanifest foundation of matter pradhâna is the primal expanding movement of time as the fourth dimension which forms the shelter of the Lord to meditate upon see also B G Oh King also being this authentic notion of Time the Supreme Lord of name and fame increases in the mode of goodness the numbers of enlightened souls and is consequently as the friend of the demigods inimical to and destructive with the unenlightened souls the materialists who are ruled by passion and ignorance ",
                "The fire in wood can be observed separately just as the air within the body and the time effect of the all pervading ether that does not mix with anything The same way the living entity can be separately considered as transcendental to its material encasement of involvement with the modes ",
                " This is what he considers Lord Brahmâ who by his austerity absorbed in yoga created the moving and unmoving living beings see has his throne in all the worlds high and low I by dint of an even more severe penance than his being absorbed in yoga will from the eternality of time and the soul achieve the same for myself By my strength I will turn this world upside down and handle everything that is not right different from before What is the use of all other practices At the end of a day of creation time will vanquish all the worlds of Vishnu anyway ",
                "S rî Hiranyakas ipu said At the end of a day of creation when he Lord Brahmâ under the influence of time is covered by the dense darkness of ignorance this cosmic creation manifests again by the light of the rays emanating from his body This world endowed with the three basic qualities of rajas sattva and tamas passion goodness and ignorance is by him created maintained and annihilated That transcendental and supreme Lord I offer my respectful obeisances ",
                "Not affected by anything you are the personification of the ever vigilant Time that by each of its segments in the form of days hours and minutes and such reduces the duration of life of all beings You are the cause of life of this material world the Great Self and Supreme Controller who was never born ",
                "He the Supreme Controller of Time Urukrama the Lord of the Great Strides Vâmana is the one strength of one s mind and life the steadiness of one s physical power and senses He the True Self is the Supreme Master of the three basic qualities who by His different natural forces creates maintains and withdraws again the entire universe ",
                "S rî Indra said Our share of the sacrifices was secured by Your Lordship protecting us oh Supreme One We have no words to describe the degree our lotus like hearts were afflicted by the Daitya our hearts that are really Your residence Alas oh Lord how insignificant is our world in the grip of Time but for the sake of the devoted souls in Your service You have shed Your light so that they may find liberation from their bondage What else but considering the visible world as unimportant would constitute their way oh Nrisimhadeva ",
                "The keepers of the wealth the Yakshas said We serving You to Your pleasure belong to Your best followers This son of Diti forced us to carry his palanquin but caused the sorrow the poverty of each and everyone Thus we acknowledge You oh Lord Nrisimha for You are the one who put him to death oh twenty fifth principle that is the Time see ",
                "The godhead was by him such a little boy fallen at His lotus feet greatly moved and filled with mercy He raised His lotus hand placed it on his head and dispelled the fear for the snake of time from all minds present there ",
                " Whatever situation it may concern whatever seems to be the reason whatever time it might be by whatever agent and relating to whatever agent caused by whatever agent or for the sake of whatever agent whatever the way of something or of whatever nature something might be is certainly all nothing but another form of the Supreme Reality Stated differently in nature one finds because of all kinds of changes a specific form of separateness but whatever form it may concern it is always a manifestation of Your Lordship s energy The illusory nature of matter creates a mind that constitutes the source of fruitful actions or karma that are difficult to control These actions are conditioned by the Time that agitates the modes of nature and is respected in a certain way by the person Thus being defeated by the alluring but deluding material energy one is tied to the sixteen spokes of the senses of action and perception the elements and the mind of the wheel of rebirth oh Unborn One Who can escape from this without taking to Your way see also B G You are that one element of Time to the tender mercy of which the soul eternally is left being defeated by the modes of Your rule I present here who as a form of material energy in all his forsaking and appearing is subjected to Your cyclic control am powerless oh Lord and Master I am crushed under the wheel with the sixteen spokes Please help me this soul of surrender to get out of this oh Almighty One Oh Almighty One I have seen that people in general desire the longevity opulence and glory of the pious leaders of heaven But our father wishing this all for himself was simply by the laughter he provoked of Your expansion as Nṛsiṁha pulled down by You in the blink of an eye and destroyed Therefore I do not want to live as long as Lord Brahmâ does or be rich and mighty I know where all these foolish blessings of the senses of the embodied being lead to I have no desire to be finished by You so powerful as the Master of Time Please lead me to the association of Your servants ",
                "Having awakened from Your slumber on the bed of Ananta in the causal ocean the great lotus of all the worlds appeared from Your navel like a banana tree does from its seed That cosmic body of Yours this universe agitated by the Time factor constitutes Your way in the form of the modes and their divinities of dealing with the material affair with prakriti ",
                "By relishing your merit being happy by acting piously defeating sin by the rapid progress of time forsaking your body and by spreading your reputation throughout even the worlds of the gods you will freed from all attachment turn back to Me ",
                "This history describes the character of the devotion spiritual knowledge and renunciation of that most exalted devotee Prahlâda Try to understand each of these stories and thus discover what belongs to the Lord the Master of maintenance creation and destruction what His qualities and activities are the wisdom handed down in disciplic succession and how He by the time factor stands for the finality of all the higher and lower living beings and their cultures however great they might be ",
                "Oh ruler of man with the arrows joined on his bow Lord S iva thus being the Master and Controller at noon set the so difficult to pierce three cities afire ",
                "The above in verse described directions of the guru for the householder apply equally to the renunciate soul be it that the householder can have sexual intercourse for a certain period of time see also B G ",
                "He should not rejoice in the certainty of the death of the body nor in the uncertainty of its life he instead should observe the supreme command of Time that rules the manifestation and disappearance of all living beings ",
                "He who lives for the money is always afraid of the government of thieves of enemies relatives animals and birds of beggars of Time and of himself ",
                "When one at the appropriate time in good association being surrounded by persons of peace repeatedly listens to the nectar of the narrations about the Lord s avatâras one will gradually see the bonds slackened of the association with one s wife and children like one awakens from a dream see also and B G ",
                "On the threefold path of dharma artha and kâma not being too zealous not engaging in ugra karma or harmful actions a person according to time and circumstance should aspire for only as much as the grace of God would provide see also ",
                "What is the value of the attachment to this insignificant vehicle of time that is doomed to be eaten by the insects to turn into stool or into ashes What is the value of being attracted to the body of one s wife compared to the value of one s attraction for the soul that is as all pervading as the ether ",
                "It is by these auspicious times of being regular to natural occurrences that the fate of human beings is improved For the human being during all seasons to have auspiciousness success and longevity one therefore on those days must perform all kinds of ceremonies At all these natural times taking a holy bath doing japa the Vedic rosary performing fire sacrifices and keeping to vows constitutes a permanent benefit with whatever that is given in respect of the Supreme Lord the twice born souls managing the deities the forefathers the godly souls the human beings in general and all other living beings Oh King the purification rituals which serve the interest of having days with the wife the children and oneself as also serve the interest of having funerals memorial days and days for doing fruitful labor must be performed at the natural times relative to sun and moon meant for them ",
                "In exceeding this number of invitees or relatives with the s raddha ceremony things will not work out perfectly as for the most suitable time and place the paraphernalia the person to receive the honor and the method applied ",
                " The guru who is the light on the path must be considered the Supreme Lord in person and he who considers him and what he heard from him as being mortal and time bound is like an elephant that has bathed and thereafter takes a dust bath ",
                "One has wasted one s time when all the prescribed activities and observances designed for the definite subjugation of the six departments of the five senses and the mind have not led to the ultimate goal the connectedness in yoga of one s individual consciousness with Him ",
                "The fine substances of the sacrifice result in the smoke that is associated with the divinity of the night the dark half of the month the sun going through the south and the new moon compare B G By that divinity one finds the item grains that are the seeds of the vegetation on the earth s surface oh ruler of the earth Thus called into existence by the father of Time they by feeding us through the sacrifices lead to one after the other birth to the time and again regular assuming of a physical form to be present in this world see also B G ",
                "The body consisting of the five elements cannot exist without the sense objects belonging to it The untrue is found in the total form of that body which just like that what belongs to it in the end turns out to be a temporary appearance ",
                "Oh king a person should perform his duties according to his varnâs rama position in society engaging with the means the place and the time that are not scripturally forbidden and he should not follow any other course of action unless there is an emergency see also and B G ",
                "He the Lord protects anyone who is of surrender He protects those who are afraid of death against the so very strong serpent of time that chases someone endlessly with its terrifying force see B G I surrender to Him who is the refuge and for whom even death flees away ",
                "I do not want to live like this in the world What is the use of this captivation from within and from without in being born as an elephant I do not want the misery and destruction because of the time factor I want to be liberated from that covering of my spiritual existence see also ",
                "Let us offer our obeisances to the truth of Him whom one considers the axle of Lord Brahmâ s lightning fast revolving sacred wheel of Time with its fifteen spokes the knowing and working senses and the five airs three naves the modes and eight segments the five elements mind false ego and intelligence that feed one s thought process compare and B G ",
                "Greed is there from His lower lip and affection from His upper lip from His nose there is the bodily luster and from His touch animalistic love manifested From His brows there is the Lord of Death Yamarâja but from His eyelashes there is eternal Time May He the One of all Prowess be favorably disposed towards us The material elements their weaver kâla time fruitive labor karma the modes of nature the gunas and the diversity brought about by His creative potency yoga mâyâ constitute a difficult to fathom completeness from which the great sages turn away in their aversion against the delusional quality of the material world May He the Controller of All and Everything be contented with us ",
                "Not even the slightest activity properly performed for Your sake is in vain because being dedicated to the Controller who is the Time You are realized as the Original Soul friendly and beneficial to all persons ",
                "Do not fear the kâlakûtha false time poison that will appear from the ocean of milk and take care not to be led by greed lust or anger with the result of the churning ",
                "You are the source of the spiritual Vedic sound the origin of the universe the soul the life breath the senses and the elements You are the basic qualities of nature and the natural disposition the eternal time the sacrifice and the dharma of truth satya as also truthfulness rita It is unto you that one utters the original syllable consisting of the three letters A U M Oh soul of all godly souls fire constitutes your mouth oh Lord of all the worlds the surface of the globe is known as your lotus feet oh self of the gods time constitutes your movement the directions are your ears and the controller of the waters Varuna is your taste ",
                "In this world which has originated from you and at the time of her destruction is burned to ashes by you with the sparks of the fire emanating from your eyes you have out of your mercy for the living beings annihilated Tripura as also put an end to the sacrifices out of desire see e g the poison of false time in this story and many other forms of misery But these matters are not part of the praises offered to you since you ban this world from your mind ",
                "Someone might be of dharma but is he friendly towards other living beings Someone can be of renunciation but he might miss the cause of liberation A person may have power over people but still not be liberated from the great force of material nature from the power of time Someone may be free from the influence of the modes of nature but never be a second one another Lord of Control and Yoga see also Someone may live a long time but still not know how to behave and be happy someone may master the art of living but still not know how to get old And when someone knows the both of them such a person still might be unlucky in another respect Nor is of someone excelling in all walks of life said that he wishes Me in my position of devotion unto Vishnu ",
                "Bali retorted All present here on this battlefield are subjected to the rule of time and successively acquire with what they do a reputation achieve a victory suffer defeat and then find their death Because the entire world is moved by time an enlightened soul who sees this will not rejoice or complain In that sense you all pretty much lost your way compare B G ",
                "The moment one living with the time and all its different elements is joined with Me in the form of Eternal Time or the pure Time Spirit that illusory energy of the modes of nature the goddess Durgâ in sum will no longer be able to bewilder you ",
                "In the form of the founding fathers the Prajâpatis He creates offspring to annihilate the miscreants He assumes the form of kings and in the form of time He is there to put an end to everything that became different in following the basic qualities of nature ",
                "You as the original cause the dissolution and the maintenance of the universe are the reservoir of endless potencies whom one calls the Original Person You are the Lordship the Controller who is the Time that holds the entire universe in its grip the way waves drag someone along who fell in them ",
                "Seeing him with the trident in his hand coming towards Him like death personified the Chief of the Mystics the Knower of Time Lord Vishnu thought the following Wherever I go this one here like the death of each will also go I will therefore enter his heart he only looks outside himself ",
                "Formerly time worked in our favor and brought us the victory over the gods but today time which indeed is the Greatest Power the Supreme Authority in our existence works against us No man is able to surpass the time factor by any power counsel cleverness fortifications spells herbs diplomacy or by whatever other means ",
                "None of the controllers of the worlds will be able to overrule your command there not to speak of the common man for I with my cakra will personally take care of all the Daityas who defy your rule ",
                "As for the time and place the person the paraphernalia the practice of the mantras and following the principles faults can be made but these are all nullified by regularly chanting Your glories in congregation ",
                "S iva being pleased with him oh servant of the state in order to keep his promise to Umâ and to show the sage his love said This disciple of your line will one month be a female and the next month be a male Sudyumna may with this arrangement then rule the world as he likes ",
                "The so highly intelligent descendants of Angirâ see are today performing a sacrifice but on every sixth day that they do this oh learned one they will fall in illusion with their self interested actions ",
                "Pleased with his unalloyed devotional service the Lord gave him His cakra disc weapon that protects the devotees but is so fearful to the ones opposing Him see also and B G ",
                " Lord Brahmâ said At the end of my lifetime a dvi parârdha see when His pastimes have ended the Lord of the End Time Vishnu the Self of Time with a single movement of His eyebrows will destroy this universe including my heavenly abode I Lord S iva Daksha Bhrigu and the other sages as also the rulers of man the rulers of the living beings and the rulers of the demigods all carry out His orders and together bow for the salvation of all living beings our heads in surrender to the principle regulating our lives ",
                " Durvâsâ who scorched by Vishnu s chakra was turned down by Lord Brahmâ next for his shelter went to the one who always resides on Kailâsa Lord Shiva Shrî Shankara Shiva said My dear one we have no power over the Supreme One the Transcendence in Person with whom I the other living beings and even Lord Brahmâ wander around within the countless universes that together with us at times arise and then are destroyed again I Sanat and the other Kumâras Nârada the great Unborn Lord Kapila Vyâsadeva Devala the great sage Yamarâja Âsuri the saint Marîci and other masters of perfect knowledge headed by him have learned to know the limits of all there is to know but none of us can fully comprehend His illusory energy of mâyâ and that what is covered by it The weapon of the Controller of the Universe the cakra is even for us difficult to handle and you should therefore seek your refuge with the Lord who will certainly bestow upon you His happiness and fortune ",
                "All my respects for you the auspicious center of spin the measure for the complete of nature who are like a fire of destruction to the unenlightened souls who lack in pious conduct You the keeper of the three worlds with a wonderful effulgence are of a supreme goodness and act as fast as the mind I try to voice ",
                "He also in full awareness of the Super soul worshiped Yajña the Lord of Sacrifices the God and Supersoul of everyone elevated above the sensual plane This happened in sacrificial ceremonies that were attended by all the godly people whom he rewarded with large donations The ingredients the mantras and the regulative principles the worship and the worshiper as also the priests in their dharma of proceeding according to the time and place all together contributed to assure that the interest of the true self was done justice ",
                "With his mind thus controlled by the affection for his son he cheated the god with words about the time that it would take and made him wait ",
                "His son Dilîpa did just like his father not succeed and was also defeated by time Thereafter Dilîpa s son Bhagîratha performed severe austerities ",
                "Râma said to him You scum of the earth since you oh criminal like a dog have kidnapped My helpless wife I as Time itself as someone not failing in His heroism will personally punish you today for that shameless act you abominable evildoer see also B G ",
                "She understood that living with friends and relatives who are all subjected to the control of the rigid laws of nature Time is like associating with travelers at a water place It is a creation according to one s karma of the Lord s illusory potency The daughter of S ukrâcârya gave up all her attachments in this dreamlike world fixed her mind fully on Lord Krishna and shook off the worries about her self about both her gross and subtle nature the linga ",
                "The king well aware of what would befit the time and place said yes and then married according to the rules of dharma with S akuntalâ the gandharva way of mutual consent ",
                "Brihadratha begot with a second wife he had a son in two halves who because the mother rejected them by Jarâ the daughter of Time see also playfully were united while she said Come alive come alive Thus a son called Jarâsandha Jarâ s hermaphrodite was born who later became a vital enemy of Lord Krishna ",
                "When in the past on the battlefield my grandfathers the Pândavas were fighting with imperishable warriors like Devavrata Bhîshma and other great commanders who were like timingilas shark eaters they crossed the so very difficult to overcome ocean of Kaurava soldiers in the boat that He is as easily as one steps over a calf s hoof print This body of mine the only seed left of the Kurus and Pândavas was scorched by As vatthâmâ s weapon when I resided in the womb of my mother but it was by Him Krishna holding the cakra in His hand protected because my mother sought His protection and Oh man of learning please describe the glories of the Lord who by His own potency appeared as a normal human being of Him the Giver of Death and Eternal Life as one calls Him of Him who manifests Himself in physical forms bound to time of Him the Original Person who is present both inside and outside of all the embodied beings ",
                "Before we came here the Personality of Godhead knew already about the distress of mother earth Together with your good selves as His parts He wants to manifest Himself by taking birth in the family of the Yadus He wants you to be there with Him for the fulfillment of His mission for as long as He the Lord of Lords with His potency of Time moves around on this earth to diminish the burden of the planet ",
                " When after millions and millions of years the cosmos runs at its end the primary elements merge with their primal forms and everything that manifested by the force of Time turns into the unmanifest You oh Lord with the Many Names are the only one to remain This so powerful Time factor by which from the smallest measure of time up to the measure of a year this entire creation works is said to be Your action the movement of You the secure abode the Supreme Controller whom I offer my surrender ",
                "Material things are purified by time by washing and bathing them by rituals by penance by worship by charity and by contentment but the soul is purified by self realization ",
                "You are the One for all living beings You are the master of the life force of the body of the soul and the senses You are the Time the Supreme Lord Vishnu the Imperishable Controller You as the Greatest One who are both the cosmic creation and the subtle reality You consisting of passion goodness and slowness are the Original Personality Overseer and the Knower of the restless mind in all fields of action ",
                "Upananda Nanda s elder brother the oldest and wisest one with the greatest experience said in that meeting what according to the time and circumstances to the interest of Râma and Krishna would be the best thing to do ",
                " Balarâma said to Krishna These boys are no incarnated masters of enlightenment nor are these calves great sages You oh Supreme Controller only You are the One who manifests Himself in all the diversity of existence How can You be everything that exists at the same time Tell Me what exactly is Your word to this By saying these words Baladeva then with His Lordship arrived at an understanding of the situation ",
                "They were worshiped by the time factor kâla the individual nature svabhâva the reform by purification samskâra desire kâma fruitive action karma the modes guna and other powers the glory of whose appearances was defeated by His greatness see also B G ",
                "We pray for You as the Time for You as the Certainty with the Time and for You as the Witness of all Time measures Our prayers are there for You in the Form of the Universe for You as the One Supervising it All for You as its Supreme Creator and for You who are the Original Cause of the Universe ",
                "You are the Almighty Lord of the Creation Maintenance and Destruction of this universe who beginningless and without acting with the modes with the potency of Time endeavors to promote the balance in relation to the modes While impeccably playing Your game You by Your glance awaken the distinctive dormant characteristics of each of these modes ",
                "Oh Creator oh Lord of the Time and the Seasons You are the one who generated this universe filled with the appearances of the natural modes that are endowed with different personal propensities in varieties of talents and physical capabilities wombs and seeds and different mentalities and forms",
                "Please protect us Your people Your friends against this insurmountable deadly fire of Time Oh Master we at Your benevolent blessed feet which drive away all fear are incapable to escape from here ",
                "The roads no longer used faded away being overgrown by grass just like written texts do that not being studied by the brahmins wither away under the influence of time ",
                "Even though the place and time the items used the hymns the rituals the priests and the fire the officiating God conscious souls the performer of the sacrifice that what was sacrificed and the dharmic result are all part of the directly visible reality of His Absolute Truth of Him the Supreme Lord Beyond the Senses they with their borrowed intelligence considered Him arrogantly just an ordinary person ",
                "He constitutes the place and time the items used the hymns the rituals the priests and the fire the officiating God conscious souls the performer of the sacrifice the performance and its dharmic result see verse He the Supreme Lord Vishnu the Master of all Yoga Masters has directly visible taken birth among the Yadus but despite having heard about this we foolishly failed to understand that ",
                "S rî S uka said When Nanda and the elders heard these words being spoken by the Supreme Lord by the Time in person in order to break the pride of Indra they accepted them as excellent ",
                "As a new born child with hardly His eyes open He sucked the poisoned milk from the breast of the greatly powerful Pûtanâ in the process also sucking away her life air just like the force of time sucks away the youth from a body see ",
                "You are the father and the guru of the entire universe the Original Lord and the insurmountable Time who when You by Your own decision assume Your transcendental forms strives to be the authority to eradicate the self conceit of those who think they are the Lord of the Universe ",
                "When he saw the two approaching like Time and Death personified he became afraid In his confusion he left the women behind and ran for his life ",
                "When you have brought Them here I will have Them killed by the elephant that is as mighty as time itself And if They manage to escape that my wrestlers who are as strong as lightning will put an end to Them ",
                "Then I will see You as the charioteer of Arjuna assume the form of Time in bringing about the destruction of the armed forces of this world ",
                "But enough of that even for a fallen soul like me there is a chance to acquire the audience of Acyuta Some time someone pulled along by the river of time may reach the other shore ",
                "And when I have fallen at the base of His feet the Almighty One will place upon my head His lotus hand that dispels the fear for the serpent of time the snake because of whose swift force the people terrified seek shelter ",
                "Thus being threatened the elephant keeper got angry and goaded the furious elephant that was like Yama time and death in the direction of Krishna ",
                "When Krishna heard this He welcoming the fight and thus considering it desirable spoke words befitting the time and place see also ",
                "When Krishna the Supreme Lord Hari saw how his army like an ocean that overflowed its boundaries besieged His city and filled His subjects with fear He as the Ultimate Cause in a Human Form considered what to the purpose of His descent into this world would be the best course of action considering the time and place ",
                "Your children your queens and your other relatives ministers advisors and subjects do not live anymore Time has swept them away The Supreme Inexhaustible Lord of Control is the Time itself more powerful than the most powerful who playing a game of herdsman and flock sets the mortal beings in motion ",
                "Not even the greatest sages enumerating My births and activities which take place in respect of the three aspects of time past present future oh King can reach the end compare and ",
                "Nevertheless I never lament or rejoice for I know that the world is driven by Time and fate combined ",
                "Our enemies with the time in their favor have won now but then again when our time has come we will win ",
                "Desiring to create oh Master You stand out as being the Unborn One as Brahmâ for the purpose of annihilation You adopt the mode of ignorance as S iva and for the sake of maintenance You are manifested as the goodness as the Vishnu avatâras of the Universe Yet You are not covered by these basic qualities oh Lord of Jagat the Living Being that is the Universe Being Kâla time Pradhâna the unmanifested state of matter the primal ether and the Purusha the Original Person You nevertheless exist independently thereof ",
                "You are the Supreme Soul of all the Worlds who gives Himself away and about whose prowess the sages speak who gave up their staff for wandering around becoming Paramahamsas see You were for that reason chosen by me in rejection of those masters of heaven the one born on the lotus Brahmâ and the one ruling existence S iva What would my interest be in others whose aspirations are destroyed by the force of Time generated by Your eyebrows",
                "I approach You for being the negation of this mâyâ this material bewilderment of time fate karma the individual propensities the subtle elements the field that is the body the life force prâna the self the transformations the eleven senses and the aggregate of all of this in the form of the subtle body called the linga That illusory reality constitutes a never ending flow like that of seeds and sprouts ",
                "What would be unknown to You oh Master oh Witness of the Mind of all Beings whose vision is not impeded by time Nevertheless I shall speak as You wish ",
                " Look how wondrously inescapable Time moves on That what is a shoe now wants to step on a head that is ornamented with a crown ",
                " He saw Him arranging opulent weddings for daughters and sons in accordance with the vidhi at the right time with wives and husbands compatible to them ",
                "But Vaidarbhî Rukminî did not like that most auspicious time of the day because she then would have to miss the embrace of her beloved Krishna Mâdhava rose during the brâhma muhûrta the hour before sunrise touched water and cleared His mind to meditate on the unequaled exclusive self luminous Self beyond all dullness of matter This True Self dispels infallible as it is by its His own nature perpetually the impurity and gives the joy of existence It is known as the Brahman that with its His energies constitutes the cause of the creation and the destruction of this universe see also B G and ",
                "The whole world delighting in misconduct is bewildered about the duties out here to be of one s own worship for You according to Your varnâs rama command May there be the obeisances unto You the Ever Vigilant unblinking eye of Time who all of a sudden at the time of one s death cuts off that headstrong hope for longevity in this life ",
                "Hiranyagarbha the one of the golden light or Brahmâ and S arva he who kills by the arrow S iva see are but the instruments in universal creation and annihilation of the Supreme Lord of the Universe of You in the form of formless Time ",
                "We who in the past in our lusting about the wealth were blinded and quarreled with each other about ruling this earth have very mercilessly harassed our citizens oh Master and have with You in the form of death standing before us arrogantly disregarded You We oh Krishna have been forced to part with our opulence and were hurt in our pride by Your mercy in the form of the irresistible power of the Time that moves so mysteriously We beg You to allow us to live in the remembrance of Your feet ",
                "S rî S uka said The son of Prithâ thus having spoken with the permission of Krishna chose at a proper time for the sacrifice the priests who were suitable brahmins who were Vedic experts ",
                " Acyuta deserves the supreme position He is the Supreme Lord the leader of the Sâtvatas He stands for all the demigods as also for the place the time and the paraphernalia and such ",
                " The Vedic word of truth that Time is the unavoidable controller has by this been proven for even the intelligence of the elders could be led astray by the words of a boy ",
                "As he ran toward Him carrying his club Krishna severed his arm with a bhalla cutting arrow In order to kill S alva He next raised His wonderful disc weapon Looking like a mountain beneath a rising sun He shone with a radiation resembling the light at the end of time ",
                "His fame as praised in the Vedas the water washing from His feet and the words of the revealed scriptures thoroughly purify this universe see also B G Even though her wealth had been ravaged by Time the earth s vitality has been awakened by the touch of His lotus feet with her fulfilling all our desires like an abundance of rain By seeing Him in person by touching Him and walking with Him conversing lying down sitting eating being bound through marriage with Him and having Him as a blood relative you normally following the hellish path of family life have now found Vishnu liberation and heaven in person who constitutes the cessation of one s searching in life See also and and B G ",
                "By the light of Your personal form we are released from the bonds of the three states of material consciousness wakefulness dreaming and sleeping Being totally immersed therein we are of spiritual happiness having bowed down to You the goal of the perfected saints the paramahamsas who by the power of Your illusion have assumed this form for the protection of the unlimited and ever fresh Vedic knowledge which is threatened by time ",
                "Neither the fire the sun the moon nor the firmament neither the earth the water the ether the breath the speech nor the mind take away when they are worshiped the sins of someone entangled in material opposites But they are wiped away by just a few moments of service to men of brahminical learning ",
                "None of these kings who enjoy Your company nor the Vrishnis know You hiding behind the curtain of mâyâ as the Supreme Soul the Time and the Lord B G ",
                "The quality of the Lord His awareness is never disturbed by time dependent matters like the creation destruction and so on of this universe not by its own activity nor by another agency see B G and The consciousness of Him the One Controller without a Second is not affected by hindrances material actions and their consequences and the basic qualities of nature with their flow of changes kles a karma and guna Others though may consider Him as being covered by His own expansions of prâna and other elements of nature just like the sun is hidden by clouds snow or eclipses ",
                "An intelligent person should renounce the desire for wealth by means of sacrifices and charity He should give up the desire for a wife and kids by engaging in temple affairs With the help of the cakra order of Time the Time that is also the destroyer of all worlds see also and B G he should forget the desire for a world for himself oh Vasudeva All sages renounced their three types of desires for the wealth the family and one s own command of a household life and went into the forest for doing penances see also B G ",
                "Having taken birth from me You have now descended because of the kings who living in defiance of the scriptures and with their good qualities destroyed by the time of Kali yuga became a burden to the earth",
                "May there be my obeisances unto You who are the Supersoul for the knowers of the Supreme Spirit and the One who in the form of Time brings death to the conditioned soul You the One who assumes the forms of effect as also the forms of cause You whose vision is not covered by Your deluding potency but who are covered to our vision ",
                "Ah who out here who but recently was born and soon will die has an inkling of the One Who Came First from whom the leading seer Brahmâ arose who was followed by the two groups of demigods controlling the senses and the principles See B G When He lies down to withdraw at that time nothing remains of the gross and the subtle nor of that what comprises them both the bodies while also the flow of Time and the S âstras are no longer there B G ",
                "If the countless embodiments of the living beings would be eternal the omnipresent Time as a consequence would not be such a sovereign rule oh Unchanging One But it is not otherwise Because the substance cannot be independent from that from which it was generated pradhâna the primeval ether You the regulator who are the Time B G must be known as being equally present everywhere as the fourth dimension When one supposes that one knows You materially one is mistaken in the falsehood of an opinion on the local order see ",
                "Material nature prakriti and the person purusha do not find their existence at a particular point in time Not originating as such from one or the other it is from the combination of these two primordial elements that living bodies find their existence in You just as bubbles find their existence as a combination of water and air And just as rivers merge into the ocean and all flavors of flower nectar merge into the honey these living beings with all their different names and qualities in de end merge again in You the Supreme see also B G ",
                "Those who are wise understand the extend to which Your mâyâ bewilders human beings and frequently render traditional service unto You the source of liberation How could there for the souls who faithfully follow You be any kind of fear about a material existence a fear that by the three rimmed wheel of Time of past present and future by Your furrowing eyebrows repeatedly is raised in those who do not take shelter of You see also B G ",
                "Neither the masters of heaven nor even You can reach the end of Your glories oh Unlimited One oh You within whom the many universes by the drive of Time each in their own shell are blown about in the sky like particles of dust The s rutis bearing fruit by neti neti eliminating that what is not the Absolute Truth find in You their ultimate conclusion see siddhânta ",
                "Oh King when Krishna took His birth among the Yadus He outshone the pilgrimage site of the heavenly river the Ganges that washes from His feet Because of His embodiment friends and foes attained their goal The undefeated and supremely perfect goddess S rî belongs to Him she for whom others are struggling His name being heard or chanted is what destroys the inauspiciousness He settled the dharma for the lines of disciplic succession the schools of the sages With Lord Krishna holding the weapon that is the wheel of Time His cakra it is no wonder that the burden of the earth was removed see also ",
                "After in the house of the lord of the Yadus Vasudeva having performed most favorable rituals to bestow piety and take away the impurities of Kali yuga the sages Vis vâmitra Asita Kanva Durvâsâ Bhrigu Angirâ Kas yapa Vâmadeva Atri Vasishthha Nârada and others were by The Soul of Time Lord Krishna sent away to go to Pindâraka a pilgrimage site ",
                "Even though the Supreme Lord very well knew what it all meant He did not want to reverse the curse of the scholars and so He in His form of Time accepted it ",
                "When the dissolution of the material elements is at hand the Lord in the form of Time which has no beginning or an end withdraws the manifest universe consisting of the gross objects and subtle modes back into the not manifest see also ",
                "Fire by darkness deprived of its quality its form turns into air and the air losing its quality of touch dissolves into the ether When the sky the ether by the Supreme Soul of Time is deprived of its quality of sound it merges into the ego of not knowing ",
                " Brahmâ and all the other embodied beings have their existence like oxen bound by a rope through their nose Being controlled by Time they trouble each other May the lotus feet of You the Supreme Personality transcendental to both material nature and the individual person bring us transcendental happiness compare ",
                " You are the cause of the creation maintenance and annihilation of this universe You are the cause of the unseen of the individual soul and of the complete whole of the manifest reality They say that You this very same personality are the time factor controlling all who appears as a wheel divided in three summer winter and spring autumn One says that You are the Supreme Personality who in the form of Time uninterrupted in Your flow effects the decay of everything ",
                " The living being beginning with Mahâ Vishnu acquires its power potency from that time aspect of Yours You establish the vastness of matter with it mahat tattva United with that same nature You therefrom generate the way an ordinary fetus is produced the golden primeval egg of the universe endowed with its seven outer layers see kos a ",
                " Just as the realm of the ether is not touched by the winds that blow the clouds a person in his real self is not affected by his physical bodies consisting of fire water and earth that are moved by the basic qualities of nature created by Time ",
                " The state of the body one undergoes from one s birth until one s death changes by the course of Time that itself cannot be seen it is the body that changes not the soul just as the phases of the moon change but not the moon itself B G Just as with flames which one cannot see apart from a fire individual souls cannot be seen separately from the bodies that constantly die and are born again also the absolute of Time itself cannot be seen despite the relativity of its speeding compelling stream ",
                " A yogi with his senses accepts and forsakes sense objects depending the moment according to the cakra order and does not attach to them just as the sun with its rays engaged in evaporating and returning bodies of water is not ruled by them ",
                " How little happiness did the sensual pleasure give me and the men who pleased my senses The enjoyment of having a man or even the grace of the gods has being spread in time all its beginning and its end ",
                " When one has fallen in the well of a material existence by sensual pleasures has been robbed of one s insight and is caught in the grip of the snake of Time who else but the Original Lord would deliver one s soul see also The moment a soul attentively sees the universe as being seized by the snake of Time he being sober will detach from everything material and be fit to serve as his own protector ",
                " The one Self the One Lord without a second who became the Foundation and Reservoir of All is Nârâyana the Godhead who by His own potency created the universe in the beginning and by His potency of Time withdraws His creation within Hi mself at the end of the kalpa When the material powers of sattva and so on are balanced by the time factor that is the potency of the True Self the Soul the Lord the Original Personality the purusha is found as the Supreme Controller the Lord of both the primary nature pradhâna and the person He the worshipable object of all conditioned and transcendental souls has His existence in the purest experience that one describes as kaivalya or beatitude the fulness of the blissful state without guna attributes see also B G and By means of the pure potency of His Self His own bewildering energy composed of the three basic qualities He oh subduer of the enemies at the onset of creation agitating in the form of Time manifests the plan of matter the sûtra the thread the rule or direction of the mahat tattva see also ",
                "For the controllers and enjoyers of karma fruitful labor there is of course the eternal duality of happiness and grief time and place to have and to be When you take all that is matter for eternal and complete your intelligence is ruled by all the different forms and changes belonging to it All living beings thinking thus oh Uddhava time and again will find themselves being born falling ill and dying and so on see after all being united with a form one is bound to the conditioning limbs of time of sun and moon day and night etc ",
                "After for long having enjoyed the heavens until his pious credits were used up having exhausted his merits he against his will falls down from heaven not properly fixed being forced from his course by time compare B G ",
                "In all the worlds among all their leaders there exists fear of Me in the form of Time the individual souls living as long as a kalpa fear Me and even the one supreme Brahmâ who lives for two parârdhas fears Me see also ",
                "As long as one is not free from this dependence there will be fear for the Lord and Controller who is the Time They then who enjoy this karmic bond will become bewildered and always be full of sorrow With the given reciprocal action or the operation of the basic qualities of nature one calls Me variously the Time the Soul the Vedic Knowledge the World the original nature or Nature at large as also the Dharma ",
                "The doctrine followed the way one deals with water the people one associates with one s surroundings and the way one behaves with time one s occupational activities one s birth or social background as also the type of meditation mantras and purificatory rites one respects are the ten factors determining the prominence of a particular mode ",
                " When one has conquered the breathing process prânâyâma and has mastered the sitting postures âsana one should attentively step by step without slackening gather one s mind by concentrating on Me at appointed times to the positions of the sun and the moon see B G and ",
                "The yogi may obtain laghimâ lightness by attaching to Me as the supreme element of the smallest elements the atoms the subtle property of Time see also cakra ",
                "When one focusses one s consciousness on Vishnu the Original Controller of the Three gunas see also B G who is the mover in the form of Time one will obtain the siddhi of îs itvâ the supremacy by means of which the conditioned body the field and its knower can be controlled ",
                "A yogi pure of character who by dint of his devotion unto Me knows how to focus his mind dhâranâ acquires insight in the three phases of time past present and future including knowledge of birth and death see tri kâlika ",
                "One says though that they these siddhis for the one who practices the highest form of yoga the bhakti yoga by means of which one obtains everything thinkable from Me are a hindrance and a waste of time ",
                "I am the goal of those who seek progress the Time of those who exert control I am the equilibrium of the basic qualities of nature as also the virtue of those endowed with good qualities ",
                "Of that what constitutes a stable vision I am the solar year of the seasons I am spring among the months I am Mârgas îrsha November December and of the lunar mansions the twenty seven nakshatras I am Abhijit Among the yugas I am Satya yuga among the steady I am Devala and Asita of the Vedic editors I am Dvaipâyana Vyâsadeva and among the scholars learned in spirituality I am S ukrâcârya ",
                " In the forest one should allow the hair on one s head and body one s facial hair and nails to grow as also the filth of one s body not extensively clean one s teeth but bathe three times a day and at night sleep on the ground ",
                "One eats what is either prepared on a fire what has ripened by time or what was pulverized with a mortar with a stone or ground with one s teeth One should personally collect whatever that is needed for one s sustenance depending the place the time and one s energy and understand that living in the forest one must not store anything for another time see also ",
                "Please uplift this person who so badly craving for some insignificant happiness bitten by the snake of time hopelessly fell down in this dark pit of material existence Oh Greatest Authority pour out Your words of mercy that lead to liberation ",
                "They also discuss the differences within the varnâs rama system wherein the father may be of a higher anuloma or a lower pratiloma class than the mother they are about heaven and hell and expound on the subjects of having possessions one s age place and time see also and ",
                "What would be the right and wrong considerations concerning the time place the things and so on is established by Me with the purpose of restricting materially motivated activities ",
                "The time that by its nature solar position lunar phase or by its objects appointment by calendar and sundial is suitable for performing one s prescribed duties is considered good and the time that impedes the performance of one s duties or is unsuitable night time e g or times of different obligations is considered bad see also B G kâla and kâlakûtha The purity or impurity of a thing or of a substance is determined validated with the help of another thing in respect of what one says about it by means of a ritual performance of purification in respect of time or according to its relative magnitude ",
                "By a combination of time air fire earth and water or by each of them separately matters are purified like grains things made of wood clay and bone thread skins liquids and things won from fire ",
                "The purification derived from a mantra is a consequence of the correct knowledge about it The purification by a certain act is the consequence of one s dedication to Me Dharma religiosity prospers by the purity of the six factors as mentioned the place the time the substance the mantras the doer and the devotional act whereas godlessness adharma is produced by the contrary ",
                "In this world the mode of goodness is of knowledge light the mode of passion is of fruitive labor karma and the mode of ignorance is of a lack of wisdom The interaction of the modes is called Time and the combination of their natural disposition svabhâva constitutes the thread or lead symbolically worn by the first three varnas see also ",
                "The agitation of the modes takes place on the basis of the primal ether and leads to changes or pradhâna constitutes the cause of the time phenomena The principle of the intellect the mahat tattva see also therefrom gives rise to a false I awareness that is the cause of three different types of bewilderment emotion vaikârika ignorance tâmasa and sensual pleasure aindriya ",
                "My best one created bodies constantly find and lose their existence as a consequence of Time the imperceptible stream not seen for being that subtle ",
                "Oh Uddhava a part of the wealth of this so called brahmin was seized by his relatives some by thieves some by providence some by time some by common people and some by higher authorities see also ",
                "The brahmin said These people are not the cause of my happiness or distress nor can I blame the demigods my body the planets my karma or the time It is according to the standard authorities the s ruti nothing but the mind that causes someone to rotate in the cycle of material life ",
                "And if we say that time would be the cause of our happiness and distress where do we find the soul in that notion The soul is not equal to the time phenomena the way fire is not equal to its heat and snow is not equal to cold With whom must one be angry when there is no duality in the transcendental position see also B G and time quotes ",
                "By My agitation of the elementary nature in the form of time of kâla the modes of tamas rajas and sattva the gunas have manifested in order to fulfill the desires of the living entity ",
                "As arranged by Me the Supporter the Soul the energy of Time one rises up from or drowns in the mighty stream of the modes of this world in which one is bound to performing fruitive labor ",
                "Material nature prakriti the foundation of which is constituted by the causal transformed ingredient of the Supreme Person the purusha together with that what is the agitating agent viz Time kâla makes up the threefold of the Absolute Truth Brahman that I am ",
                "When the form of the universe that is pervaded by Me has manifested the planetary variety of its time periods of creation maintenance and decay this variety with its different worlds losing its synergy arrives at a dissolution into its five composing gross elements see yugas manvantaras and B G The mortal frame at the time of annihilation will merge with the item the item with the grains the grains with the earth and the earth with the fragrance Fragrance becomes merged with the water the water with its quality of taste the taste with the fire and the fire with the form Form merges with air air merges with touch and touch merges thereupon with the ether Ether merges with the subtle object of sound and the senses of sound etc become merged with their sources the gods of the sun and moon etc The sources My dear Uddhava merge with the mind of the ego of goodness the controller of the sound that dissolves in the original state of the elements the ego of slowness This all powerful primal elementary nature then merges with the cosmic intelligence mahat That greater principle dissolves in its own basic qualities and they in their turn merge with their ultimate abode the unmanifest state of nature that merges with the infallible Time Time merges with the individuality the jîva of the Supreme in command of the illusory potency and that individuality merges with Me the Supreme Self Unborn âtmâ who characterized by creation and annihilation is perfectly established in Himself and remains alone see also ",
                "Material substance the place the fruit of action time knowledge activity the performer faith the state of consciousness and the species and destinations of life thus all belong to the three gunas ",
                "The individual soul of identification with his attention directed at the body the senses the life force and the mind assumes depending the gunas and the karma his form within the great universal Self With the lead the sûtra of the complete of nature thus very differently denominated as a dog ape or human being he then controlled by time moves around in material existence ",
                "Spiritual knowledge jñâna entails the discrimination of spirit and matter and is nourished by scripture and penance personal experience historical accounts and logical inference It is based upon that what is there in the beginning what stays the same in between and what remains in the end of this creation namely the Time and Ultimate Cause of brahman the Absolute Truth see also B G and kâla ",
                "Either alone or in association one should with respect for the position of the moon e g at special occasions and at festivals engage in singing and dancing and so on with royal opulence and generous contributions ",
                "S rî S uka said And then oh King under the strong influence of the time of Kali yuga religiousness truthfulness cleanliness tolerance and mercy as also the quality of life physical strength and memory will diminish day after day see also ",
                "Oh King of whatever that kings may enjoy in the world with all their power is by Time nothing more preserved than some accounts and histories compare with ",
                "Prithu Purûravâ Gâdhi Nahusha Bharata Kârtavîryârjuna Mândhâtâ Sagara Râma Khathvânga Dhundhuhâ or Kuvalayâs va Raghu Trinabindu Yayâti S aryâti S antanu Gaya Bhagîratha Kakutstha Naishadha Nala from the descendants of Nishadha Nriga Nâbhâga Hiranyakas ipu Vritra Râvana who made the whole world lament Namuci S ambara Bhauma Hiranyâksha and Târaka as also many other demons and kings of great control over others were all heroes who well informed were unconquerable and subdued everyone Living for me oh mighty one they expressed great possessiveness but by the force of Time being subjected to death they failed to accomplish their goals historical accounts is all that remained of them see also B G ",
                "The honorable king Parîkchit said By what means my Lord do the people living in Kali yuga eradicate the faults accumulating because of that age Please explain that to me How about the yugas the duties prescribed for them the time they last and when they end as also the Time itself that represents the movement of the Controller of Lord Vishnu the Supreme Soul see also time quotes page ",
                "The qualities of goodness passion and ignorance that thus depending the age are observed in a person undergo being impelled by the operation of Time permutations within the mind ",
                "S rî S uka said Time beginning with the smallest unit of the atom and culminating in the two halves or parârdhas of the life of Brahmâ oh King has been described in together with the duration of the yugas Now hear about the annihilation of a kalpa ",
                "Fire then takes away the taste of water after which it deprived of this quality dissolves Next follows fire that by air is deprived of its form because it takes its quality of touch away after which the air enters the ether that takes away that quality Then oh King the ether dissolves in the original element of nature âdi false ego in ignorance that takes away its quality of sound Subsequently the senses are seized by the vital power of the universe tejas or false ego in passion my best while the gods are absorbed by the universal modification vikara the false ego of goodness Cosmic intelligence mahat seizes the false ego with all its functions after which mahat is absorbed by the modes of nature of sattva and so on These three modes oh King are then under the pressure of Time overtaken by the inexhaustible doer the original unmanifest form of nature The original doer is not subject to transformation in divisions of time shad ûrmi and such qualities being unmanifest without a beginning and an end it or He constitutes the infallible eternal cause ",
                "This state constitutes the prâkritika pralaya dissolution wherein all the material elements of nature and energies of the unseen Original Person are completely dismantled by Time and helplessly merge ",
                "Even though it is knowable to us the changeable nature of the phenomenal world or even a single atom can in no way be explained without or as standing apart from the Self inside of the Time the Lord the expansion of the universe the fourth dimension for if that would be so if there would be not such a Self it should being equal to the consciousness stay the way it is ",
                " The more or less favorable living conditions of all living beings subject to transformation are rapidly and continuously wiped out by the mighty force of the current of Time and constitute the causes of their birth and death These states of existence created by the Time the form of the Lord without a beginning or an end one does not see directly just as the movements of the planets in the sky are not seen directly see also ",
                "The brahmin sages inspired by the Infallible Lord situated in their hearts came to that dividing among each other of the Vedas when they saw that under the influence of time the intelligence of the people diminished the life span shortened and the strength weakened see also ",
                "Srî Yâjñavalkya said My obeisances unto the Supreme Personality of Godhead who appearing as the sun and just like the ether in the form of the Supersoul inside and in the form of Time outside is present in the hearts of the four kinds of living entities beginning from Brahmâ down to the blades of grass as born from wombs eggs moist and seed see also You who cannot be covered by material terms all by Yourself with the flow of years made up of the tiny fragments of kshanas lavas and nimeshas see carry out the maintenance of this universe by taking away and returning its water in the form of rain ",
                "We know of nothing else but the attainment of Your feet the very form of liberation oh Lord that benefits the person who has to fear from all sides Brahmâ whose time takes two parârdhas is most afraid on account of this because You are the Time And how much more would that not be true for the worldly entities created by him see ",
                " He overlooked the entire expanse of all the stars the mountains and oceans the directions of the great islands and continents the enlightened and unenlightened souls the forests countries rivers cities and mines the peasant villages the cow pastures and the various engagements of the varnâs rama society Of this universe being manifested as real he saw the basic elements of nature and all their gross manifestations as also the Time itself in the form of the different yugas and kalpas and whatever other object of use in material life ",
                " May you have knowledge of the threefold nature of time tri kâlika oh brahmin as also wisdom in combination with a free heart May there for you being blessed with brahminical potency be the status of teacher of the Purâna ",
                "Oh best one of Bhrigu S aunaka the story I described is infused with the potency of the Lord with the Chariot wheel in His hand Krishna as the Lord of Time for anyone who hears it himself or makes someone else listen to it there will never be a repetition of births based on karma a worldly conditioned existence ",
                "The club He carries constitutes the principle element of prânaor the vital air relating to the sensory power physical power and the power of mind His excellent conch shell is the element water and His Sudars ana disc is the principle of tejas the vital power the dignity the fire in opposition His sword is pure as the atmosphere the ether element His shield consists of the mode of ignorance His bow S ârnga is the specific order or spirit the rûpa of time and His quiver of arrows consists of the karma the action or the karmendriyas ",
                "The cyclic order of time viz the sun and the moon constitutes the exercise of respect for the Godhead spiritual initiation dîkshâ is the purification process for the spiritual soul and devotional service to the Fortunate One is how one puts an end to a bad course sin ",
                "Sûta said This regulator of all the planets the sun revolving in their midst around mount Meru see was by the Lord in the form of Time created from the proto material primal energy pradhâna of Vishnu the Supreme Soul of all embodied beings ",
                "Oh brahmin the material energy of the Lord is thus described in nine the time the place the endeavor the performer the instrument the specific ritual the scripture the paraphernalia and the result compare B G ",
                "The Supreme Lord assuming the form of Time is there for the regulation of the planetary motion to the rule of twelve months or mâsas see also B G beginning with Madhu In each of the twelve He accompanying the sun god moves differently with His six associates He as a certain Deva together with a different Apsara Râkshasa Nâga Yaksha sage and Gandharva ",
                "All these personalities constitute the glories of Vishnu the Supreme Personality of Godhead in the form of the sun god they take away the sinful reactions of everyone who in the morning and the evening day after day remembers them ",
                "The gross and subtle movements of time are also discussed including the generation of the lotus and the killing of Hiranyâksha for the sake of delivering the earth from the ocean ",
                " Finally there is an account about Vishnurata Parîkchit the intelligent saintly king who had to relinquish his body the story of how the seer Vyâsa and others conveyed the branches of the Veda the pious narration about Mârkandeya the composition of the universal form of the Mahâpurusha and the arrangement of time in relation to the sun the self of the universe ",
                "When one self controlled and fasting studies this collection of verses at the holy places of Pushkara Mathurâ or Dvârakâ one will be freed from the fear of time or of a material life see also ",
                "Thus one following the other the saintly kings received this science understanding it that way but in the course of time in this world this great way of connecting oneself was scattered o subduer of the enemies ",
                "Whenever and wherever it is sure that one weakens in righteousness and a predominance of injustice does manifest o descendant of Bharata at that time I do manifest Myself ",
                "Surely there exists nothing of knowledge in this world that can compare to this purification and he who is mature in his own yoga will enjoy that in due course of time within himself ",
                "When surely he is never for the good of the senses engaged in the necessary fruitive labor at that time he is a renouncer of selfhood elevated in yoga one says ",
                "Those who know Me as ruling all as well as the godly and the sacrifices also they with their minds connected in Me even know me at the time of their death too ",
                "Therefore go on remembering Me at all times and fighting with your mind and intelligence surrendered to Me certainly you will attain Me without doubt ",
                "According the Vedas there are these two ways of light and darkness in passing from this world by which one either does not return or does return again Of knowing any of these different path s o son of Prithâ the yogi is never bewildered therefore always get unified in yoga o Arjuna ",
                "This material nature which is one of My energies is working under My direction O son of Kuntî producing all moving and nonmoving beings Under its rule this manifestation is created and annihilated again and again ",
                "Of the Daityas non theist sons of Diti who churned the ocean I am Prahlâda of what rules I am the Time of the animals the lion and of the birds I am Garuda Vainateya ",
                "Of the letters I am the first one the A of the compound words I am the dual one and certainly am I the eternal of Time and the Creator facing all directions Brahmâ ",
                "See here and now the universe completely all at the same time with all that moves and not moves in this body of Mine o conqueror of sleep and also whatever else you wish to see ",
                "Time I am the great destroyer of the worlds engaged here in destroying all people except for you brothers only will all the soldiers who are situated on both sides find their end ",
                "When one following that tries to see that the diversity of the living beings is resting in oneness and that it expanded to that reality at that time one attains the Absolute of the Spirit ",
                "When to all the gates of the body the enlightenment of knowledge develops at that time one says is the mode of goodness prevailing ",
                "Donations given dutifully irrespective the return at the proper time and place and to suitable persons that giving is considered to be of goodness ",
                "That charity which is given at the wrong place the wrong time and to unworthy persons and as well is given without respect and proper attention that is said to be in the mode of ignorance ",
                "The Supreme Master resides in the heart of all living entities o Arjuna directing each creature subject to the mechanical of time and matter ",
            )
        )
    )


# selfhelp text
def selfhelp():
    speak(
        random.choice(
            (
                "If you want to achieve greatness stop asking for permission. ~Anonymous",
                "Things work out best for those who make the best of how things work out. ~John Wooden",
                "To live a creative life, we must lose our fear of being wrong. ~Anonymous",
                "If you are not willing to risk the usual you will have to settle for the ordinary. ~Jim Rohn",
                "Trust because you are willing to accept the risk, not because it's safe or certain. ~Anonymous",
                "Take up one idea. Make that one idea your life - think of it, dream of it, live on that idea. Let the brain, muscles, nerves, every part of your body, be full of that idea, and just leave every other idea alone. This is the way to success. ~Swami Vivekananda",
                "All our dreams can come true if we have the courage to pursue them. ~Walt Disney",
                "Good things come to people who wait, but better things come to those who go out and get them. ~Anonymous",
                "If you do what you always did, you will get what you always got. ~Anonymous",
                "Success is walking from failure to failure with no loss of enthusiasm. ~Winston Churchill",
                "Just when the caterpillar thought the world was ending, he turned into a butterfly. ~Proverb",
                "Successful entrepreneurs are givers and not takers of positive energy. ~Anonymous",
                "Whenever you see a successful person you only see the public glories, never the private sacrifices to reach them. ~Vaibhav Shah",
                "Opportunities don't happen, you create them. ~Chris Grosser",
                "Try not to become a person of success, but rather try to become a person of value. ~Albert Einstein",
                "Great minds discuss ideas; average minds discuss events; small minds discuss people. ~Eleanor Roosevelt",
                "I have not failed. I've just found 10,000 ways that won't work. ~Thomas A. Edison",
                "If you don't value your time, neither will others. Stop giving away your time and talents- start charging for it. ~Kim Garst",
                "A successful man is one who can lay a firm foundation with the bricks others have thrown at him. ~David Brinkley",
                "No one can make you feel inferior without your consent. ~Eleanor Roosevelt",
                "The whole secret of a successful life is to find out what is one's destiny to do, and then do it. ~Henry Ford",
                "If you're going through hell keep going. ~Winston Churchill",
                "The ones who are crazy enough to think they can change the world, are the ones that do. ~Anonymous",
                "Don't raise your voice, improve your argument. ~Anonymous",
                "What seems to us as bitter trials are often blessings in disguise.~ Oscar Wilde",
                "The meaning of life is to find your gift. The purpose of life is to give it away. ~Anonymous",
                "The distance between insanity and genius is measured only by success. ~Bruce Feirstein",
                "When you stop chasing the wrong things you give the right things a chance to catch you. ~Lolly Daskal",
                "Don't be afraid to give up the good to go for the great. ~John D. Rockefeller",
                "No masterpiece was ever created by a lazy artist.~ Anonymous",
                "Happiness is a butterfly, which when pursued, is always beyond your grasp, but which, if you will sit down quietly, may alight upon you. ~Nathaniel Hawthorne",
                "If you can't explain it simply, you don't understand it well enough. ~Albert Einstein",
                "Blessed are those who can give without remembering and take without forgetting. ~Anonymous",
                "Do one thing every day that scares you. ~Anonymous",
                "What's the point of being alive if you don't at least try to do something remarkable. ~Anonymous",
                "Life is not about finding yourself. Life is about creating yourself. ~Lolly Daskal",
                "Nothing in the world is more common than unsuccessful people with talent. ~Anonymous",
                "Knowledge is being aware of what you can do. Wisdom is knowing when not to do it. ~Anonymous",
                "Your problem isn't the problem. Your reaction is the problem. ~Anonymous",
                "You can do anything, but not everything. ~Anonymous",
                "Innovation distinguishes between a leader and a follower. ~Steve Jobs",
                "There are two types of people who will tell you that you cannot make a difference in this world: those who are afraid to try and those who are afraid you will succeed. ~Ray Goforth",
                "Thinking should become your capital asset, no matter whatever ups and downs you come across in your life. ~Dr. APJ Kalam",
                "I find that the harder I work, the more luck I seem to have. ~Thomas Jefferson",
                "The starting point of all achievement is desire. ~Napolean Hill",
                "Success is the sum of small efforts, repeated day-in and day-out. ~Robert Collier",
                "If you want to achieve excellence, you can get there today. As of this second, quit doing less-than-excellent work. ~Thomas J. Watson",
                "All progress takes place outside the comfort zone. ~Michael John Bobak",
                "You may only succeed if you desire succeeding; you may only fail if you do not mind failing. ~Philippos",
                "Courage is resistance to fear, mastery of fear - not absense of fear. ~Mark Twain",
                "Only put off until tomorrow what you are willing to die having left undone. ~Pablo Picasso",
                "People often say that motivation doesn't last. Well, neither does bathing - that's why we recommend it daily. ~Zig Ziglar",
                "We become what we think about most of the time, and that's the strangest secret. ~Earl Nightingale",
                "The only place where success comes before work is in the dictionary. ~Vidal Sassoon",
                "The best reason to start an organization is to make meaning; to create a product or service to make the world a better place. ~Guy Kawasaki",
                "I find that when you have a real interest in life and a curious life, that sleep is not the most important thing. ~Martha Stewart",
                "It's not what you look at that matters, it's what you see. ~Anonymous",
                "The road to success and the road to failure are almost exactly the same. ~Colin R. Davis",
                "The function of leadership is to produce more leaders, not more followers. ~Ralph Nader",
                "Success is liking yourself, liking what you do, and liking how you do it. ~Maya Angelou",
                "As we look ahead into the next century, leaders will be those who empower others. ~Bill Gates",
                "A real entrepreneur is somebody who has no safety net underneath them. ~Henry Kravis",
                "The first step toward success is taken when you refuse to be a captive of the environment in which you first find yourself. ~Mark Caine",
                "People who succeed have momentum. The more they succeed, the more they want to succeed, and the more they find a way to succeed. Similarly, when someone is failing, the tendency is to get on a downward spiral that can even become a self-fulfilling prophecy. ~Tony Robbins",
                "When I dare to be powerful - to use my strength in the service of my vision, then it becomes less and less important whether I am afraid. ~Audre Lorde",
                "Whenever you find yourself on the side of the majority, it is time to pause and reflect. ~Mark Twain",
                "The successful warrior is the average man, with laser-like focus. ~Bruce Lee",
                "Take up one idea. Make that one idea your life -- think of it, dream of it, live on that idea. Let the brain, muscles, nerves, every part of your body, be full of that idea, and just leave every other idea alone. This is the way to success. ~Swami Vivekananda",
                "Develop success from failures. Discouragement and failure are two of the surest stepping stones to success. ~Dale Carnegie",
                "If you don't design your own life plan, chances are you'll fall into someone else's plan. And guess what they have planned for you? Not much. ~ Jim Rohn",
                "If you genuinely want something, don't wait for it -- teach yourself to be impatient. ~Gurbaksh Chahal",
                "Don't let the fear of losing be greater than the excitement of winning. ~Robert Kiyosaki",
                "If you want to make a permanent change, stop focusing on the size of your problems and start focusing on the size of you! ~T. Harv Eker",
                "You can't connect the dots looking forward; you can only connect them looking backwards. So you have to trust that the dots will somehow connect in your future. You have to trust in something - your gut, destiny, life, karma, whatever. This approach has never let me down, and it has made all the difference in my life. ~Steve Jobs",
                "Successful people do what unsuccessful people are not willing to doDon't wish it were easier, wish you were better. ~Jim Rohn",
                "The number one reason people fail in life is because they listen to their friends, family, and neighbors. ~Napoleon Hill",
                "The reason most people never reach their goals is that they don't define them, or ever seriously consider them as believable or achievable. Winners can tell you where they are going, what they plan to do along the way, and who will be sharing the adventure with them. ~Denis Watiley",
                "In my experience, there is only one motivation, and that is desire. No reasons or principle contain it or stand against it. ~Jane Smiley",
                "Success does not consist in never making mistakes but in never making the same one a second time. ~George Bernard Shaw",
                "I don't want to get to the end of my life and find that I lived just the length of it. I want to have lived the width of it as well. ~Diane Ackerman",
                "You must expect great things of yourself before you can do them. ~Michael Jordan",
                "Motivation is what gets you started. Habit is what keeps you going. ~Jim Ryun",
                "People rarely succeed unless they have fun in what they are doing. ~Dale Carnegie",
                "There is no chance, no destiny, no fate, that can hinder or control the firm resolve of a determined soul. ~Ella Wheeler Wilcox",
                "Our greatest fear should not be of failure but of succeeding at things in life that don't really matter. ~Francis Chan",
                "You've got to get up every morning with determination if you're going to go to bed with satisfaction. ~George Lorimer",
                "To be successful you must accept all challenges that come your way. You can't just accept the ones you like. ~Mike Gafka",
                "Success is...knowing your purpose in life, growing to reach your maximum potential, and sowing seeds that benefit others. ~ John C. Maxwell",
                "Be miserable. Or motivate yourself. Whatever has to be done, it's always your choice. ~Wayne Dyer",
                "To accomplish great things, we must not only act, but also dream, not only plan, but also believe.~ Anatole France",
                "Most of the important things in the world have been accomplished by people who have kept on trying when there seemed to be no help at all. ~Dale Carnegie",
                "You measure the size of the accomplishment by the obstacles you had to overcome to reach your goals. ~Booker T. Washington",
                "Real difficulties can be overcome; it is only the imaginary ones that are unconquerable. ~Theodore N. Vail",
                "It is better to fail in originality than to succeed in imitation. ~Herman Melville",
                "Fortune sides with him who dares. ~Virgil",
                "Little minds are tamed and subdued by misfortune; but great minds rise above it. ~Washington Irving",
                "Failure is the condiment that gives success its flavor. ~Truman Capote",
                "Don't let what you cannot do interfere with what you can do. ~John R. Wooden",
                "You may have to fight a battle more than once to win it. ~Margaret Thatcher",
            )
        )
    )


# facts text
def facts():
    speak(
        random.choice(
            (
                "The speed of light is generally rounded down to 186,000 miles per second. In exact terms it is 299,792,458 m/s.",
                "It takes 8 minutes 17 seconds for light to travel from the Sun’s surface to the Earth.",
                "October 12th, 1999 was declared “The Day of Six Billion” based on United Nations projections.",
                "10 percent of all human beings ever born are alive at this very moment.",
                "The Earth spins at 1,000 mph but it travels through space at an incredible 67,000 mph.",
                "Every year over one million earthquakes shake the Earth.",
                "When Krakatoa erupted in 1883, its force was so great it could be heard 4,800 kilometres away in Australia.",
                "The largest ever hailstone weighed over 1kg and fell in Bangladesh in 1986.",
                "Every second around 100 lightning bolts strike the Earth.",
                "Every year lightning kills 1000 people.",
                "In October 1999 an Iceberg the size of London broke free from the Antarctic ice shelf .",
                "If you could drive your car straight up you would arrive in space in just over an hour.",
                "Human tapeworms can grow up to 22.9m.",
                "The Earth is 4.56 billion years old...the same age as the Moon and the Sun.",
                "The dinosaurs became extinct before the Rockies or the Alps were formed.",
                "Female black widow spiders eat their males after mating.",
                "When a flea jumps, the rate of acceleration is 20 times that of the space shuttle during launch.",
                "If our Sun were just inch in diameter, the nearest star would be 445 miles away.",
                "The Australian billygoat plum contains 100 times more vitamin C than an orange.",
                "Astronauts cannot belch – there is no gravity to separate liquid from gas in their stomachs.",
                "The air at the summit of Mount Everest, 29,029 feet is only a third as thick as the air at sea level.",
                "One million, million, million, million, millionth of a second after the Big Bang the Universe was the size of a ...pea.",
                "DNA was first discovered in 1869 by Swiss Friedrich Mieschler.",
                "The molecular structure of DNA was first determined by Watson and Crick in 1953.",
                "The first synthetic human chromosome was constructed by US scientists in 1997.",
                "The thermometer was invented in 1607 by Galileo.",
                "Englishman Roger Bacon invented the magnifying glass in 1250.",
                "Alfred Nobel invented dynamite in 1866.",
                "Wilhelm Rontgen won the first Nobel Prize for physics for discovering X-rays in 1895.",
                "The tallest tree ever was an Australian eucalyptus – In 1872 it was measured at 435 feet tall.",
                "Christian Barnard performed the first heart transplant in 1967 – the patient lived for 18 days.",
                "The wingspan of a Boeing 747 is longer than the Wright brother’s first flight.",
                "An electric eel can produce a shock of up to 650 volts.",
                "‘Wireless’ communications took a giant leap forward in 1962 with the launch of Telstar, the first satellite capable of relaying telephone and satellite TV signals.",
                "The earliest wine makers lived in Egypt around 2300 BC.",
                "The Ebola virus kills 4 out of every 5 humans it infects.",
                "In 5 billion years the Sun will run out of fuel and turn into a Red Giant.",
                "Giraffes often sleep for only 20 minutes in any 24 hours. They may sleep up to 2 hours (in spurts – not all at once), but this is rare. They never lie down.",
                "A pig’s orgasm lasts for 30 minutes.",
                "Without its lining of mucus your stomach would digest itself.",
                "Humans have 46 chromosomes, peas have 14 and crayfish have 200.",
                "There are 60,000 miles of blood vessels in the human body.",
                "An individual blood cell takes about 60 seconds to make a complete circuit of the body.",
                "Utopia ia a large, smooth lying area of Mars.",
                "On the day that Alexander Graham Bell was buried the entire US telephone system was shut down for 1 minute in tribute.",
                "The low frequency call of the humpback whale is the loudest noise made by a living creature.",
                "The call of the humpback whale is louder than Concorde and can be heard from 500 miles away.",
                "A quarter of the world’s plants are threatened with extinction by the year 2010.",
                "Each person sheds 40lbs of skin in his or her lifetime.",
                "At 15 inches the eyes of giant squids are the largest on the planet.",
                "The largest galexies contain a million, million stars.",
                "The Universe contains over 100 billion galaxies.",
                "Wounds infested with maggots heal quickly and without spread of gangrene or other infection.",
                "More germs are transferred shaking hands than kissing.",
                "The longest glacier in Antarctica, the Almbert glacier, is 250 miles long and 40 miles wide.",
                "The fastest speed a falling raindrop can hit you is 18mph.",
                "A healthy person has 6,000 million, million, million haemoglobin molecules.",
                "A salmon-rich, low cholesterol diet means that Inuits rarely suffer from heart disease.",
                "Inbreeding causes 3 out of every 10 Dalmation dogs to suffer from hearing disability.",
                "The world’s smallest winged insect, the Tanzanian parasitic wasp, is smaller than the eye of a housefly.",
                "If the Sun were the size of a beach ball then Jupiter would be the size of a golf ball and the Earth would be as small as a pea.",
                "It would take over an hour for a heavy object to sink 6.7 miles down to the deepest part of the ocean.",
                "There are more living organisms on the skin of each human than there are humans on the surface of the earth.",
                "The grey whale migrates 12,500 miles from the Artic to Mexico and back every year.",
                "Each rubber molecule is made of 65,000 individual atoms.",
                "Around a million, billion neutrinos from the Sun will pass through your body while you read this sentence...and now they are already past the Moon.",
                "Quasars emit more energy than 100 giant galaxies.",
                "Quasars are the most distant objects in the Universe.",
                "The saturn V rocket which carried man to the Moon develops power equivalent to fifty 747 jumbo jets.",
                "Koalas sleep an average of 22 hours a day, two hours more than the sloth.",
                "Light would take .13 seconds to travel around the Earth.",
                "Males produce one thousand sperm cells each second – 86 million each day.",
                "Neutron stars are so dense that a teaspoonful would weigh more than all the people on Earth.",
                "One in every 2000 babies is born with a tooth.",
                "Every hour the Universe expands by a billion miles in all directions.",
                "Somewhere in the flicker of a badly tuned TV set is the background radiation from the Big Bang.",
                "Even travelling at the speed of light it would take 2 million years to reach the nearest large galaxy, Andromeda.",
                "The temperature in Antarctica plummets as low as -35 degrees celsius.",
                "At over 2000 kilometres long The Great Barrier Reef is the largest living structure on Earth.",
                "A thimbleful of a neutron star would weigh over 100 million tons.",
                "The risk of being struck by a falling meteorite for a human is one occurence every 9,300 years.",
                "The driest inhabited place in the world is Aswan, Egypt where the annual average rainfall is .02 inches.",
                "The deepest part of any ocean in the world is the Mariana trench in the Pacific with a depth of 35,797 feet.",
                "The largest meteorite craters in the world are in Sudbury, Ontario, canada and in Vredefort, South Africa.",
                "The largest desert in the world, the Sahara, is 3,500,000 square miles.",
                "The largest dinosaur ever discovered was Seismosaurus who was over 100 feet long and weighed up to 80 tonnes.",
                "The African Elephant gestates for 22 months.",
                "The short-nosed Bandicoot has a gestation period of only 12 days.",
                "The mortality rate if bitten by a Black Mamba snake is over 95%.",
                "In the 14th century the Black Death killed 75,000,000 people. It was carried by fleas on the black rat.",
                "A dog’s sense of smell is 1,000 times more sensitive than a humans.",
                "A typical hurricane produces the nergy equivalent to 8,000 one megaton bombs.",
                "90% of those who die from hurricanes die from drowning.",
                "To escape the Earth’s gravity a rocket need to travel at 7 miles a second.",
                "If every star in the Milky Way was a grain of salt they would fill an Olympic sized swimming pool.",
                "Microbial life can survive on the cooling rods of a nuclear reactor.",
                "Micro-organisms have been brought back to life after being frozen in perma-frost for three million years.",
                "Our oldest radio broadcasts of the 1930s have already travelled past 100,000 stars.",
                "Butterflies cannot fly if their body temperature is less than 86 degrees.",
                "Neurons multiply at a rate 250,000 neurons per minute during early pregnancy.",
                "Elephants have the longest pregnancy in the animal kingdom at 22 months. The longest human pregnancy on record is 17 months, 11 days.",
                "A female oyster produces 100 million young in her lifetime, the typical hen lays 19 dozen eggs a year, and it is possible for one female cat to be responsible for the birth of 20,736 kittens in four years. Michelle Druggar holds the record for largest human family, having given birth to 17 children.",
                "750ml of blood pumps through your brain every minute which is 15-20% of blood flow from the heart.",
                "The human brain is about 75% water.",
                "Dragonflies are capable of flying sixty miles per hour, making them one of the fastest insects. This is good since they are in a big hurry, as they only live about twenty-four hours.",
                "Flies jump backwards during takeoff.",
                "A housefly will regurgitate its item and eat it again.",
                "Termites outweigh humans by almost ten to one.",
                "A spider's web is not a home, but rather a trap for its item. They are as individual as snowflakes, with no two ever being the same. Some tropical spiders have built webs over eighteen feet across.",
                "More people are afraid of spiders than death. Amazingly, few people are afraid of Champagne corks even though you are more likely to be killed by one than by a spider.",
                "Your brain consumes 25 watts of power while you’re awake. This amount of energy is enough to illuminate a lightbulb.",
                "It is impossible to lick your elbow.",
                "Intelligent people have more zinc and copper in their hair.",
                "In every episode of Seinfeld there is a Superman somewhere.",
                "Possums have one of the shortest pregnancies at 16 days. The shortest human pregnancy to produce a healthy baby was 22 weeks, 6 days -- the baby was the length of a ballpoint pen.",
                "Wearing headphones for just an hour will increase the bacteria in your ear by 700 times.",
                "The most poisonous spider is the black widow. Its venom is more potent than a rattlesnake's.",
                "13% of Americans actually believe that some parts of the moon are made of cheese.",
                "The world's youngest parents were 8 and 9 and lived in China in 1910.",
                "Fish that live more than 800 meters below the ocean surface don't have eyes.",
                "Butterflies range in size from a tiny 1/8 inch to a huge almost 12 inches.",
                "Some Case Moth caterpillars (Psychidae) build a case around themselves that they always carry with them. It is made of silk and pieces of plants or soil.",
                "Most household dust is made of dead skin cells.",
                "One in eight million people has progeria, a disease that causes people to grow faster than they age.",
                "The male seahorse carries the eggs until they hatch instead of the female.",
                "Negative emotions such as anxiety and depression can weaken your immune system.",
                "Stephen Hawking was born exactly 300 years after Galileo died.",
                "Mercury is the only planet whose orbit is coplanar with its equator.",
                "The Morgan's Sphinx Moth from Madagascar has a proboscis (tube mouth) that is 12 to 14 inches long to get the nectar from the bottom of a 12 inch deep orchid discovered by Charles Darwin.",
                "Some moths never eat anything as adults because they don't have mouths. They must live on the energy they stored as caterpillars.",
                "In 1958 Entomologist W.G. Bruce published a list of Arthropod references in the Bible. The most frequently named bugs from the Bible are: Locust: 24, Moth: 11, Grasshopper: 10, Scorpion: 10, Caterpillar: 9, and Bee: 4.",
                "People eat insects – called Entomophagy (people eating bugs) – it has been practiced for centuries throughout Africa, Australia, Asia, the Middle East, and North, Central and South America. Why? Because many bugs are both protein-rich and good sources of vitamins, minerals and fats.",
                "Grapes explode when you put them in the microwave. Go on, try it then",
                "Ramses brand condom is named after the great pharaoh Ramses II who fathered over 160 children.",
                "Peanuts are one of the ingredients of dynamite.",
                "The average chocolate bar has 8 insects' legs in it.",
                "In York, it is perfectly legal to shoot a Scotsman with a bow and arrow (except on Sundays)",
                "No piece of square dry paper can be folded in half more than 7 times",
                "The average human eats 8 spiders in their lifetime at night.",
                "The Beetham Tower cost over £150 million to build.",
                "The Beetham Tower has 47 floors.",
                "Stewardesses is the longest word typed with only the left hand.",
                "An average human loses about 200 head hairs per day.",
                "Mexico City sinks about 10 inches a year",
                "It's impossible to sneeze with your eyes open.",
                "In France, a five year old child can buy an alcoholic drink in a bar",
                "During the chariot scene in Ben Hur, a small red car can be seen in the distance.",
                "Because metal was scarce, the Oscars given out during World War II were made of wood.",
                "By raising your legs slowly and lying on your back, you cannot sink into quicksand.",
                "The glue on Israeli postage is certified kosher.",
                "In 10 minutes, a hurricane releases more energy than all of the world's nuclear weapons combined.",
                "On average, 100 people choke to death on ball-point pens every year.",
                "Thirty-five percent of the people who use personal ads for dating are already married.",
                "The electric chair was invented by a dentist.",
                "The top butterfly flight speed is 12 miles per hour. Some moths can fly 25 miles per hour!",
                "The Brimstone butterfly (Gonepterix rhamni) has the longest lifetime of the adult butterflies: 9-10 months.",
                "Bruce Lee was so fast that they actually had to s-l-o-w film down so you could see his moves.",
                "A Boeing 747s wingspan is longer than the Wright brother's first flight.",
                "Representations of butterflies are seen in Egyptian frescoes at Thebes, which are 3,500 years old.",
                "Babies are born without knee caps. They don't appear until the child reaches 2-6 years of age.",
                "14% of all facts and statistics are made up and 27% of people know that fact.",
                "Every time you lick a stamp, you're consuming 1/10 of a calorie.",
                "Eskimos have over 15 words for the English word of 'Snow'",
                "Butterflies can see red, green, and yellow.",
                "Some people say that when the black bands on the Woolybear caterpillar are wide, a cold winter is coming.",
                "Americans on the average eat 18 acres of pizza every day.",
                "Banging your head against a wall uses 150 calories an hour.",
                "Almonds are a member of the peach family.",
                "The plastic things on the end of shoelaces are called aglets.",
                "“Ithyphallophobia is a morbid fear of seeing, thinking about or having an erect penis.",
                "The average shelf-life of a latex condom is about two years.",
                "14% of Americans have skinny-dipped with a member of the opposite sex at least once.",
                "Male bats have the highest rate of homosexuality of any mammal.",
                "A man's beard grows fastest when he anticipates sex.",
                "A man will ejaculate approximately 18 quarts of semen in his lifetime.",
                "Sex is biochemically no different from eating large quantities of chocolate.",
                "Humans and dolphins are the only species that have sex for pleasure.",
                "For every 'normal' webpage, there are five porn pages.",
                "Venus observa is the technical term for the missionary position.",
                "Sex is the safest tranquilizer in the world. IT IS 10 TIMES MORE EFFECTIVE THAN VALIUM.",
                "Samuel Clemens (Mark Twain) was born on and died on days when Halley’s Comet can be seen.",
                "US Dollar bills are made out of cotton and linen.",
                "The 57 on the Heinz ketchup bottle represents the number of pickle types the company once had.",
                "Americans are responsible for about 1/5 of the world’s garbage annually.",
                "Giraffes and rats can last longer without water than camels.",
                "Your stomach produces a new layer of mucus every two weeks so that it doesn’t digest itself.",
                "98% of all murders and rapes are by a close family member or friend of the victim.",
                "A B-25 bomber crashed into the 79th floor of the Empire State Building on July 28, 1945.",
                "The Declaration of Independence was written on hemp (marijuana) paper.",
                "The dot over the letter “i” is called a tittle.",
                "Benjamin Franklin was the fifth in a series of the youngest son of the youngest son.",
                "Triskaidekaphobia means fear of the number 13.",
                "Paraskevidekatriaphobia means fear of Friday the 13th, which occurs one to three times a year.",
                "In Italy, 17 is considered an unlucky number. In Japan, 4 is considered an unlucky number.",
                "A female ferret will die if it goes into heat and cannot find a mate.",
                "In ancient Rome, when a man testified in court he would swear on his testicles.",
                "The ZIP in “ZIP code” means Zoning Improvement Plan.",
                "Coca-Cola contained Coca (whose active ingredient is cocaine) from 1885 to 1903.",
                "A “2 by 4 is really 1 1/2 by 3 1/2.",
                "It’s estimated that at any one time around 0.7% of the world’s population is drunk.",
                "40% of McDonald’s profits come from the sales of Happy Meals.",
                "Every person, including identical twins, has a unique eye & tongue print along with their fingerprint.",
                "The “spot” on the 7-Up logo comes from its inventor who had red eyes. He was an albino.",
                "315 entries in Webster’s 1996 dictionary were misspelled.",
                "The “save” icon in Microsoft Office programs shows a floppy disk with the shutter on backwards.",
                "Albert Einstein and Charles Darwin both married their first cousins",
                "Camel’s have three eyelids.",
                "On average, 12 newborns will be given to the wrong parents every day.",
                "John Wilkes Booth’s brother once saved the life of Abraham Lincoln’s son.",
                "Warren Beatty and Shirley McLaine are brother and sister.",
                "Chocolate can kill dogs; it directly affects their heart and nervous system.",
                "Daniel Boone hated coonskin caps.",
                "55.1% of all US prisoners are in prison for drug offenses.",
                "Most lipstick contains fish scales.",
                "Dr. Seuss pronounced his name “soyce”.",
                "Slugs have four noses.",
                "Ketchup was sold in the 1830s as medicine.",
                "India has a Bill of Rights for cows.",
                "American Airlines saved $40,000 in 1987 by taking out an olive from First Class salads.",
                "About 200,000,000 M&Ms are sold each day in the United States.",
                "Because metal was scarce, the Oscars given out during World War II were made of wood.",
                "There are 318,979,564,000 possible combinations of the first four moves in Chess.",
                "There are no clocks in Las Vegas gambling casinos.",
                "Coconuts kill about 150 people each year. That’s more than sharks.",
                "Half of all bank robberies take place on a Friday.",
                "The name Wendy was made up for the book Peter Pan. There was never a recorded Wendy before it.",
                "The international telephone dialing code for Antarctica is 672.",
                "The first bomb the Allies dropped on Berlin in WWII killed the only elephant in the Berlin Zoo.",
                "The average raindrop falls at 7 miles per hour.",
                "If you put a drop of liquor on a scorpion, it will instantly go mad and sting itself to death.",
                "Bruce Lee was so fast that they had to slow the film down so you could see his moves.",
                "The first CD pressed in the US was Bruce Springsteen’s “Born in the USA”.",
                "IBM’s motto is “Think”. Apple later made their motto “Think different”.",
                "The original name for butterfly was flutterby.",
                "One in fourteen women in America is a natural blonde. Only one in sixteen men is.",
                "The Olympic was the sister ship of the Titanic, and she provided twenty-five years of service.",
                "When the Titanic sank, 2228 people were on it. Only 706 survived.",
                "Every day, 7% of the US eats at McDonald’s.",
                "During his entire life, Vincent Van Gogh sold exactly one painting, “Red Vineyard at Arles”.",
                "By raising your legs slowly and lying on your back, you cannot sink into quicksand.",
                "One in ten people live on an island.",
                "It takes more calories to eat a piece of celery than the celery has in it to begin with.",
                "28% of Africa is classified as wilderness. In North America, its 38%.",
                "Charlie Chaplin once won third prize in a Charlie Chaplin look-alike contest.",
                "Chewing gum while peeling onions will keep you from crying.",
                "Sherlock Holmes NEVER said “Elementary, my dear Watson”",
                "Humphrey Bogart NEVER said “Play it again, Sam” in Casablanca",
                "They NEVER said “Beam me up, Scotty” on Star Trek.",
                "Sharon Stone was the first Star Search spokes model.",
                "More people are afraid of open spaces (kenophobia) than of tight spaces (claustrophobia).",
                "There is a 1 in 4 chance that New York will have a white Christmas.",
                "The Guinness Book of Records holds the record for being the book most often stolen from Libraries.",
                "Thirty-five percent of the people who use personal ads for dating are already married.",
                "$203,000,000 is spent on barbed wire each year in the U.S.",
                "Every US president has worn glasses (just not always in public).",
                "Bats always turn left when exiting a cave.",
                "Jim Henson first coined the word “Muppet”. It is a combination of “marionette” and “puppet.”",
                "The Michelin man is known as Mr. Bib. His name was Bibendum in the company’s first ads in 1896.",
                "The word “lethologica” describes the state of not being able to remember the word you want.",
                "About 14% of injecting drug users are HIV positive.",
                "A word or sentence that is the same front and back (racecar, kayak) is called a “palindrome”.",
                "A snail can sleep for 3 years.",
                "People photocopying their buttocks are the cause of 23% of all photocopier faults worldwide.",
                "China has more English speakers than the United States.",
                "One in every 9000 people is an albino.",
                "There are about a million ants per person. Ants are very social animals and will live in colonies that can contain almost 500,000 ants.",
                "The electric chair was invented by a dentist.",
                "You share your birthday with at least 9 million other people in the world.",
                "Everyday, more money is printed for Monopoly sets than for the U.S. Treasury.",
                "Every year 4 people in the UK die putting their trousers on.",
                "Cats have over one hundred vocal sounds; dogs only have about ten.",
                "Our eyes are always the same size from birth but our nose and ears never stop growing.",
                "In every episode of “Seinfeld” there is a Superman picture or reference somewhere.",
                "Rats multiply so quickly that in 18 months, two rats could have over million descendants.",
                "Wearing headphones for just an hour will increase the bacteria in your ear by 700 times.",
                "Each year in America there are about 300,000 deaths that can be attributed to obesity.",
                "Many butterflies can taste with their feet to find out whether the leaf they sit on is good to lay eggs on to be their caterpillars' item or not.",
                "There are more types of insects in one tropical rain forest tree than there are in the entire state of Vermont.",
                "About 55% of all movies are rated R.",
                "About 500 movies are made in the US and 800 in India annually.",
                "Arabic numerals are not really Arabic; they were created in India.",
                "The February of 1865 is the only month in recorded history not to have a full moon.",
                "There is actually no danger in swimming right after you eat, though it may feel uncomfortable.",
                "The cruise liner Queen Elizabeth II moves only six inches for each gallon of diesel that it burns.",
                "More than 50% of the people in the world have never made or received a telephone call.",
                "A shark is the only fish that can blink with both eyes.",
                "There are about 2 chickens for every human in the world.",
                "The word “maverick” came into use after Samuel Maverick, a Texan refused to brand his cattle.",
                "Two-thirds of the world’s eggplant is grown in New Jersey.",
                "Termites have been known to eat item twice as fast when heavy metal music is playing.",
                "There are more beetles than any other animal. In fact, one out of every four animals is a beetle.",
                "The rhinoceros beetle is the strongest animal and is capable of lifting 850 times its own weight.",
                "On a Canadian two-dollar bill, the American flag is flying over the Parliament Building.",
                "An American urologist bought Napoleon’s penis for $40,000.",
                "No word in the English language rhymes with month, orange, silver, or purple.",
                "Dreamt is the only English word that ends in the letters “MT”.",
                "$283,200 is the absolute highest amount of money you can win on Jeopardy.",
                "Almonds are members of the peach family.",
                "Rats and horses can’t vomit.",
                "The penguin is the only bird that can’t fly but can swim.",
                "There are approximately 100 million acts of sexual intercourse each day.",
                "Winston Churchill was born in a ladies room during a dance.",
                "Maine is the only state whose name is just one syllable.",
                "Americans on average eat 18 acres of pizza every day.",
                "Venus is the only planet that rotates clockwise.",
                "Charlie Chaplin once won third prize in a Charlie Chaplin look-alike contest.",
                "Every time you lick a stamp you consume 1/10 of a calorie.",
                "You are more likely to be killed by a champagne cork than by a poisonous spider.",
                "Hedenophobic means fear of pleasure.",
                "Ancient Egyptian priests would pluck every hair from their bodies.",
                "A crocodile cannot stick its tongue out.",
                "An ant always falls over on its right side when intoxicated.",
                "All polar bears are left-handed.",
                "The catfish has over 27000 taste buds (more than any other animal)",
                "A cockroach will live nine days without its head before it starves to death.",
                "Many insects can carry 50 times their own body weight. This would be like an adult person lifting two heavy cars full of people.",
                "There are over a million described species of insects. Some people estimate there are actually between 15 and 30 million species.",
                "Most insects are beneficial to people because they eat other insects, pollinate crops, are item for other animals, make products we use (like honey and silk) or have medical uses.",
                "Butterflies and insects have their skeletons on the outside of their bodies, called the exoskeleton. This protects the insect and keeps water inside their bodies so they don’t dry out. ",
                "Elephants are the only mammals that cannot jump.",
                "An ostrich’s eye is bigger than its brain.",
                "Starfish have no brains.",
                "11% of the world is left-handed.",
                "Rubber bands last longer when refrigerated.",
                "The national anthem of Greece has 158 verses.",
                "There are 293 ways to make change for a dollar.",
                "A healthy (non-colorblind) human eye can distinguish between 500 shades of gray.",
                "A pregnant goldfish is called a twit.",
                "Lizards can self-amputate their tails for protection. It grows back after a few months.",
                "Los Angeles’ full name is “El Pueblo de Nuestra Senora la Reina de los Angeles de Porciuncula”.",
                "A cat has 32 muscles in each ear.",
                "A honeybee can fly at fifteen miles per hour.",
                "Tigers have striped skin, not just striped fur.",
                "A “jiffy” is the scientific name for 1/100th of a second.",
                "The average child recognizes over 200 company logos by the time he enters first grade.",
                "The youngest pope ever was 11 years old.",
                "The first novel ever written on a typewriter is Tom Sawyer.",
                "A rhinoceros horn is made of compacted hair.",
                "Elwood Edwards did the voice for the AOL sound files (i.e. “You’ve got Mail!”).",
                "A polar bears skin is black. Its fur is actually clear, but like snow it appears white.",
                "Elvis had a twin brother named Garon, who died at birth, which is why Elvis middle name was Aron.",
                "Dueling is legal in Paraguay as long as both parties are registered blood donors.",
                "Donkeys kill more people than plane crashes.",
                "Shakespeare invented the words “assassination” and “bump.”",
                "If you keep a goldfish in the dark room, it will eventually turn white.",
                "Women blink nearly twice as much as men.",
                "The name Jeep comes from “GP”, the army abbreviation for General Purpose.",
                "Right handed people live, on average, nine years longer than left handed people do.",
                "There are two credit cards for every person in the United States.",
                "Cats’ urine glows under a black light.",
                "A “quidnunc” is a person who is eager to know the latest news and gossip.",
                "Leonardo Da Vinci invented the scissors, the helicopter, and many other present day items.",
                "In the last 4000 years no new animals have been domesticated.",
                "25% of a human’s bones are in its feet.",
                "On average, 100 people choke to death on ballpoint pens every year.",
                "“Canada” is an Indian word meaning “Big Village”.",
                "Only one in two billion people will live to be 116 or older.",
                "Rape is reported every six minutes in the U.S.",
                "The human heart creates enough pressure in the bloodstream to squirt blood 30 feet.",
                "A jellyfish is 95% water.",
                "The world's longest snake (by reliable documentation) is the reticulated python, with a maximum length of, perhaps, 30 feet.",
                "Common Cobra venom is not on the list of top 10 venoms yet it is still 40 times more toxic than cyanide.",
                "The venom of the Australian Brown Snake is so powerful only 1/14,000th of an ounce is enough to kill a human.",
                "Truck driving is the most dangerous occupation by accidental deaths (799 in 2001).",
                "Banging your head against a wall uses 150 calories an hour.",
                "Elephants only sleep for two hours each day.",
                "On average people fear spiders more than they do death.",
                "The strongest muscle in the human body is the tongue. (the heart is not a muscle)",
                "In golf, a ‘Bo Derek’ is a score of 10.",
                "In the U.S, Frisbees outsell footballs, baseballs and basketballs combined.",
                "In most watch advertisements the time displayed on a watch is 10:10.",
                "If you plant an apple seed, it is almost guaranteed to grow a tree of a different type of apple.",
                "Al Capone’s business card said he was a used furniture dealer.",
                "The only real person to be a PEZ head was Betsy Ross.",
                "There are about 450 types of cheese in the world. 240 come from France.",
                "A dragonfly has a lifespan of 24 hours.",
                "In Iceland, a Big Mac costs $5.50.",
                "Broccoli and cauliflower are the only vegetables that are flowers.",
                "There is no solid proof of who built the Taj Mahal.",
                "In a survey of 200000 ostriches over 80 years, not one tried to bury its head in the sand.",
                "A dime has 118 ridges around the edge. A quarter has 119.",
                "”Judge Judy” has a $25,000,000 salary, while Supreme Court Justice Ginsberg has a $190,100 salary.",
                "Andorra, a tiny country between France & Spain, has the longest average lifespan: 83.49 years.",
                "Mr. Rogers was an ordained Presbyterian minister.",
                "In America you will see an average of 500 advertisements a day.",
                "John Lennon’s first girlfriend was named Thelma Pickles.",
                "You can lead a cow upstairs but not downstairs.",
                "The average person falls asleep in seven minutes.",
                "“The sixth sick sheik’s sixth sheep’s sick” is said to be the toughest tongue twister in English.",
                "There are 336 dimples on a regulation US golf ball. In the UK its 330.",
                "“Duff” is the decaying organic matter found on a forest floor.",
                "The US has more personal computers than the next 7 countries combined.",
                "Kuwait is about 60% male (highest in the world). Latvia is about 54% female (highest in the world).",
                "The Hawaiian alphabet has only 12 letters.",
                "In 10 minutes, a hurricane releases more energy than all the world’s nuclear weapons combined.",
                "At the height of its power in 400 BC, the Greek city of Sparta had 25,000 citizens and 500,000 slaves.",
                "Julius Caesar’s autograph is worth about $2,000,000.",
                "People say “bless you” when you sneeze because your heart stops for a millisecond.",
                "US gold coins used to say “In Gold We Trust”.",
                "In “Silence of the Lambs”, Hannibal Lector (Anthony Hopkins) never blinks.",
                "A shrimp’s heart is in its head.",
                "In the 17th century, the value of pi was known to 35 decimal places. Today, to 1.2411 trillion.",
                "Pearls melt in vinegar.",
                "“Lassie” was played by a group of male dogs; the main one was named Pal.",
                "Nepal is the only country that doesn’t have a rectangular flag.",
                "Switzerland is the only country with a square flag.",
                "Antarctica is the only continent on which no Lepidoptera have been found.",
                "There are about 24,000 species of butterflies. The moths are even more numerous: about 140,000 species of them were counted all over the world.",
                "Gabriel, Michael, and Lucifer are the only angels named in the Bible.",
                "Johnny Appleseed planted apples so that people could use apple cider to make alcohol.",
                "Abraham Lincoln’s ghost is said to haunt the White House.",
                "God is not mentioned once in the book of Esther.",
                "The odds of being born male are about 51.2%, according to census.",
                "Scotland has more redheads than any other part of the world.",
                "There is an average of 61,000 people airborne over the US at any given moment.",
                "Prince Charles and Prince William never travel on the same airplane in case there is a crash.",
                "The most popular first name in the world is Muhammad.",
                "The surface of the Earth is about 60% water and 10% ice.",
                "For every 230 cars that are made, 1 will be stolen.",
                "Jimmy Carter was the first U.S. President to be born in a hospital.",
                "Lightning strikes the earth about 8 million times a day.",
                "Humans use a total of 72 different muscles in speech.",
                "If you feed a seagull Alka-Seltzer, its stomach will explode.",
                "Only female mosquitoes bite.",
                "The U.S. Post Office handles 43 percent of the world’s mail.",
                "Venus and Uranus are the only planets that rotate opposite to the direction of their orbit.",
                "John Adams, Thomas Jefferson, and James Monroe died on July 4th.",
                "Baby Ruth candy bar was named after Grover Cleveland’s daughter, Ruth, not the baseball player.",
                "Dolphins can look in different directions with each eye. They can sleep with one eye open.",
                "The Falkland Isles (pop. about 2000) has over 700000 sheep (350 per person).",
                "There are 41,806 different spoken languages in the world today.",
                "The city of Venice stands on about 120 small islands.",
                "The past-tense of the English word “dare” is “durst”",
                "Beetles taste like apples, wasps like pine nuts, and worms like fried bacon.",
                "Of all the words in the English language, the word 'set' has the most definitions!",
                "What is called a French kiss in the English speaking world is known as an English kiss in France.",
                "Almost is the longest word in the English language with all the letters in alphabetical order.",
                "Rhythm is the longest English word without a vowel.",
                "In 1386, a pig in France was executed by public hanging for the murder of a child",
                "A cockroach can live several weeks with its head cut off!",
                "Human thigh bones are stronger than concrete.",
                "You can't kill yourself by holding your breath",
                "There is a city called Rome on every continent.",
                "Your heart beats over 100,000 times a day!",
                "The skeleton of Jeremy Bentham is present at all important meetings of the University of London",
                "Right handed people live, on average, nine years longer than left-handed people",
                "Your ribs move about 5 million times a year, every time you breathe!",
                "One quarter of the bones in your body, are in your feet!",
                "Like fingerprints, everyone's tongue print is different!",
                "Fingernails grow nearly 4 times faster than toenails!",
                "Most dust particles in your house are made from dead skin!",
                "Present population of 5 billion plus people of the world is predicted to become 15 billion by 2080.",
                "Women blink nearly twice as much as men.",
                "Adolf Hitler was a vegetarian, and had only ONE testicle.",
                "Honey is the only item that does not spoil.",
                "Months that begin on a Sunday will always have a Friday the 13th.",
                "Coca-Cola would be green if coloring weren’t added to it.",
                "On average a hedgehog's heart beats 300 times a minute.",
                "More people are killed each year from bees than from snakes.",
                "The average lead pencil will draw a line 35 miles long or write approximately 50,000 English words.",
                "More people are allergic to cow's milk than any other item.",
                "Camels have three eyelids to protect themselves from blowing sand.",
                "The placement of a donkey's eyes in it’s' heads enables it to see all four feet at all times!",
                "The six official languages of the U.N. are: English, French, Arabic, Chinese, Russian and Spanish.",
                "Earth is the only planet not named after a god.",
                "It's against the law to burp, or sneeze in a church in Nebraska, USA.",
                "You're born with 300 bones, but by the time you become an adult, you only have 206.",
                "Some worms will eat themselves if they can't find any item!",
                "The world’s oldest piece of chewing gum is 9000 years old!",
                "The longest recorded flight of a chicken is 13 seconds",
                "Owls are the only birds that can see the color blue.",
                "A man named Charles Osborne had the hiccups for 69 years!",
                "A giraffe can clean its ears with its 21-inch tongue!",
                "The average person laughs 10 times a day!",
                "The Bible, the world's best-selling book, is also the world's most shoplifted book.",
                "Someone paid $14,000 for the bra worn by Marilyn Monroe in the film 'Some Like It Hot'.",
                "Your tongue is the only muscle in your body that is attached at only one end.",
                "More than 1,000 different languages are spoken on the continent of Africa.",
                "Buckingham Palace in England has over six hundred rooms.",
                "There was once an undersea post office in the Bahamas.",
                "Ninety percent of New York City cabbies are recently arrived immigrants.",
                "It's possible to lead a cow upstairs...but not downstairs.",
                "A snail can sleep for three years. ",
                "No word in the English language rhymes with MONTH.",
                "Average life span of a major league baseball: 7 pitches.",
                "Our eyes are always the same size from birth, but our nose and ears never stop growing.",
                "Go. is the shortest complete sentence in the English language.",
                "The pound key on your keyboard () is called an octotroph. ",
                "The only domestic animal not mentioned in the Bible is the cat. ",
                "Table tennis balls have been known to travel off the paddle at speeds up to 160 km/hr. ",
                "Pepsi originally contained pepsin, thus the name. ",
                "The original story from Tales of 1001 Arabian Nights begins, Aladdin was a little Chinese boy",
                "Nutmeg is extremely poisonous if injected intravenously. ",
                "Honey is the only natural item that is made without destroying any kind of life.",
                "The volume of the earth's moon is the same as the volume of the Pacific Ocean. ",
                "Cephalacaudal recapitulation is the reason our extremities develop faster than the rest of us. ",
                "Chinese Crested dogs can get acne. ",
                "Each year there is one ton of cement poured for each man woman and child in the world. ",
                "The house fly hums in the middle octave key of F. ",
                "The only capital letter in the Roman alphabet with exactly one end point is P. ",
                "The giant red star Betelgeuse has a diameter larger than that of the Earth's orbit around the sun. ",
                "Hummingbirds are the only animals that can fly backwards. ",
                "A cat's jaw cannot move sideways.",
                "The human heart creates enough pressure when it pumps out to the body to squirt blood 30 feet.",
                "The flea can jump 350 times its body length. It's like a human jumping the length of a football field.",
                "Some lions mate over 50 times a day.",
                "Rubber bands last longer when refrigerated. ",
                "The average person's left hand does 56% of the typing. ",
                "The longest one-syllable word in the English language is screeched.",
                "All of the clocks in the movie Pulp Fiction are stuck on 4:20. ",
                "Dreamt is the only English word that ends in the letters mt.",
                "Maine is the only state (in USA) whose name is just one syllable. ",
                "The giant squid has the largest eyes in the world. ",
                "In England, the Speaker of the House is not allowed to speak. ",
                "Mr. Rogers was an ordained minister. ",
                "A rat can last longer without water than a camel.",
                "Your stomach has to produce a new layer of mucus every two weeks or it will digest itself.",
                "A female ferret will die if it goes into heat and cannot find a mate.",
                "A 2 X 4 is really 1-1/2 by 3-1/2",
                "On average, 12 newborns will be given to the wrong parents daily.",
                "There are no words in the dictionary that rhyme with orange, purple, silver and month.",
                "The caterpillars of some Snout Moths (Pyralididae) live in or on water-plants.",
                "The females of some moth species lack wings, all they can do to move is crawl.",
                "If one places a tiny amount of liquor on a scorpion, it will instantly go mad and sting itself to death.",
                "The first CD pressed in the US was Bruce Springsteen's Born in the USA",
                "Sherlock Holmes NEVER said, Elementary, my dear Watson.",
                "California consumes more bottled water than any other product.",
                "California has issued 6 drivers licenses to people named Jesus Christ.",
                "In 1980, a Las Vegas hospital suspended workers for betting on when patients would die.",
                "Nevada is the driest state in the U.S.. Each year it averages 7.5 inches (19 cm) of rain.",
                "In Utah, it is illegal to swear in front of a dead person.",
                "Salt Lake City, Utah has a law against carrying an unwrapped ukulele on the street.",
                "Arizona was the last of the 48 adjoining continental states to enter the Union.",
                "It is illegal to hunt camels in the state of Arizona.",
                "Wyoming was the first state to give women the right to vote in 1869.",
                "Denver, Colorado lays claim to the invention of the cheeseburger.",
                "The first license plate on a car in the United States was issued in Denver, Colorado in 1908.",
                "The state of Maryland has no natural Lakes.",
                "Illinois has the highest number of personalized license plates than any other state.",
                "Residents of Houston, Texas lead the U.S. in eating out - approximately 4.6 times per week.",
                "Laredo, Texas is the U.S.'s farthest inland port.",
                "Rugby, North Dakota is the geographical center of North America.",
                "Butte County, South Dakota is the geographical center of the U.S.",
                "Louisiana's capital building is the tallest one of any U.S. state.",
                "Hawaii is the only coffee producing state.",
                "One in seven workers in Boston, Massachusetts walks to work.",
                "The Dull Men's Hall of Fame is located in Carroll, Wisconsin.",
                "Gary, Indiana is the murder capital of the U.S. - probably the world.",
                "Alabama was the first state to recognize Christmas as an official holiday.",
                "The largest NFL stadium is the Pontiac Silverdome in Detroit, Michigan.",
                "Michigan was the first state to have roadside picnic tables.",
                "No matter where you stand in Michigan, you are never more than 85 miles from a Great Lake.",
                "The official beverage of Ohio is tomato juice.",
                "Georgia's state motto is Wisdom, Justice and Moderation.",
                "The U.S. city with the highest rate of lightning strikes per capita is Clearwater, Florida.",
                "It's illegal to spit on the sidewalk in Norfolk, Virginia.",
                "The first streetlights in America were installed in Philadelphia around 1757.",
                "The highest point in Pennsylvania is lower than the lowest point in Colorado.",
                "If you were to take a taxicab from New York City to Los Angeles, it would cost you $8,325.",
                "The NY phone book had 22 Hitlers before WWII. The NY phone book had 0 Hitlers after WWII.",
                "In New York State, it is illegal to but any alcohol on Sundays before noon.",
                "There were 240 pedestrian fatalities in New York City in 1994.",
                "Columbia University is the second largest landowner in New York City, after the Catholic Church.",
                "Montpelier, Vermont is the only state capital without a McDonalds.",
                "Maine is the only state that has borders with only one other state.",
                "The first McDonald's restaurant in Canada was in Richmond, British Columbia.",
                "In 1984, a Canadian farmer began renting advertising space on his cows.",
                "There are more donut shops in Canada per capita than any other country.",
                "0.3% of all road accidents in Canada involve a Moose.",
                "In the great fire of London in 1666 half of London was burnt down but only 6 people were injured.",
                "In Quebec, there is an old law that states margarine must be a different color than butter.",
                "The largest taxi fleet in the world is found in Mexico City. The city boasts a fleet of over 60,000 taxis.",
                "More than 90% of the Nicaraguan people are Roman Catholic.",
                "Cuba is the only island in the Caribbean to have a railroad.",
                "Jamaica has the most churches per square mile than any other country in the world.",
                "The angel falls in Venezuela are nearly 20 times taller than Niagara Falls.",
                "Canada is the only country not to win a gold medal in the summer Olympic games while hosting.",
                "The Amazon is the world's largest river, 3,890 miles (6,259 km) long.",
                "The town of Calma, Chile in the Atacama Desert has never had rain.",
                "The people of France eat more cheese than any other country in the world.",
                "King Louis XIX ruled France for 15 minutes.",
                "The most common name in Italy is Mario Rossi.",
                "Greece's national anthem has 158 verses.",
                "In ancient Greece idiot meant a private citizen or layman.",
                "Bulgarians are known to be the biggest yogurt eaters in the world.",
                "Czechs are the biggest consumers of beer per male in the world.",
                "A Czech man, Jan Honza Zampa, holds the record for drinking one liter of beer in 4.11 seconds.",
                "Netherlands is the only country with a national dog.",
                "When we think of Big Ben in London, we think of the clock. Actually, it's the bell.",
                "The Automated Teller Machine (ATM) was introduced in England in 1965.",
                "Buckingham Palace has 602 rooms.",
                "Icelanders consume more Coca-Cola per Capita than any other nation.",
                "Until 1997, there were more pigs than people in Denmark.",
                "There is a hotel in Sweden built entirely out of ice; it is rebuilt every year.",
                "Sweden has the least number of murders annually.",
                "Lithuania has the highest suicide rate in the world.",
                "The country code for Russia is 007",
                "Russians generally answer the phone by saying,I'm listening",
                "The U.S. bought Alaska for 2 cents an acre from Russia.",
                "1 in 5 of the world's doctors are Russian.",
                "Antarctica is the only continent that does not have land areas below sea level.",
                "The people of Israel consume more turkeys per capita than any other country.",
                "Nepal is the only country that has a non-rectangular flag. It is also asymmetrical.",
                "1,800 cigarettes are smoked per person each year in China.",
                "Respiratory Disease is China's leading cause of death.",
                "There are more than 40,000 characters in the Chinese script.",
                "More people speak English in China than the United States.",
                "The toothbrush was invented in China in 1498.",
                "Mongolia is the largest landlocked country.",
                "Vatican City is the smallest country in the world, with a population of 1000 and just 108.7 acres.",
                "In Japan, watermelons are squared. It's easier to stack them that way.",
                "98% of Japanese are cremated.",
                "The number four is considered unlucky in Japan because it is pronounced the same as death",
                "The average Japanese household watches more than 10 hours of television a day.",
                "The Philippines has about 7,100 islands, of which only about 460 are more than 1 square mile in area.",
                "Yo-yos were used as weapons by warriors in the Philippines in the 16th century.",
                "Australian soldiers used the song We're Off to See the Wizard as a marching song in WWII.",
                "The Australian $5 to $100 notes are made of plastic.",
                "The Nullarbor Plain of Australia covers 100,000 square miles (160,900 km) without a tree.",
                "Tasmania, Australia has the cleanest air in the inhabited world.",
                "Greenland is the largest island in the world.",
                "The first female guest host of Saturday Night Live was Candace Bergen.",
                "In 1933, Mickey Mouse, an animated cartoon character, received 800,000 fan letters.",
                "The Simpsons is the longest running animated series on TV.",
                "The first toilet ever seen on television was on Leave It to Beaver.",
                "In every episode of Seinfeld there is a Superman somewhere.",
                "The average human brain has about 100 billion nerve cells.",
                "Nerve impulses to and from the brain travel as fast as 170 miles (274 km) per hour.",
                "The thyroid cartilage is more commonly known as the adams apple.",
                "Your stomach needs to produce a new layer of mucus every two weeks or it would digest itself.",
                "The average life of a taste bud is 10 days.",
                "The average cough comes out of your mouth at 60 miles (96.5 km) per hour.",
                "Relative to size, the strongest muscle in the body is the tongue.",
                "When you sneeze, all your bodily functions stop even your heart.",
                "Babies are born without knee caps. They don't appear until the child reaches 2-6 years of age.",
                "Right handed people live, on average, nine years longer than left handed people do.",
                "Children grow faster in the springtime.",
                "It takes the stomach an hour to break down cows’ milk.",
                "Women blink nearly twice as much as men.",
                "Blondes have more hair than dark-haired people do.",
                "There are 10 human body parts that are only 3 letters long (eye hip arm leg ear toe jaw rib lip gum).",
                "If you go blind in one eye you only lose about one fifth of your vision but all your sense of depth.",
                "The average human head weighs about 8 pounds.",
                "In the average lifetime, a person will walk the equivalent of 5 times around the equator.",
                "An average human scalp has 100,000 hairs.",
                "The average human blinks their eyes 6,205,000 times each year.",
                "Your skull is made up of 29 different bones.",
                "Ancient Egyptians shaved off their eyebrows to mourn the deaths of their cats.",
                "Hair is made from the same substance as fingernails.",
                "The surface of the human skin is 6.5 square feet (2m).",
                "15 million blood cells are destroyed in the human body every second.",
                "The pancreas produces Insulin.",
                "The most sensitive cluster of nerves is at the base of the spine.",
                "The human body is comprised of 80% water.",
                "The average human will shed 40 pounds of skin in a lifetime.",
                "Human thighbones are stronger than concrete.",
                "There are 45 miles of nerves in the skin of a human being.",
                "Canadian researchers have found that Einstein's brain was 15% wider than normal.",
                "While in Alcatraz, Al Capone was inmate 85.",
                "Astronaut Neil Armstrong first stepped on the moon with his left foot.",
                "Jim Morrison, of the 60's rock group The Doors, was the first rock star to be arrested on stage.",
                "Frank Lloyd Wright's son invented Lincoln Logs.",
                "Peter Falk, who played Columbo, has a glass eye.",
                "Barbie's full name is Babara Millicent Roberts.",
                "The mother of Michael Nesmith of The Monkees invented whiteout.",
                "Isaac Asimov is the only author to have a book in every Dewey-decimal category.",
                "Shakespeare invented the word assassination and bump.",
                "It is believed that Leonardo Da Vinci invented the scissors.",
                "Adolf Hitler's mother seriously considered having an abortion but was talked out of it by her doctor.",
                "The shortest British monarch was Charles I, who was 4 feet 9 inches.",
                "Tina Turner's real name is Annie Mae Bullock.",
                "Beethoven dipped his head in cold water before he composed.",
                "President John F Kennedy could read 4 newspapers in 20 minutes.",
                "Bob Dylan's real name is Robert Zimmerman.",
                "Sigmund Freud had a morbid fear of ferns.",
                "Anne Boleyn, Queen Elizabeth I's mother, had six fingers on one hand.",
                "Orville Wright was involved in the first aircraft accident. His passenger, a Frenchman, was killed.",
                "The sound of E.T. walking was made by someone squishing her hands in jelly.",
                "Cher's last name was Sarkissian. She changed it because no one could pronounce it.",
                "Sugar was first added to chewing gum in 1869 by a dentist, William Semple.",
                "Paper was invented early in the second century by Chinese eunuch.",
                "Sir Isaac Newton was only 23 years old when he discovered the law of universal gravitation.",
                "Hannibal had only one eye after getting a disease while attacking Rome.",
                "A blue whales heart only beats nine times per minute.",
                "A cat uses its whiskers to determine if a space is too small to squeeze through.",
                "A chameleon's tongue is twice the length of its body.",
                "A crocodiles tongue is attached to the roof of its mouth.",
                "Rodent's teeth never stop growing.",
                "A shark can detect one part of blood in 100 million parts of water.",
                "The penguin is the only bird that can swim but can't fly.",
                "The cheetah is the only cat that can't retract its claws.",
                "A lion's roar can be heard from five miles away.",
                "Emus and kangaroos can't walk backwards.",
                "Cats have over 100 vocal sounds; dogs only have 10.",
                "A mole can dig a tunnel 300 feet (91 m) long in just one night.",
                "Insects outnumber humans 100,000,000 to one.",
                "Sharkskin has tiny tooth-like scales all over.",
                "Chameleons can move their eyes in two directions at the same time.",
                "Koalas never drink water. They get fluids from the eucalyptus leaves they eat.",
                "A cow gives nearly 200,000 glasses of milk in her lifetime.",
                "When sharks take a bite, their eyes roll back and their teeth jut out.",
                "Camels chew in a figure 8 pattern.",
                "Proportional to their size, cats have the largest eyes of all mammals.",
                "Sailfish can leap out of the water and into the air at a speed of 50 miles (81 km) per hour.",
                "The catfish has the most taste buds of all animals, having over 27,000 of them.",
                "A skunk's smell can be detected by a human a mile away.",
                "A lion in the wild usually makes no more than 20 kills a year.",
                "In space, astronauts cannot cry, because there is no gravity, so the tears can't flow.",
                "The state of Florida is bigger than England.",
                "One in every 4 Americans has appeared on television.",
                "The average American/Canadian will eat about 11.9 pounds of cereal per year!",
                "There are over 58 million dogs in the US",
                "Dogs and cats consume over $11 billion worth of pet item a year",
                "Baby robins eat 14 feet of earthworms every day",
                "In Raiders of the Lost Ark there is a wall carving of R2-D2 and C-3P0 behind the ark",
                "I is the most spoken word in the English language",
                "You is the second most spoken English word",
                "Spain leads the world in cork production",
                "There are 1,792 steps in the Eiffel Tower",
                "There is a city in Norway called Hell",
                "The human feet perspire half a pint of fluid a day",
                "An Olympic gold medal must contain 92.5 percent silver",
                "There are 240 dots on an arcade Pac-Man game",
                "The San Francisco Cable cars are the only mobile National Monuments",
                "Lee Harvey Oswald's cadaver tag sold at an auction for $6,600 in 1992.",
                "A pound of houseflies contains more protein than a pound of beef",
                "The average American works 24,000 hours in their lifetime just to pay their taxes",
                "40% of all people who come to a party in your home snoop in your medicine cabinet",
                "A duck's quack doesn't echo, and no one knows why.",
                "Non-dairy creamer is flammable.",
                "Pinocchio is Italian for pine head.",
                "There are more than 10 million bricks in the Empire State Building.",
                "Rubber bands will last much longer when they are refrigerated.",
                "When a rubber band is placed in the fridge, it causes the polymers to relax. This keeps the band from breaking down as fast as it normally does.",
                "There are 293 ways to make change for a dollar.",
                "This includes change in dimes, quarters, and combinations of the two. Can you figure out all 293 combinations?",
                "The Grand Theft Auto franchise has lawsuits that total over $1 billion.",
                "There are a lot of controversies that surround the game due to the nature of its gameplay. Rockstar North has faced many legal claims of copyright and influencing young players to commit sexual and illegal acts. ",
                "READ ALSO: 18 October Fun Facts Embracing Autumn",
                "All clocks in Pulp Fiction are set to 4:20.",
                "Some people state that there are two scenes where the clocks are not set to this. Watch the movie to figure out which scenes they were.",
                "The eye of an ostrich is bigger than its brain.",
                "Its eyes are around the size of a billiard ball. One eye is also smaller than the other. Perhaps this is why they tend to run in circles.",
                "A dime has 118 ridges on its edge.",
                "The ridges allow the coin to determine if it is real or fake. This was implemented on all coins before the 18th century. The ridges also make it harder to make counterfeit coins.",
                "On average, a secretary will use its left hand for 56% of what they type on a keyboard.",
                "This is because most of the most common letters in the English language are on the left side of the keyboard. The right side of the keyboard only contains i and n as the most common letters. ",
                "The largest pair of eyes in the world belongs to the giant squid.",
                "Its eyes are the size of soccer balls and are at least 25 centimeters across. The largest fish eye is only around 9 centimeters wide which belongs to the swordfish. ",
                "The Pokemon Rhydon was the first to ever be created.",
                "According to the lead video game designer, Ken Sugimori, Rhydon was the first-ever Pokemon to be created by the team. No, it was not Bulbasaur.",
                "READ ALSO: 15 Interesting Facts About Marble",
                "Super Mario Land was the most popular game on the Game Boy during its release.",
                "It is also the first platform where the Mario games had released. To this day, Super Mario Land continues to be one of the highest-ranked games loved by retro gamers.",
                "The dot over the small letter 'i' is called a tittle.",
                "This dot is an integral part of the lowercase i and j. These dots also appear over the letters in various languages.",
                "Japan has 23 vending machines per person.",
                "This ratio is the same per capita. Japan has the highest amount of vending machines in the world. ",
                "Soccer balls were once used for playing basketball.",
                "The first basketballs were not produced until 1894. This means that for three years upon the invention of the game, people were using non-regulated balls.",
                "A candle’s flame is hot and blue in zero gravity.",
                "Diffusion feeds the flame with oxygen. This then allows carbon dioxide to move away from the point of combustion.",
                "Putting sugar on a cut will make it heal faster.",
                "Pour some sugar on top of the wound and wrap it with a bandage. The granules of the sugar crystals will absorb any moisture that bacteria thrive on. ",
                "X-rays can’t detect real diamonds.",
                "The reason for this is because the x-ray cannot penetrate or identify the materials in the diamond. ",
                "There are 7 different types of twins.",
                "Apart from the well known identical and fraternal types of twins, there are 5 more. These include half-identical, mirror image, mixed, chromosome, and superfecundation, and superfetation.",
                "The national flag of Libya was formerly just the color green.",
                "Through the years of 1977 to 2011, its national was a single color. There were no other designs added to the flag. ",
                "The plastic tips of shoelaces are called aglets.",
                "These tips can also be made of metal. Its purpose is to help make the lace easier to hold when running through the holes of the shoes.",
                "Sign language has tongue twisters.",
                "They are called finger fumblers. Many who have practiced sign language over the years still fumble over certain sequences in ASL.",
                "Penguins fly underwater.",
                "Rather, they swim so well underwater that it seems as if they are flying. Penguins can swim up to speeds of 25 mph. ",
                "READ ALSO: 3 Essential Types of Insurance You Have To Know",
                "Minnie the Mouse’s first name is not Minnie.",
                "It is Minerva. Minnie Mouse is a nickname that was given to the character by Ub Iwerks and Walt Disney. Minnie’s actual name is rarely used.",
                "Rudolph the Reindeer is female.",
                "This can be observed through their antlers. Female deer shed their antlers in the spring and grow them back into full size by winter. The male reindeer stops its growth during the winter.",
                "A jiffy is a proper unit of time.",
                "It is exactly 1/100th of a second. This is slower than a Planck which is sextillion times faster.",
                "April 11, 1954, was recorded as the most boring day in the world.",
                "Statistics show that no significant occurrences took place in the world. This was calculated by a computer search program. This program could calculate the number of important events that occur all over the world simultaneously.",
                "Tiramisu translates to ‘take me to heaven’ in Italian.",
                "This implies that this used_item_groups is so good that it would take you to heaven. This Italian dessert is well-loved all over the world and has several alternative twists you can create. ",
                "Buttermilk does not contain any butter.",
                "The butter in its name refers to the origins of the drink. Now that’s a confusing random fact you probably didn’t know. ",
                "READ ALSO: 100 Interesting Facts That Will Boggle Your Mind",
                "Brunch was invented as a way of curing hangovers.",
                "This meal would be enjoyed as a late breakfast that leans more towards fatty items. This came from the belief that alcohol causes cravings for greasy item to increase. ",
                "Hitler’s nephew betrayed him.",
                "William Hitler attempted to blackmail his uncle with threats regarding his paternal grandfather was a Jewish merchant. After this failed, he fled Germany and wrote an article for Look magazine titled ‘Why I Hate My Uncle’. It was after this that William joined the US navy.",
                "The continental plates move at the same rate that fingernails grow.",
                "Research suggests that in some time in the far future, a supercontinent may form. The continents are continuously moving a few centimeters a year.",
                "Sailors working for the Royal Navy need special permission to grow their beards.",
                "Once this is approved, they are given two weeks to grow a full set before presenting himself to a Master at Arms. This person then decides if the beard looks presentable enough to keep.",
                "There are fewer stars than there are trees on Earth.",
                "According to statistics, there are around 3 trillion trees on the planet and only about 400 billion stars in the Milky Way. ",
                "Mary and James are the most popular names around the world.",
                "Between the years of 1917 to 2016, over 5 million baby boys were named James. On the other hand, over 3.5 million baby girls were named Mary. ",
                "READ ALSO: 5 Facts About Succeeding in Business",
                "Children are born less frequently on Saturdays.",
                "Among all the days in the week, most children in the world are born on a Thursday. Babies born on weekends in December is also the least common.",
                "Danish mothers are known to be the most hardworking moms in the world.",
                "According to statistics, 82% of mothers in Denmark are employed. Meanwhile, neighboring countries like Sweden and the Netherlands only rate at 50% working mothers.",
                "75 burgers are sold in McDonald’s every second.",
                "You can view how many burgers have been sold in real-time on the McDonald’s website. You can also track the number of other products that have also been sold from all over the world. ",
                "1,700 people become millionaires every day in the U.S.",
                "The United States has over 8 million families who have a yearly income of over $1 million. ",
                "You can’t hum while holding your nose.",
                "I bet you just tried it, didn’t you?",
                "You are more likely to have a weird or scary dream when sleeping on your stomach.",
                "This is because different sleeping positions give different pressure on your body. Sleeping on your stomach restricts other movements compared to sleeping on your side or back.",
                "Research believes that this may be why the intensity of your dreams differentiate depending on your sleeping position.",
                "Your eyeballs do not grow or change their size as you age.",
                "Generally, only the vertical measure changes, but only by a small amount. By the time we reach 20–21 years old, our eyes will be at their permanent state.",
                "Blue-eyed people have higher alcohol tolerance.",
                "Research in 2000 found that those with lighter eye colors are less likely to abuse alcohol. Thus lesser consumption and developing a higher tolerance. ",
                "Pubic hair lives for about 3 weeks long.",
                "Pubic hair also indicates if the human body is ready for reproduction. It also protects the genitals area during reproduction. ",
                "Male bees can only mate once.",
                "After mating with a female, the male bee’s endophallus is removed. Its abdomen also rips open and results in the male bee’s death. Now that’s what we call a bad date.",
                "Smelling green apples help with weight loss.",
                "A research stated that the smell can help curb your hunger. The neutral sweet scent that comes from green apples and bananas is enough to temporarily forget about hunger.",
                "A snail has 2,500 teeth.",
                "These teeth can be found on their tongue that’s covered in ridges. Snails eat by rubbing their tongue on its item while the ridges cut it into tiny pieces. ",
                "snail random facts",
                "Source: Pexels",
                "You can die from staying up for two weeks straight.",
                "Sleep deprivation can cause your mental and motor responses to become unstable. Thus causing a higher risk to your safety. ",
                "Pigeons can't fart.",
                "Farts are caused by a noticeable number of eruptions from intestinal gas. However, for a bird, their intestine is short which causes them to get rid of waste more frequently. ",
                "Space partly smells like diesel fuel and barbeque.",
                "This is mainly due to the amount of dying stars in our galaxy. The combustion releases a compound called polycyclic aromatic hydrocarbons. ",
                "One strand of hair can hold up to 3 ounces of weight.",
                "The average person’s head contains about 100,000 strands of hair. If you do the math, your hair can support up to 12 tons worth of weight.",
                "Watching horror movies before viewing abstract art will enhance the experience.",
                "This comes from a recent study that proved our reaction towards abstract images improve when frightened. This is especially so for those who are not big fans of the movie genre.",
                "Children’s book author, Roald Dahl was a spy.",
                "To be more precise, he was an agent for the British Security Coordination. He was tasked to gather intel during the second world war. ",
                "NASCAR drivers lose weight while racing.",
                "Because of the high temperatures during a race, the average driver can lose up to 10 pounds worth of weight by sweating. A racecar regularly reaches temperatures of 170 degrees despite the built-in ventilation. ",
                "Indians read the most in the world.",
                "On average, they spend 10 hours more of their time during the week just reading. In the age of digital media, this country still prefers an old fashioned book rather than their phones.",
                "Cap'n Crunch was once sued for not using real berries.",
                "An American woman by the name of Jeanine Sugawara was shocked to find out the cereal was falsely advertising its contents. However, the complaint was quickly dismissed when the judge stated that there is no such thing as Crunch Berry.",
                "Cap'n Crunch's full name is Horatio Magellan Crunch.",
                "He was named after the famous explorer Ferdinand Magellan. His ship is also called the S.S. Guppy.",
                "The most widely printed book in the world is the catalog for IKEA.",
                "IKEA has over 200 million copies of its catalog circulated annually. This surpasses the amount of printing the Christian bible has. ",
                "Crocodiles are one of the planet’s oldest living creatures.",
                "These animals have survived for over 200 million years. This may be due to their superb ability to be able to go for long periods without eating. ",
                "The Aurora Borealis has a sister phenomenon.",
                "This can be viewed in the southern hemisphere. It is also called the Aurora Australis and the light show it executes is as beautiful. ",
                "The salty taste of bacon isn’t natural.",
                "The salty flavor that we all love comes from the curing and brining process. After the meat is prepared, it is flavored and preserved.",
                "There was a fifth member of the Beatles.",
                "Stuart Sutcliffe was a painter and bassist who was an original member before the band came into fame. He soon later died due to a brain hemorrhage. ",
                "Apple once had a clothing line.",
                "They mainly consisted of graphic tees with cheesy designs on them. Ironically, they would’ve made a bigger hit today than in 1986.",
                "3 Musketeer chocolate bars used to have 3 flavors.",
                "These candy bars came in three breakable pieces and were originally made for sharing. Each candy bar contained a different flavor – strawberry, chocolate, and vanilla. ",
                "You cannot crack your knuckles.",
                "The sound it makes is due to the gasses that are released when you put pressure on them. When joints are stretched, the pockets of gas between the joints are released. ",
                "The bones of the human body can multiply in density.",
                "This occurs in very extreme cases. People who have discovered to have this rare gene mutation have cases of walking away from car accidents and other impact injuries without fractures.",
                "Your funny bone is a nerve.",
                "This funny bone connects the shoulder to the elbow and also has the ulnar nerve running along it. This nerve is responsible for sensing feelings in the ring and pinky fingers.",
                "A french pig was executed for killing a child.",
                "A pig from the middle ages was tied to a crime involving a murder. The pig was said to have come in contact with a child’s face which caused complications due to the injuries. Ultimately, this led to the death of the 3-month old child. The pig was sent to jail and eventually publicly executed.",
                "Pineapples are named after pinecones.",
                "This is mainly due to the similarity of its spiky outer skin. Pinecones are also the product of the pine tree. ",
                "Scotland has over 400 words to refer to snow.",
                "One of Scotland’s three official languages have a great extent of expressive phrases for snow. A few among these words include snaw (snow) and flindrinkin (light snow showers).",
                "There are more than 200 flavors of Kit Kat in Japan.",
                "You can find Japan exclusive flavored Kit Kats such as Adzuki (Red bean), Tamaruya Honten Wasabi (Japanese horseraused_item_groups), Shinshu Apple and more. Don’t forget to bring these exotic flavors home to share with your loved ones on your next trip to Japan!",
                "New Zealand was once auctioned on eBay.",
                "In 2006, an Australian man auctioned the country on the e-commerce platform with a starting bid of $0.01. The bidding prices were raised to $3,000 before officials of the website shut it down.",
                "There is a city in Oregon called Boring.",
                "It also has a sister city called Dull in Scotland. This town was named after its founder William H. Boring. ",
                "Leeches were used to predict the weather.",
                "This was a common practice during the Victorian era. However, further research proved that this type of prediction method was not reliable. ",
                "The ‘?!’ punctuation mark has a term.",
                "It is called an interrobang and was invented during the 1960s. This combination of punctuation made it easy for advertisements to express emotional questions. The interrobang originally looked like both marks were overlapped (‽).",
                "Only owning one guinea pig is illegal in Switzerland.",
                "Guinea pigs are herd animals and therefore become severely depressed when alone. Switzerland considered only owning one illegal to practice social rights for animals.",
                "A man in Florida once threw a live alligator through a drive-thru window.",
                "A 24-year-old man named Joshua James was being handed his order at a Wendy’s drive-thru before he randomly threw an alligator into the window. He was then later charged for assault, theft, and the illegal possession of an alligator. ",
                "drive through, random facts",
                "Image from Adobe Stock",
                "Great Britain once had a number where you can report rogue traffic cones.",
                "Britain launched a hotline in 1992 to improve public services. However, this policy was mocked for being pointless and a waste of government funds. ",
                "The largest recorded snowflake is 15 inches wide.",
                "This was dated January 28, 1887, in Montana, U.S.A. It was stated that army personnel witnessed the falling snowflake as frisbee-sized.",
                "McDonald’s once had bubblegum flavored broccoli.",
                "This was done as an attempt to make kids enjoy eating vegetables. However, this strange method failed for obvious reasons. ",
                "American Airlines saved money by getting rid of olives from their meals.",
                "About three decades ago the airline started looking into ways on how to save money. They started by removing a single olive on their first-class meals which lead to an annual savings of $40,000 during the 1980s.",
                "‘OMG’ was first used in a letter to Winston Churchill in 1917.",
                "A retired British admiral of the Royal Navy one day became excited due to headlines regarding the mobilizing forces of Britain. Excited to share this news with a college, the admiral quickly wrote a small acronym for ‘Oh My God!’ in his letter. Little did he know that he had just invented one of the most popular acronyms of today’s society.",
                "Sailors consider black cats good luck.",
                "Some sailors use black cats as the ship’s cat in hopes of a safe voyage. Some fishermen’s wives would even keep a black cat at home as well in hopes of their husband’s safe return. ",
                "A janitor invented the flaming hot Cheetos.",
                "One of the janitors working at the Frito-Lay plant in Southern California pitched the idea of adding chili powder to the regular product. Top Brass liked the idea and so did many consumers. This leads the janitor to a generous promotion where he is now an executive for PepsiCo.",
                "North Korean teachers were required to play the accordion.",
                "This musical instrument is also known as the “people’s instrument” due to its convenient size suitable for taking to marches. Students from all schools in the country are required to learn them. This explains why teachers need to pass an accordion exam before they get their teaching licenses during the 1990s.",
                "Melting glaciers make fizzy noises.",
                "This is also known as bergy seltzer because of how similar it sounds to fizzing soda. The sound is mainly due to the melting waters that free tiny bubbles trapped in ice.",
                "Male students attending Brigham Young University cannot grow beards.",
                "This Mormon flagship university has several strict guidelines that include banning premarital sex, alcohol, and tattoos. Unless you have a medical condition or a specific religion, men from this university are not allowed to grow a beard.",
                "A pistol can only be used by one hand.",
                "Many of these guns are mainly used for self-defense. Many models also only fire one shot after every pulled trigger. ",
                "Black taxis in London are tall for a reason.",
                "This is so that gentlemen are able to ride in them without having to remove their top hats. For many, this is considered to be very convenient. ",
                "Flipping a shark will render it temporarily immobile.",
                "Certain types of sharks enter a state of tonic immobility due to the shock when flipped over. Some killer whales have been recorded to exploit this weakness and purposely flip them over.",
                "The largest living organism is an aspen grove.",
                "This grove is referred to as Pando and can be found in the state of Utah. The grove is made up of 47,000 identical quaking aspen trees that cover over 106 acres.",
                "Someone will write and recite a poem at your funeral if you die in the Netherlands.",
                "This is mostly the case for those who die without a next of kin. This practice is done so that there will be at least one person at their funeral. ",
                "Alan Shepard played golf on the moon.",
                "He is also the one and only person to play golf on the moon during the Apollo 14 mission. His second shot traveled further than 200 yards. ",
                "Ioannis Ikonomou has been the chief translator of the European Parliament since 2002.",
                "This Greek translator has 32 languages under his belt and even knows languages outside of the EU. How’s that for a hidden talent?",
                "Kummerspeck is German for the weight gained during emotional eating.",
                "This vaguely translates to ‘grief bacon’. In some languages, this German term is said to add insult to injury. ",
                "SEARS once sold houses.",
                "SEARS once offered kit houses that you would have to assemble yourself. The original instruction manual was 75 pages long. This was much before every city and state in the U.S had Walmart.",
                "An encrypted monument stands outside of the CIA headquarters in Virginia.",
                "This monument was created by artist Jim Sanborn and features four inscriptions. Three of four of the inscriptions have been cracked already. However, no one seems to be able to crack the final code to this day.",
                "Cold water is just as cleansing as hot water.",
                "Modern detergents allow clothes to be clean equally with either warm or cold water. The only difference between the two is that warm water requires more energy used. ",
                "David Bowie helped topple the Berlin wall.",
                "During a performance in 1987, Bowie’s performance of ‘Heroes’ near the Reichstag could be heard by the police in East Berlin. This resulted in a police crackdown because such music was forbidden there. Some say that the violent police crackdown during the concert was a necessity in changing the mood against the state.",
                "Tap water in Manhattan is not Kosher.",
                "Very small crustaceans have been found in the tap water within New York City. Automatically, this cannot be considered kosher water.",
                "A park ranger from the U.S was once hit by lightning 7 times.",
                "This occurred between the years 1942 to 1977 at the Shenandoah National Park in Virginia. This park ranger holds the name of the Human lightning Rod in the Guinness Book of World Records.",
                "The fedora was originally a hat made for women.",
                "Over time, it became acceptable for men to wear them as well. Today, both men and women wear them mainly for fashion purposes.",
                "The story of Beauty and the Beast was aimed to make women open to arranged marriages.",
                "The original 1740 tale stated that the beast was an elephant-fish hybrid. The story mainly revolves around the idea of accepting arranged marriages for the sake of an alliance that was greater than them as a person. ",
                "Timothy Leary escaped prison.",
                "Rather than break out, he simply walked away from the minimum-security prison he had been placed in since 1970. He then later changed out of his prison uniform at a nearby gas station. ",
                "Bottled water has an expiration date.",
                "To be more precise, the expiration date is for the bottle itself, and not the water. This is because, over time, the plastic will start leaking into the water. ",
                "Now, this is one useful random fact you should know!",
                "Many pets from the U.S run away on July 4th.",
                "This is highly due to their fear of fireworks during America’s Day of Independence. ",
                "Queen Elizabeth cannot sit on the Iron Throne.",
                "By Royal law, the Queen is not allowed to sit upon a foreign throne. This was vital information that was only discovered when the Queen visited the set of Game of Thrones in Northern Ireland.",
                "Incan people used knots to keep track of records.",
                "These knots were also tied to pendants and cords. The knots are better known as quipu and its meanings would depend on its position on the cord. Over 600 examples of quipu have been discovered already.",
                "Monty Python has one of the most requested songs for funerals in England.",
                "‘Always look on the bright side of life’ is a common favorite during funeral services. This song was originally from the comedy classic Life of Brian. ",
                "The state of Virginia may contain hidden treasure.",
                "The Beale CIphers are a set of coded texts that are said to reveal the location of hidden treasure. Research believes that this is worth over $43 million in gold, silver, and jewels. Only one of three texts have been encoded so far.",
                "Fake ambulances are hired in Russia by the wealthy.",
                "They use this as a taxi for a faster and more convenient way of getting to their destinations. These vehicles can be hired for approximately $200 an hour. ",
                "Most businesses do not see the practicality of having diaper tables.",
                "However, they quickly change their minds after being given the idea that women would have to change their baby’s diapers on a bathroom floor. Who knows how much bacteria are on those floors!",
                "The world’s most successful pirate was a woman.",
                "She was a 19th-century Chinese pirate known as Ching Shih. She was the widow of Cheng I and easily succeeded her husband’s crew of over 1,800 ships and 80,000 men. ",
                "The KKK was taken down with help from Superman.",
                "A 16 episode series titled’ The Adventures of Superman’ was aired on the radio during the 1940s. This series incorporated the findings that activist Stetson Kennedy had been able to get while undercover in the KKK. ",
                "The Baseball Hall of Fame had a secret inductee.",
                "In 1988, a surreptitious addition to the Baseball Hall of Fame was made without anyone knowing. A bar owner visiting the hall slipped a photo of his father in a baseball cap into one of the glass cases. This photo remained in the hall for over 6 years before being discovered.",
                "Milk wagons are the reason why we have roadway lines.",
                "These lines were devised by a man who saw dots of milk spilling out from a moving milk wagon. These lines are considered to be the most important traffic safety device. ",
                "The most successful predator is a wild dog.",
                "According to research, canines have an 85% success rate when hunting. This is higher than that of a lion or cheetah. A lion only has a 17-19% success rate.",
                "A man was once saved by a sea lion.",
                "During a suicide attempt, a man by the name of Kevin Hines survived jumping off the Golden Gate bridge. However, during his jump, he had broken his back thus causing him to drown. A nearby sea lion witnessed this and swam beneath the man to keep him afloat until coast guards arrived. ",
                "Saliva can be used to monitor alcohol intake.",
                "It can also detect how often you smoke or use drugs. Some research also uses it to diagnose diseases. ",
                "The world’s largest pyramid cannot be found in Egypt.",
                "The Great Pyramid of Cholula is found in Cholula, Puebla, Mexico and holds the record for the largest pyramid in the world. This structure is four times as big as the Great Pyramid of Giza. ",
                "Some pandas fake a pregnancy to get better healthcare.",
                "An example would be a Chinese panda named Ai Hin who was believed to show signs of pregnancy. Zookeepers ensured that she was given better care for the upcoming baby that never arrived. ",
                "Cacti come in many different colors other than green.",
                "Some species of cacti are naturally brown or brown-green in color. Many will have a waxy substance on the surface which prevents water loss from transpiration.",
                "The inventor of the frisbee became a frisbee himself.",
                "During the 1950s, Steady Ed invented the frisbee, and later on the sport of disc golf in the 70s. He dedicated his life to frisbees that it was stated in his final testimonials that he wished to have his ashes turned into a frisbee itself. This was so that he would be able to play with his son even when he wasn’t around.",
                "Dolphins have names.",
                "Research has found that dolphins use their unique vocal whistles to identify and differentiate one another. When a specific call was played back to the dolphins, they would respond differently each time. ",
                "One species of ants can only be found in Manhattan, New York.",
                "Biologists discovered a new ant species between 63rd and 76th street in Broadway. This ant species was found to have a higher concentration of carbon in their bodies which was linked to a high corn-syrup diet. ",
                "Around 30,000 rubber ducks were lost at sea in 1992.",
                "These ducks we included in a shipment from Hong Kong to the U.S. One of the crates that were filled with rubber ducks was lost in the Pacific Ocean. To this day, rubber ducks continue to pop up on the shores of Australia to Alaska.",
                "Charles Darwin’s pet turtle outlived him.",
                "The turtle named Harriet lived 124 years after Darwin’s death. It was 176 years old by the time it passed away.",
                "Losing weight alters brain activity.",
                "Some records show that when women lose weight, their memory improves. Research has also linked obesity to poor memory, most especially those with pear-shaped body types.",
                "The folds in a chef’s hat represents the number of ways you can cook an egg.",
                "The toque is also known to determine the position of the chef in the kitchen and how experienced they are. How’s that for a hidden talent?",
                "Cactus spines can be used to make hooks.",
                "It can also be used to make needles and combs. Its fruit can also be used for item. ",
                "Depending on how they descend, waterfalls have different classifications.",
                "This can be divided into 10 main categories. The most common one is the plunge and multi-step. ",
                "Neil Armstrong never said ‘That's one small step for man’.",
                "The astronaut corrected that it was ‘that’s one step for a man, one giant leap for mankind’. No astronaut has ever uttered ‘Houston, we have a problem’ in real life as well. ",
                "The odds of getting a royal flush is 1 in 649,740.",
                "A straight flush is more likely in this case. Out of all the possible combinations, the chances of getting a single pair is 42%. ",
                "Driving south from Detroit will lead to Canada.",
                "Specifically, the directions are to head north from Windsor, Ontario, and crossing the Detroit River. Eventually, you would have already crossed the borders of the U.S.",
                "More people speak English as their second language than those who use it as their mother tongue.",
                "English has a total of nearly 2 billion fluent speakers and it is also the most widely spoken language in the world. However, only 350 million people speak this language natively. ",
                "Sleep deprivation makes it harder to lose weight.",
                "Lack of sleep can interfere with one’s hormone balance and decreases leptin. It also increases ghrelin which is the hormone that triggers hunger. ",
                "The world’s largest single drop waterfall is the Kaieteur Falls.",
                "This is located on the Potaro River in the center of Guyana’s rainforest. The average flow rate of the waterfall is 663 cubic meters per second.",
                "Your teeth are unique.",
                "No two teeth from different people will ever be alike. They are similar to fingerprints. ",
                "The most popular state bird is the Northern cardinal.",
                "Out of the 50 states in the U.S, seven voted for this species as their favorite bird. Perhaps this may be due to the Angry Birds trend from a few years back?",
                "Triskaidekaphobia is the fear of the number 13.",
                "This is mainly connected to the belief that the number 13 is unlucky and can bring upon misfortune. This is why you won’t see the number 13 on most building elevators. ",
                "A single dollar bill costs 5 cents to make.",
                "The current U.S circulation is approximately worth $1.79 trillion. That’s a lot of dollars. ",
                "Baby sea otters are unable to swim.",
                "Their mothers carry them wrapped around a piece of kelp while they hunt to keep them from drowning. This is only done until the pup learns how to float on his own eventually. ",
                "Snakes can predict earthquakes.",
                "Studies have found that most species of snakes can detect earthquakes before it occurs. They are also able to sense quakes that occur as far as 75 miles away and up to five days away. I guess you could say, they’re just that down to earth.",
                "The King of Hearts is the only King without a mustache.",
                "The king of hearts is also known as the ‘Suicide king’ because of the sword he inserts into his head. It sounds like he was going through a rough breakup.",
                "Only two diseases have been completely eradicated.",
                "These are smallpox and rinderpest. The last naturally occurring outbreak of smallpox was recorded in 1977.",
                "The only English word that ends with ‘mt’ is Dreamt.",
                "This is yet another example of idiosyncrasies in the English language. ",
                "May 29th is Put a Pillow on Your Fridge Day.",
                "This odd holiday dates back to the early 1900s where families would place a piece of cloth on top of their larders. This is regularly celebrated in Europe and the U.S. to bring luck and wealth. ",
                "The opposite sides of dice will always equal 7.",
                "The side of 1 and 6, 2 and 5, and 3 and 4 all equal 7. Go on, try it for yourself. ",
                "The metal studs found on denim jeans serve a particular purpose.",
                "These studs are placed on particular areas where the cloth is most likely to wear out faster. They act as a support for the cloth and make the jeans more durable. These studs are also known as rivets.",
                "The average adult spends more time on the toilet than exercising.",
                "On average, an adult will spend over 3 hours on the toilet per week but only one and a half hours exercising. Maybe it’s time we start working out in the bathroom?",
                "About 7% of American adults believe chocolate milk comes from brown cows.",
                "While this may not seem like a lot, this totals to around 16.4 million Americans. Imagine how’d they feel when you tell them buttermilk doesn’t have any butter in it.",
                "Cats can't taste sweet flavors.",
                "This is mainly because they do not possess the base pair of genes that detect sweet flavors. As a result, sugar does not code on their tongues as proper proteins. ",
                "Bananas get their curved shape by growing towards the sun.",
                "All bananas go through a process known as negative geotropism which indicates that they will grow towards the source of sunlight. This is also why it is rare to see a straight banana. ",
                "Your fingernails on your dominant hand grow faster.",
                "It also takes around half a year for a fingernail to grow from its base to the tip. These nails also grow faster on the bigger fingers rather than the smaller ones. ",
                "Apple seeds contain cyanide.",
                "If you chew or digest them, these seeds turn into hydrogen cyanide which is poisonous to humans. The same goes with apricot, cherry, and peach seeds.",
                "Frigate birds can sleep while flying.",
                "This is possible because they can keep only half of their brain’s hemispheres awake. Theoretically, they could fly forever if they didn’t need item or water. ",
                "Only one capital in the U.S has no McDonald’s.",
                "This capital is known as Montpelier, Vermont and it is also the smallest capital in America. Most residents have to drive to the next town over just to get a taste of the world-famous fries. ",
                "People used to answer the phone with “ahoy”.",
                "Back during the 1800s phone greetings were inspired by the Simpsons when Mr. Burns would pick up the phone with “Ahoy-hoy”. Thomas Eddison, however, wanted users to use ‘Hello’ instead. We know who won that argument.",
                "Playing dance music helps ward off mosquitoes.",
                "According to a study in 2019, the song ‘Scary Monsters and Nice Sprites’ contains both low and high frequencies that yellow fever mosquitoes dislike. Just dance the mosquitoes away.",
                "Billy goats urinate on their heads to become more attractive.",
                "This is one of the many strange mating rituals that occur on the animal planet. Generally, they do this during the summer through fall. ",
                "Movie trailers were originally shown after the movie.",
                "Trailers were first introduced during the 1910s, hence the name. But now modern marketing finds that it is more efficient for trailers to play before a film rather than after. ",
                "Among all the Disney princesses, Mulan has the highest kill count.",
                "Throughout the film, Mulan had killed almost 2,000 people in an avalanche. This includes the hun leader, Shan-Yu. ",
                "Pinocchio cannot say ‘my nose will grow now’.",
                "If he was to speak this sentence, it would create a paradox. For his nose to grow, he would have to lie, however, it cannot grow otherwise the statement would be true. ",
                "You are 13.8% more likely to die on your date of birth.",
                "This comes from the Swiss Mortality statistics based on deaths from 1969 to 2008. ",
                "Alaska is the only state whose name is on a single row on the keyboard.",
                "A and S are right next to each other, and so are L and K. Alaska is also one of the few words you can type on the second line of the QWERTY keyboard. ",
                "Many oranges are green.",
                "Often when ripe, oranges are in its natural green. In some locations in the world such as South America and tropical islands, oranges are green all year round. ",
                "Vantablack has a trademark.",
                "Anish Kapoor had won the executive rights to the world’s darkest shade of black and stated that only they could use it. Outraged, the creator of the pinkest pink made the color available for everyone but Kapoor. ",
                "Now that’s a colorful war.",
                "A man once set a record by putting on over 260 t-shirts.",
                "After the 20th shirt, Ted Hastings required assistance until the 260th shirt. Around 150 shirts in, his team was concerned about his ability to breathe due to the amount of fabric hugging him. In the end, he was able to beat the previous record of 257 shirts. ",
                "Queen Elizabeth is a trained mechanic.",
                "During her time as a princess, she was a member of the Women’s Auxiliary Territorial Service during WWII. This also makes her the only member of the Royal family to have served in the military. ",
                "Optical illusions can be found at the bottom of the sea.",
                "While exploring an underwater volcano, explorers came across a small lake-like pool that was upside down. On video, it looks almost too unreal. Further research stated that this lake is what keeps deep-sea creatures alive.",
                "123456 is the most common password.",
                "Following this are several variations of this sequence such as 123456789, 012345, etc. Other passwords include ‘iloveyou’, ‘sunshine’, and ‘password’. ",
                "The average American spends 2.5 days annually looking for lost items.",
                "This statistic comes from one of the nation’s largest independent lost-and-found surveys. The most commonly misplaced items are phones, keys, wallet, purse, and remotes. ",
                "Technicolored squirrels roam the lands of Southern India.",
                "They weigh around 4 pounds and measure up to 3 feet from tip to tail. The Malabar giant squirrel also blends better in the forest due to its multi-colored fur. ",
                "Basenji dogs can't bark.",
                "However, this does not mean that they are silent. The sound that they release is more of a warbling yodel. This specific breed of dog is the only dog that cannot bark.",
                "Dragonflies can’t walk.",
                "Even though they have six legs, they are unable to walk with them. Their legs are too weak for them to walk for long periods. ",
                "Space travel makes mice run in circles.",
                "NASA once sent 20 rodents to the International Space Station to see how the effects of space affected mice. Not long after they got there, the mice all started running in loops inside their cages. No one is exactly sure why this occurred, but researchers stated that the critters might’ve just been enjoying their time in space.",
                "The chicken is the closest relative to the T-Rex.",
                "Recent studies have now accepted that prehistoric dinosaurs have more in common with present-day birds than they do with reptiles. Research has found that their genetic makeup is shared more with chickens and ostriches rather than lizards. ",
                "Two-thirds of millennials go to sleep naked.",
                "According to a 2018 survey from MattressAdvisor, more than 60% of millennials regularly sleep in the nude. This is mostly due to the comforting feeling of skin against sheets. Are you part of this group?",
                "Sloths hold their breaths longer than dolphins.",
                "On the rare occasions that sloths leave their trees to go for a swim, they hold their breath underwater for up to 40 minutes long. This is 30 minutes longer than the average dolphin’s record. ",
                "More monopoly money is printed annually rather than actual currency.",
                "The U.S government prints around $974 million annually to replace old money. However, the company behind the beloved board game prints over $30 billion worth in fake money annually. ",
                "‘Schoolmaster’ is an anagram of ‘the classroom’.",
                "Schoolmaster is an old term used to refer to a male teacher. This does not mean that they are the actual master of the school. ",
                "Ravens are always aware when someone is watching them.",
                "These birds are known for being notoriously clever creatures and they display what is called the ‘theory of mind’. This is also known as the ability to attribute mental states to others. ",
                "An eagle is capable of killing a young deer and flying away with it.",
                "Bald eagles especially are one of the strongest and most aggressive species of eagles. These birds are capable of swooping in and carrying away your baby. ",
                "Baby spiders are called spiderlings.",
                "This is the official name for spiders that have yet to reach maturity. They are also born in groups of thousands. Maybe this name will make spiders seem cuter now.",
                "A palindrome is a sentence that is the same when read backward and forwards.",
                "A few examples include ‘Do geese see God’ and ‘racecar’. This is also noted to be one of the hardest sentences to construct using the English language. Can you come up with your own?",
                "The roar of a lion can be heard up to 5 miles away.",
                "Its roar is about 114 decibels. This is roughly 25 times louder than a gas-powered lawnmower. ",
                "The average male will become bored after 26 minutes of shopping.",
                "However, a study found that most women will not get bored until two hours later. Maybe next time you should just leave your boyfriend or husband at home. ",
                "The U.S. Navy uses Xbox controllers for their periscopes.",
                "Instead of using a control stick, the Navy opted for this controller so that the learning time could be reduced. This was also the least complicated way of controlling the periscopes. ",
                "There is a species of spider dubbed the ‘hobo spider’.",
                "These spiders are not deadly. However, most bites from these critters come from when they are accidentally crushed or squeezed by a person.",
                "Baby octopuses are the size of a flea.",
                "An octopus will lay tens and thousands of eggs. However, not all of these eggs survive. Some are eaten by predators and some do not live long enough to reach adulthood. Baby octopuses are also called larvae.",
                "The premiere of the TV reality show ‘16 and pregnant’ helped lower the rate of teen pregnancy.",
                "Statistics show that the rate dropped by 5.7% within the 18 months that the show was airing. I guess kids do copy everything they see on TV.",
                "Up to 20% of power outages in the U.S are due to squirrels.",
                "Squirrels tend to chew on the wires that run along the neighborhood. Luckily, these problems are easy to fix. ",
                "95% of people text things that they would never say in person.",
                "However, what’s more, interesting is what people are willing to say rather. People are more willing to be revealed over the phone rather than in person because of the psychological thrill it entails.",
                "The Mayo Clinic made glow in the dark cats while trying to find a cure for AIDS.",
                "This was mainly due to the specific protein in their bodies that was fluorescent green. They also used this protein to discover if their tests worked out or not. Glowing was a sign that the tests were successful.",
                "The Antarctic glaciers are made up of 3% penguin urine.",
                "Due to the extreme temperatures in this region of the world, the urine cannot evaporate. Hence how they freeze and mold with the glaciers around them.",
                "Facebook, Instagram, and Twitter are banned in China.",
                "Anyone caught using these platforms while inside the country can and will be arrested. Over 8,000 other domains are also blocked within the country. This rule is also implemented for visiting foreigners. ",
                "Honeybees can recognize human faces.",
                "Bees see in a compilation of over 5000 individual images. They combine all of these images to recognize their surroundings.",
                "The happiest prisoner on death row had an IQ of 46.",
                "Joe Arridy was sent to the gas chamber and entered with a smile on his face. It was then later discovered that he was innocent all along. What a twist.",
                "Violin bows are made from horsehair.",
                "A single violin bow uses around 160-180 strands of hair. They are all then attached next to each other to form a ribbon. Thick and kinked hairs are removed so that the bow remains smooth and straight.",
                "IKEA is an acronym.",
                "It stands for Ingvar Kamprad Elmtaryd Agunnaryd. This is also the name of the company’s owner.",
                "Stephen Hawking held a reception for time travelers in 2009.",
                "However, he did not publish news of this until after the event so that only real-time travelers were able to know about it. No one attended the event.",
                "A Norwegian Island made dying illegal.",
                "It is illegal to die in Svalbard because they are unable to safely bury the deceased due to the permafrost ground. When people on this island are about to die, they are brought to mainland Norway to pass away there.",
                "People who post their fitness routine on social media are more likely to have psychological problems.",
                "A study finds that this may lead an individual to addiction for attention and esteem. More often, people who share all of their physical activities online have a goal to boast about their activities. ",
                "Dr Pepper does not have a period.",
                "This is mainly due to the font of the logo that would make it look like ‘Di: Pepper’. Many were confused by this, hence the change. ",
                "There is an underwater version of rugby.",
                "Yes, it is also called underwater rugby. Two teams of 6 compete in a regulation pool while freediving. ",
                "You can burn calories just by standing.",
                "The average 150-pound person can burn 114 calories an hour just by standing still. Did this motivate you a little? ",
                "Mice typically only live for 6 months in the wild.",
                "This is mainly due to the number of predators that eat them. However, if they are kept as pets, their life span increases significantly by two years.",
                "The backward punctuation mark is used to identify sarcasm.",
                "It can also be used to signify irony. Or does it⸮",
                "If a body is too obese, it can cause complications when being cremated.",
                "A crematorium in Cincinnati was once set on fire during the cremation process of a 500-pound body. The body caused the fire to enlarge and spread to nearby containers which eventually set the entire place ablaze.",
                "Flossing your teeth improves memory.",
                "Here’s one useful random fact for you: Research suggests that if you keep your gums healthy and avoid gum disease, it will lead to the prevention of stiff blood vessels. Preventing stiff blood vessels also prevent memory problems. ",
                "A common souvenir people bring from the U.S are red solo cups.",
                "For many, this is considered a novelty as many party scenes in movies use these cups. It is also one of the most iconic cups in modern cinematics. ",
                "St. Lucia is the only country that is named after a woman.",
                "The French named this country after Saint Lucy of Syracuse. Lucy was a Christian martyr who had died during the time of the Diocletianic Persecution. ",
                "Ben & Jerry’s made a graveyard for their former flavors.",
                "These flavors all consist of discontinued ice cream flavors. Each flavor has its photo and as well as its life span. Each one even has its epitaph for people to remember it by.",
                "Pluto was one temporarily closer to the sun than Neptune.",
                "NASA recorded this instance during the years between 1979 to 1999. However, this was not the only time in history for a phenomenon like this to occur. ",
                "About $3.70 is given to an American child per tooth they lose.",
                "That’s triple the amount that the tooth fairy is said to give you. Maybe kids should take this opportunity to invest in being a dentist?",
                "There is a scientific term for brain freezes.",
                "The term is called sphenopalatine ganglion neuralgia. No wonder people stick with brain freeze.",
                "Only one letter in the English alphabet cannot be found on the periodic table.",
                "Can you guess which? The letter J is the only letter in the English alphabet that isn’t used for any of the elements in the periodic table.",
                "The calling shotgun for the front passenger seat comes from a messenger.",
                "The original term is the shotgun messenger and it was a term used to call the person who sat next to the driver. These messengers would act as guards and use a shotgun to prevent criminals from getting too close. Due to Hollywood’s influence, this term eventually made its way into our society.",
                "A single strand of spaghetti is individually referred to as spaghetto.",
                "Spaghetti is a plural word in Italian. Most pasta fans are surprised when they find out about it, were you one of them?",
                "Pointing your car keys to your head helps you find your car faster.",
                "This is because your head works like a radio transmitter. This then increases the rage of the car remote’s signal, thus allowing your car to respond faster. ",
                "The fat Buddha statue and in pictures is not Buddha himself.",
                "It was stated that the real buddha is nowhere near that fat. The real one was incredibly skinny because of self-deprivation.",
                "A jockey from 1923 managed to finish a race after dying.",
                "During the middle of a race, the jockey suffered from a heart attack. However, his horse was able to finish the race on its own and won. This event in history is the first and only time it has ever occurred since.",
                "The largest prime number has 17,425,170 digits.",
                "This new prime number is 2 multiplied by itself 57,885,161 and then subtracting 1. This number can also be read as 257,885,161-1.",
                "The largest grand piano in the world was built by a teenager.",
                "The piano is over 18 feet long and has 85 keys. This is 3 keys short of the standard 88. This grand piano can be found in New Zealand.",
                "A Polish doctor once faked an outbreak to keep the Nazi’s away.",
                "Eugene Lazowski announced that there was a typhus outbreak that ended up saving over 8,000 people during the second world war. At the time, this was considered to be a smart move because Lazowski decided to tug on the German’s phobias about hygiene.",
                "The 1996 film called Scream increased the usage of caller IDs in the U.S.",
                "The movie featured a serial killer who would anonymously call and murder his victims. After the film started showing in cinemas, the number of caller ID users tripled instantly in the U.S.",
                "The spiked dog collars are meant to protect their necks from attacks.",
                "This iconic accessory was invented by the Ancient Greeks to protect their dogs against wolves when attacked. The spikes would prevent predators from instantly causing heavy damage to their dog’s necks. ",
                "Jack Daniels died from a toe injury.",
                "The founder of the famous whiskey injured his toe after kicking a safe. This then later led to an infection until he eventually died of blood poisoning. ",
                "The boss in Metal Gear Solid 3 takes a week to beat.",
                "The player can either defeat them by not playing for one week or simply by changing the date on the console. Battle the boss as you would normally until it is almost beaten after that quit the game and change the date one or two weeks later. ",
                "The correct English translation for Jesus is Joshua.",
                "You are only able to get Jesus if you originally translate it from Hebrew to Greek, then Latin, then English. Jesus in Hebrew is Yeshua, which explains why google translate leads you to Joshua. ",
                "The first service animals were established in Germany.",
                "Service animals were more commonly used by the time of the first world war. However, references in history suggests that the use of service animals has been practiced since the mid-16th century.",
                "The planet was named Pluto thanks to an 11-year-old girl.",
                "She named the planet after the Roman God of the underworld because the planet was found to be frozen and lonely. Previously to this, the planet was referred to as planet X. This little girl lived till the age of 90 and died in April 2009.",
                "Around 50% of the mined gold on Earth comes from one source.",
                "This gold all once came from a single plateau located in South Africa. This plateau is known as the Witwatersrand. ",
                "12 plants and 5 animals make up 75% of diets around the world.",
                "The United Nations item and Agricultural Organization states that this lowers major health risks. By concentrating our diets to this specific group, we become more resilient to diseases, pests, and climate change.",
                "Shakira was rejected from her elementary choir group.",
                "Her music teacher at the time stated that her vibrato was too strong and that she sounded like a goat. Little did she know that this student would grow up to become one of the most inspirational female pop singers of all time.",
                "Minnesota has the world’s quietest room.",
                "The noise in this room is measured in negative decibels. It is so quiet, that you are able to heart your own heartbeat and as well as the sound of your bones moving. People say that this room is so quiet that it can also drive you crazy within 45 minutes.",
                "The Japanese have its own word for book hoarders.",
                "‘Tsundoku’ is the Japanese word for people who love to buy books but never read them. More often, these books are just left piled up inside their homes and never used. ",
                "Every decade, brain fibers lose 10% of their strength.",
                "At times of acute stress, they shrink even more than the average amount. This decline continues between the age of 20 to 80.",
                "The happiest countries in the world have the highest antidepressant consumers.",
                "These countries are ranked per capita. But at least now we know the medication really does work.",
                "The official language of Ireland is not the most spoken in the country.",
                "Instead, its inhabitants are more comfortable speaking in Polish rather than Irish. English is also more dominant since the beginning of the 19th century.",
                "Sourtoe cocktails are served in Yukon.",
                "The drink has been served since the 1970s and contains a shot of whiskey and a human toe floating inside the glass. All toes that are served with the drink have been mummified beforehand. ",
                "The release of Pokemon GO increased game-related accidents by 26.5%.",
                "During the first 5 months of its release, damages over $25.5 million had occurred. 2 deaths were also reported to have taken place while the individuals were playing Pokemon GO.",
                "Mexican prisons do not punish you for non-violent attempts to escape.",
                "This is because it is part of human nature to want freedom, and therefore is not seen as a crime. However, factors such as property damage and violence will cause prison guards to practice the necessary punishments.",
                "A BBC radio announcer once stated that they did not have any news.",
                "This occurred on April 18, 1930, where no major events took place that day. The announcer blatantly said that there was no news. ",
                "Starfishes do not have blood.",
                "If you were to cut one of its legs, blood will not spill out. Instead, they simply regrow their lost limbs as if nothing happened. Starfishes get their nutrients by using the seawater found in their vascular systems.",
                "The most mathematical flag in the world belongs to Nepal.",
                "In its constitution, there is a detailed step-by-step guide on how to draw the flag to its exact measurements. Nepal is also one of the few countries that have a triangular flag. ",
                "It did not take more than $1M to build Mt. Rushmore.",
                "The construction of this famous monument, Mount Rushmore, took over 14 years to complete. Over 400 workers built this monument from the group-up from 1927 to 1941.",
                "Samsung translates to ‘three stars’ in Korean.",
                "This name was chosen by the founder mainly due to its symbolism. He wanted the phone company to be powerful and everlasting as the stars. ",
                "Geckos eat the skin they shed.",
                "This is to prevent predators from finding and eating them. A small price to pay for survival.",
                "The U.S. government issued Santa Claus a pilot’s license.",
                "They also gave the jolly fat man a copy of airway maps in 1927 and had promised to keep the runway lights on. Many believe that this airway route is what he mainly uses to deliver his presents so quickly. ",
                "Polar bears charge at a group of walruses while hunting.",
                "They attack those that are crushed or left behind during the mass panic. Seeing a polar bear directly attack its prey is considered rare according to observations and studies. ",
                "You are able to become less depressed if you stay up.",
                "Research has found that there is a correlation between those who pull all-nighters and hazing out of your depression. This is because the longer you go without sleep, the more active your brain becomes. However, it is not recommended to go without sleep as it is a necessary need for humans to be able to function normally. ",
                "Adult cats will only meow at humans.",
                "Kittens will meow to its mother, but more frequently to humans as it matures. Kittens only meow at other cats to indicate hunger or the need for warmth. ",
                "There are more possible arrangements in shuffling a deck of cards than there are stars in the sky.",
                "Shuffling a deck of cards has 8×10^67 of approximate arrangements possible. It is highly unlikely to end up with the same sequence of cards per shuffle of the deck.",
                "Therapy has been found to be least effective at battling depression compared to playing video games.",
                "While addiction to video games may contribute to depression, research has found that it is not so when done in moderation. Some video games help with rage and emotional control which promotes positive results.",
                "Bi-weekly has two connotations.",
                "It can either be every two weeks or twice a week. The prefix bi- may be used to imply both or either meanings depending on context. ",
                "In 1911, the Mona Lisa was stolen from the Louvre.",
                "Ironically, people were more excited to see its exhibit empty rather than to see the painting itself. The number of visitors also increased during this incident. ",
                "The human vagina always holds a small amount of yeast.",
                "Even without an infection, the vagina will always hold some kind of fungus. Regular cleaning of this area is advised to prevent the growth of bacteria and infections. ",
                "Connecticut accidentally issued an Emergency Evacuation Alert in 2005.",
                "This alert notified all residents that they needed to evacuate the state immediately. However, only 1% of its population tried to evacuate. ",
                "Russians believe that eating ice cream will keep you warm.",
                "This is a method used to trick your body into thinking that its body temperature is crashing. This will then automatically force your body heat to readjust to survive. ",
                "Lemurs were one the size of gorillas.",
                "They were a common sight before in Madagascar and used that land as its stomping grounds. This species of giant lemurs are now sadly extinct. ",
                "Adding all the numbers on a roulette wheel will equivalent to 666.",
                "Hence the 666 strategy. Other people believe that this is proof that gambling is inherently evil.",
                "The day after Thanksgiving is the busiest day for American plumbers.",
                "This day is also known by many plumbers as “Brown Friday”. Many other drain companies also prepare ahead for this day to avoid major problems in their pipe systems.",
                "Mount Everest is not the tallest mountain in our solar system.",
                "This title goes to Olympic Mons, which is said to be three times as big as Everest. This mountain can be found on the planet Mars. ",
                "Russell Horning is famous on the internet as the Backpack Kid.",
                "Russel was born and raised in Georgia. He gained popularity after Rihanna gave him a shoutout on Instagram in 2016. He is also known for creating the floss dance.",
                "The oldest currency in the world is the British pound.",
                "It dates over 1,200 years old in usage. The British pound is also seen as the identity of British Sovereignty. ",
                "The Great Pyramid of Giza has more than four sides.",
                "This pyramid has a total of 8 sides which is double the normal amount for most pyramids. This includes the sides of its base.",
                "The only innate fear we have at birth is the fear of falling and loud noises.",
                "All other fears we learn after birth is learned. These fears are more often learned through traumatic experiences and by social means. ",
                "The average horse is capable of 746 watts of power.",
                "This is approximately 15 horsepower. This term was developed during the 18th century when the steam engine was being marketed. ",
                "The Leaning Tower of Pisa is tilted due to the soil at its base.",
                "The soft soil it is built on is also the reason why it was able to survive 4 major earthquakes. This tower took over 344 years to build and was completed in 1178.",
                "Slavery was once legal in Mauritania.",
                "This was legal until 2007. However, 1-4% of the population still continues to live as slaves within its borders. ",
                "England and Portugal still maintain their alliance until today.",
                "This is also known as the longest unbroken alliance in world history. This treaty began in 1386 between the two countries and they still remain strong allies until today. ",
                "Garlic is known to attract leeches.",
                "Leeches take around 15 seconds to attach themselves to a hand covered in garlic. However, it takes them over 40 seconds to suck the blood dry from its victim. ",
                "Uranus takes 84 years to orbit the sun.",
                "This is longer than any of the other planet’s orbits in our solar system. One rotation on Uranus is around 17 hours long. It is also the coldest planet in our solar system.",
                "A shape with 26 sides is known as a rhombicuboctahedron.",
                "Playing dice are the only common objects that have this shape. These dice are also ideal for labeling with the English alphabet. ",
                "Applying equal pressure on an egg will prevent it from breaking.",
                "It is nearly impossible to break eggs this way because its sides are the strongest part of the egg’s shell. The curved form of the egg helps it distribute pressure evenly which makes the take difficult to accomplish. ",
                "There are over 6,000 species of grass.",
                "The most common examples are rice, wheat, oats, and sugarcanes. Some of these species have been on our planet for more than 1,000 years already. ",
                "The Earth holds about 11 quintillion pounds of air.",
                "The force that gravity holds it in place is equivalent to a pressure of 15 psi. There are a total of 5 regions in our atmosphere. ",
                "There is a term for a person that is perpetually afraid of being late.",
                "Time anxiety is the fear of running out of time. People who suffer from this have an obsession with spending their time in the most meaningful way possible. Commonly, people who suffer this are also in a constant rush which makes those around them feel uncomfortable.",
                "The queens in ant colonies live for around 30 years.",
                "This was observed by keeping queen ants in captivity. On average they lived for almost 29 years. However, it is estimated that they are able to live much longer in the wild with a full colony.",
                "The map size in GTA V is twice the actual size of Manhattan.",
                "This totals to around 28 square miles long. Compared to the previous games, this map is much larger than most of the games combined. ",
                "The paint on the Eiffel tower is equivalent to ten elephants.",
                "The tower is re-painted every 7 years. There are also over 30 replicas of this tower all around the world. ",
                "The largest living thing on Earth is a tree.",
                "This Giant Sequoia is named General Sherman and can be found protected at California’s Sequoia National Park. The tree is recorded to be 52,500 cubic feet in volume.",
                "You can fit about 400 grapevines in one acre of land.",
                "This is equivalent to almost 5 tons of grapes in total. That’s a lot of wine. ",
                "Multiplying 1089 by 9 will give you the same numbers in reverse.",
                "You just grabbed your calculator for this one, didn’t you?",
                "August 17th celebrates Black Cat Appreciation Day.",
                "This was declared by Wayne Morris as a way to remember his sister and her black cat. This day is meant to symbolize the bond between black cats and their owners. ",
                "Eminem’s ‘Rap God’ holds the world record for the most words in a hit single.",
                "The 6-minute song has 1,560 words in total and are all delivered clearly by the artist. That’s a little over 4 words per second.",
                "A British scientist studying chocolate has her taste buds insured.",
                "Her taste buds are insured for one million British Pounds. This entails that she would need to avoid very hot items in order to keep her taste buds from damage. ",
                "The Titanic included dogs as their passengers.",
                "When the ship sank in 1912, 3 dogs were able to survive. Prior to the ship sinking, they had all been traveling with their owners in the first-class cabins. ",
                "China has a series of underground tunnels.",
                "These tunnels are over 3000 miles long. Its main purpose is to store and transport mobile intercontinental ballistic missiles from one destination to the other.",
                "Pitbulls are ranked as the most affectionate breed of dogs.",
                "They are also ranked the least aggressive despite their appearances. These dogs usually become aggressive if they are forcibly trained. ",
                "Tanning beds increase your risk of developing melanoma.",
                "If you were to use one of these machines before the age of 30, your chances increase by 75%. Tanning beds and lamps have been proven to be the highest cancer risk according to research.",
                "One of the cleanest dog breeds in the world are poodles.",
                "This dog breed does not excessively molt and rarely smell. Often people who have pet allergies adopt poodles because of their grooming routine. ",
                "Owls have specialized feathers.",
                "These feathers have the edges protrude out to dissipate the airflow when it flies. Because of this, their flights are silent which makes them deadly hunters at night. ",
                "The human brain uses 20 percent of the oxygen in your body.",
                "This is continuous and it does not rest even when we sleep. This is why when people yawn it is believed to be the brain’s way of cooling down. ",
                "Staying in a negative relationship can lower your immune system.",
                "Studies indicate that negative emotions lead to lowered immune responses against diseases. Other studies support this by stating that stress and emotions can adversely affect the immune system.",
                "The insect population of the world is 1 billion times more than the world population.",
                "The world has 7 billion people living today. If you take that and multiply it to a billion, that’s a scary thought.",
                "People once believed that chewing on tree bark will keep your gums healthy.",
                "It was also used as a method of pain relief. The bark of a willow tree contains high traces of salicylic acid which is an active ingredient in aspirin today.",
                "Kangaroos never stop growing.",
                "From birth, they will continue to grow until they pass away. They are also the largest marsupials on the planet. ",
                "Red blood cells circulate your body at an amazing speed.",
                "Within 20 seconds, one RBC would have already completed one lap around your circulatory system. These blood cells help deliver oxygen to all the organs and cells in the body. ",
                "The flashes of light when you rub your eyes are called phosphenes.",
                "This phenomenon allows you to see light without any actual light source present. This occurs due to many possible sources like magnetic stimulation, or random firing of cells in the visual system. ",
                "Cordozar Calvin Broadus Jr. is Snoop Dog’s real name.",
                "The famous rapper takes his celebrity name from his mother. According to him, his mother stated that it was because he looked like Snoopy from the iconic Peanuts franchise. ",
            )
        )
    )


# reminders
#######################################################################
#######################################################################
def water_notification():
    speak("please drink water")


# reminder
def take_a_break():
    speak("I think you need to take a break")


# reminder
def shower_notification():
    speak("Did you shower")


# reminder
def cleanup():
    speak("I think you need to take some time to clean up")


# OPEN&CLOSE windows apps
def winapp():
    from AppOpener import open, close

    speak("Here is a list of installed programs ")
    print("1. Open <any_name> TO OPEN APPLICATIONS")
    print("2. Close <any_name> TO CLOSE APPLICATIONS")
    print("")
    open("help")
    print("TRY 'OPEN <any_key>'")
    open("LS", match_closest=True)
    inp = takeCommand().lower()
    if "close " in inp:
        app_name = inp.replace("close ", "").strip()
        close(app_name, match_closest=True, output=False)
    if "open " in inp:
        app_name = inp.replace("open ", "")
        open(app_name, match_closest=True)


# Hello
#######################################################################
#######################################################################
def Hello():
    speak(
        random.choice(
            (
                "Hey What can i do for you today?",
                "How do you do Please state your problem.",
                "Hi! what we doing?",
                "How have you been?",
                "How do you do?",
                "Hey",
                "Hey man",
                "Hey! ,Hi!",
                "How’s it going?",
                "How are you doing?",
                "What’s up?",
                "What’s new? ",
                "What’s going on?",
                "How’s everything? ",
                "How are things?",
                "How’s life?",
                "How’s your day? ",
                "How’s your day going?",
                "Good to see you ",
                "Nice to see you",
                "Long time no see or ",
                "It’s been a while",
                "Greetings",
                "Yo!",
                "Howdy!",
                "Sup? or Whazzup?",
                "G’day mate!",
                "Hiya!",
            )
        )
    )


# Greetings gm, GN , GAFN
def wish():
    current_time = time.localtime()
    if current_time[3] < 12:
        message = " Good Morining "
        speak(" " + "'" + message + " Im ready to go!")
    elif current_time[3] in range(12, 18):
        message = " Good Afternoon "
        speak(" " + "'" + message + " Hey Josh, did you eat yet?")
    else:
        message = " Good Evening " + "Josh"
        speak(" " + "'" + message + " i hope we are having a pleasant evening")


# sleep mode
#######################################################################
#######################################################################
def ZZZ():
    if __name__ == "__main__":
        print("Ok, i'll be quite...")
        print("if you need me im here")
        input("Tap me if you need me for something ZZZZzzzZZzzZZ")
        print(
            "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ.... ok ok Im up"
        )
        speak(random.choice(("I'm awake now...", "OK,OK Im here.., whats up?")))


# pc diognosics
def feel():
    gpu = GPUtil.getGPUs()[0]
    speak(gpu.temperature)
    battery = psutil.sensors_battery()
    speak(("My energy levels are at : ", battery.percent))
    speak(("My plug? : ", battery.power_plugged))
    print(gpu.temperature)
    print("Battery percentage : ", battery.percent)
    print("Power plugged in : ", battery.power_plugged)
    print("Battery left : ", battery.secsleft)


# pc specs
def aboutme():
    speak(platform.machine())
    speak(platform.version())
    speak(platform.platform())
    speak(platform.uname())
    speak(platform.system())
    speak(platform.processor())


# wikipedia api
#######################################################################
#######################################################################
def wikime1(query):
    import wikipedia

    speak("Let me think")
    query = query.replace("i need information on", "")
    result = wikipedia.summary(query, sentences=5)
    speak("i think i found what you need...")
    speak(result)
    speak(
        random.choice(
            (
                "Why do you ask?",
                "Does that question interest you?",
                "What is it you really want to know?",
                "What do you think?",
                "What comes to your mind when you ask that?",
            )
        )
    )


# wikipedia api
def wikime2(query):
    import wikipedia

    speak("Let me think")
    query = query.replace("i want to know about", "")
    result = wikipedia.summary(query, sentences=5)
    speak("i think i found what you need...")
    speak(result)
    speak(
        random.choice(
            (
                "Why do you ask?",
                "Does that question interest you?",
                "What is it you really want to know?",
                "What do you think?",
                "What comes to your mind when you ask that?",
            )
        )
    )


# wolfram API
def brain():
    import wolframalpha

    question = str(takeCommand())
    app_id = ""
    client = wolframalpha.Client(app_id)
    res = client.query(question)
    answer = next(res.results).text
    print(answer)
    speak(answer)


def cams():
    webbrowser.open("https://camview.mygeeni.com")
    os.startfile("C:\Program Files\BlueStacks_nxt\HD-Player.exe")
    ZZZ()


def fancy():
    # def call order
    os.chdir("C:\\Users\\HP\\Documents")
    os.startfile("C:\\Users\\HP\\Documents\\Glass2k.exe")
    time.sleep(5)
    os.chdir("C:\\Program Files (x86)\\Lively Wallpaper")
    os.startfile("C:\\Program Files (x86)\\Lively Wallpaper\\Lively.exe")
    os.chdir("C:\\Users\\HP\\Documents")
    import pyautogui

    pyautogui.hotkey("winleft", "d")
    music()
    os.system("start explorer shell:appsfolder\\13545x2.Kauna_s6p2eat6f0r4t!App")
    resize()
    wish()
    cmands()


def dice():
    # importing module for random number generation
    # range of the values of a dice
    min_val = 1
    max_val = 6

    # to loop the rolling through user input
    roll_again = "yes"

    # loop
    while roll_again == "yes" or roll_again == "y":
        print("Rolling The Dices...")
        print("The Values are :")

        # generating and printing 1st random integer from 1 to 6
        print(random.randint(min_val, max_val))

        # generating and printing 2nd random integer from 1 to 6
        print(random.randint(min_val, max_val))

        # asking user to roll the dice again. Any input other than yes or y will terminate the loop
        roll_again = input("Roll the Dices Again?")


def password():
    passlen = int(input("enter the length of password"))
    s = "abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()?"
    p = "".join(random.sample(s, passlen))
    print(p)


# Ai defined here
#######################################################################
#####
# cant' share mic with zoom or google meet ##################################################################
def OpenAIbot():
    import os
    import sys
    import openai
    from langchain.chains import ConversationalRetrievalChain, RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    from langchain.llms import OpenAI
    from langchain.vectorstores import Chroma

    os.environ['OPENAI_API_KEY'] = ''
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = True

    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("C:\\Users\\HP\\FoodMonitor\\FoodMonitor")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = takeCommand().lower()
        if query in ['quit', 'q', 'exit']:
            Take_query()
        result = chain({"question": query, "chat_history": chat_history})
        speak(result['answer'])
        print(result['answer'])

        chat_history.append((query, result['answer']))
        query = None


def OpenAIbottrine():
    import os
    import sys
    import openai
    from langchain.chains import ConversationalRetrievalChain, RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    from langchain.llms import OpenAI
    from langchain.vectorstores import Chroma

    os.environ['OPENAI_API_KEY'] = ''
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = True

    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    if PERSIST and os.path.exists("persisttrine"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persisttrine", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("C:\\Users\\HP\\FoodMonitor\\trineMonitor")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persisttrine"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = input("Prompt: ")
        if query in ['quit', 'q', 'exit']:
            Take_query()
        result = chain({"question": query, "chat_history": chat_history})
        speak(result['answer'])
        print(result['answer'])

        chat_history.append((query, result['answer']))
        query = None


def OpenAIbotdatasci():
    import os
    import sys
    import openai
    from langchain.chains import ConversationalRetrievalChain, RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    from langchain.llms import OpenAI
    from langchain.vectorstores import Chroma

    os.environ['OPENAI_API_KEY'] = ''
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = True

    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    if PERSIST and os.path.exists("persistdata"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persistdata", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("C:\\Users\\HP\\FoodMonitor\\dataMonitor")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = takeCommand().lower()
        if query in ['quit', 'q', 'exit']:
            Take_query()
        result = chain({"question": query, "chat_history": chat_history})
        speak(result['answer'])
        print(result['answer'])

        chat_history.append((query, result['answer']))
        query = None


def OpenAIbotingram():
    import os
    import sys
    import openai
    from langchain.chains import ConversationalRetrievalChain, RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    from langchain.llms import OpenAI
    from langchain.vectorstores import Chroma

    os.environ['OPENAI_API_KEY'] = ''
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = True

    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("C:\\Users\\HP\\FoodMonitor\\ingramMonitor")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = takeCommand().lower()
        if query in ['quit', 'q', 'exit']:
            Take_query()
        result = chain({"question": query, "chat_history": chat_history})
        speak(result['answer'])
        print(result['answer'])

        chat_history.append((query, result['answer']))
        query = None



def Take_query():
    flag = True
    while flag is True:
        # user voice
        query = takeCommand().lower()

        # Pre-Trained ML responses microsoft/DialoGPT-medium-
        import transformers

        nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        chat = nlp(transformers.Conversation(query), pad_token_id=50256)
        res = str(chat)
        res = res[6 + res.find("bot >> ") :].strip()
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        ###### hard coded personal responses-Main Commands #####################
        if "data analysis mode" in query:
            speak("Entering Analysis Mode: Please stand by...")
            math()
            DA()
        
        elif "assistant" in query:
            Hello()
            speak("Hello i am your personal information assistant, ask me questions... ask me to write up stuff for you")
            speak("Loading...")
            OpenAIbottrine() 

        elif "Math Academy" in query:
            mathAch()

        elif "kitchen" in query:
            inventory()

        elif "work" in query:
            work()

        elif "trine advisor" in query:
            OpenAIbottrine()

        elif "sudoku" in query:
            Sudoku()

        elif "pray" in query:
            pray()
            continue
        #fix 
        elif "commands" in query:
            cmands()
            continue

        elif "we doing" in query:
            tellDay()
            weather()
            speak("tap me if you need me")
            ZZZ()
            continue

        elif "my schedule" in query or "calendar" in query:
            calender()
            continue

        elif "open an app" in query:
            music()
            winapp()
            continue

        elif "quote" in query:
            gita()
            selfhelp()
            continue

        elif "what time" in query:
            speak(Time())
            continue

        elif "sleep" in query:
            gita()
            speak(" night night  ")
            ZZZ()
            continue

        elif "be quiet" in query:
            speak(" ok... got it  ")
            ZZZ()
            continue

        elif "my files" in query:
            speak(
                "your files are open, if the window didn't come up check the taskbar."
            )
            os.startfile("C:\\Users\\HP\\")
            continue

        elif "want to watch" in query:
            speak(random.choice(("one sec...", "pulling up some stuff!")))
            webbrowser.open("https://srstop.link")
            webbrowser.open("https://www.youtube.com/feed/subscriptions")
            webbrowser.open("https://www.pogdesign.co.uk/cat/")
            ZZZ()
            continue

        elif "lost my phone" in query:
            speak(random.choice(("one sec... lets call it", "i can help!")))
            webbrowser.open("https://callmyphone.org")
            continue

        elif "gym" in query:
            speak(" Ah shit, here we go again!")
            speak("set yourself up .... get ready ... we will start soon")
            speak("get all the gear you will need ready")
            speak("here we go...")
            speak("ok, ready! set! GO!")
            exorcise()
            continue

        elif "don't listen" in query or "stop listening" in query:
            speak(random.choice(("Ill be back in 10")))
            time.sleep(10)
            continue

        elif "open youtube" in query:
            speak("Opening Youtube")
            webbrowser.open("https://youtube.com")
            continue

        elif "open opera" in query:
            speak("Opening Opera")
            webbrowser.open("https://google.com")
            continue

        elif "open stack overflow" in query:
            speak("..Opening Stack. Happy coding")
            webbrowser.open("https://stackoverflow.com")
            continue

        elif "rstudio" in query:
            rstudio()
            ZZZ()

        elif "vs code" in query:
            vscode()
            ZZZ()

        elif "BlueStacks" in query:
            bluestacks()
            ZZZ()

        elif "play some music" in query or "play a song" in query or "party" in query:
            speak(random.choice(("Here you go some music")))
            webbrowser.open("https://www.youtube.com/watch?v=lP26UCnoH9s")
            continue

        elif "play music" in query or "play my song" in query or "dance party" in query:
            speak("Here you go some music")
            webbrowser.open("https://www.youtube.com/watch?v=43ynqJ1wcwk")
            continue

        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        ##################### Sustaining a conversation #######################

        elif "bored" in query or "entertain" in query:
            import pyjokes

            random.choice(
                (
                    res,
                    facts(),
                    facts(),
                    speak(pyjokes.get_joke()),
                    water_notification(),
                    take_a_break(),
                )
            )
            continue

        elif "fact" in query:
            speak(random.choice((" ok, ok , ok ", "ok", "sure", res)))
            facts()
            continue

        elif "who made you" in query or "who created you" in query:
            speak(
                random.choice(
                    (
                        "I have been created by Josh",
                        res,
                        "God made me with the help of josh",
                    )
                )
            )
            continue

        elif "joke" in query:
            import pyjokes

            speak(
                random.choice(
                    (
                        pyjokes.get_joke(),
                        "fuck you. the  end.",
                        "no",
                        pyjokes.get_joke(),
                        pyjokes.get_joke(),
                        pyjokes.get_joke(),
                        pyjokes.get_joke(),
                        pyjokes.get_joke(),
                        pyjokes.get_joke(),
                    )
                )
            )
            continue

        elif "good morning" in query:
            speak(random.choice(("Hey!", res)))
            speak(random.choice(("How are you?", res)))
            continue

        elif "i love you" in query:
            speak(random.choice(("Aww. I Wuv you too", res)))
            continue

        elif "mr. penguin" in query:
            wish()
            music()
            speak(
                random.choice(
                    ("I am Penguin. Version four point o at your service...", "what?")
                )
            )
            continue

        elif "who am i" in query or "what am i" in query:
            speak(
                random.choice(
                    (
                        "If you can talk then you should be human.",
                        "you could be a dumbass... but i'm not an expert.",
                        res,
                    )
                )
            )
            continue

        elif "yes" in query:
            speak(random.choice(("Good. Do you need any thing?", res)))
            continue

        elif "is love" in query:
            speak(random.choice(("baby dont hurt me, dont hurt me.... no more", res)))
            continue

        elif "who are you" in query:
            speak(
                random.choice(
                    ("I am a virtual assistant. created by Josh", "who are you?", res)
                )
            )
            continue

        elif "your name" in query:
            speak(
                random.choice(
                    (
                        res,
                        " My name is Mr Penguin. I am a computer program created by Joshua Rocky Lizardi.",
                    )
                )
            )
            continue

        elif "about yourself" in query:
            speak(
                " My name is Mrs Smith I am a computer program created by Joshua Rocky Lizardi."
            )
            speak("in my current form my specs are")
            aboutme()
            speak(res)
            continue

        elif "why are you here" in query:
            speak(random.choice(("I was created. So I am.  ", res)))
            continue

        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ####Informing: Giving information Question Google homepage,Wolfram & Wikipedia#

        elif "i need information on" in query:
            wikime1(query)
            continue

        elif "i want to know about" in query:
            wikime2(query)
            continue

        elif "weather" in query or "rain" in query or "snow" in query:
            weather()
            continue

        elif "question" in query:
            speak(" Sure, whats on your mind?")
            brain()
            continue

        elif (
            "what is" in query
            or "who is" in query
            or "how much is" in query
            or "when is" in query
        ):
            import wolframalpha

            client = wolframalpha.Client("V4PHXG-YKLJPVJL5V")
            res = client.query(query)
            try:
                print(next(res.results).text)
                speak(next(res.results).text)
            except StopIteration:
                print("No results")
            continue

        elif "open wikipedia" in query:
            webbrowser.open("https://wikipedia.com")
            continue

        elif "where is" in query:
            query = query.replace("where is", "")
            location = query
            speak("User asked to Locate")
            speak(location)
            webbrowser.open("https://www.google.nl / maps / place/" + location + "")
            continue

        elif "calculate" in query:
            import wolframalpha

            app_id = ""
            client = wolframalpha.Client(app_id)
            indx = query.lower().split().index("calculate")
            query = query.split()[indx + 1 :]
            res = client.query(" ".join(query))
            answer = next(res.results).text
            print("The answer is " + answer)
            speak("The answer is " + answer)
            continue

        elif "search for" in query:
            query = query.replace("search for", "")
            webbrowser.open(query)
            continue

        elif "look up" in query:
            query = query.replace("look up", "")
            webbrowser.open(query)
            continue

        elif "google" in query:
            query = query.replace("google", "")
            webbrowser.open(query)
            continue

        ################################################################################
        ################################################################################
        elif "show cams" in query:
            cams()
            SeeScreen()

        elif "inventory" in query:
            currentinventroy()

        elif "dice" in query:
            dice()

        elif "i need a new password" in query:
            password()
        
        elif "fancy mode" in query:
            fancy()
        ################################################################################
        ############## Leave App######################################################

        elif "don't need anything" in query or "go away" in query:
            speak("ok then bye...")
            break

        elif "exit" in query or "fuck off" in query:
            speak("Thanks for giving me your time")
            break

        elif "goodbye" in query or "later" in query:
            speak(
                random.choice(
                    (
                        " k k see ya soon...",
                        "bye",
                        "see ya",
                        "Farewell",
                        "So long",
                        "Take care",
                        "See you around",
                        "Catch you later",
                    )
                )
            )
            gita()
            speak("also...")
            facts()
            speak(
                random.choice(
                    (
                        " k k see ya soon...",
                        "bye",
                        "see ya",
                        "Farewell",
                        "So long",
                        "Take care",
                        "See you around",
                        "Catch you later",
                        "bye, take care... i'll be here if you need me",
                    )
                )
            )
            print("Bye! take care..")
            break

        elif query:
            if res == "I'm sorry, I don't understand.":
                print("")
                speak("")
            elif res != "I'm sorry, I don't understand.":
                speak(res)
                print(res)
            continue

@ray.remote
def PenGUIn():
    #SeeScreen()
    #SeeWebCam()
    #fancy()
    os.chdir("C:\\Users\\HP\\Documents")
    cmands()
    tellDay()
    weather()
    Take_query()


if __name__ =="__main__":
    ray.init()
# Execute func1 and func2 in parallel.
    ray.get([SeeWebCam.remote(), PenGUIn.remote()])

