import os
import streamlit as st
import pandas as pd
import datetime
from openai import OpenAI
import tiktoken
from moviepy.video.io.VideoFileClip  import VideoFileClip
import math
import time
import whisper 
import spacy
import re
from pytube import YouTube
import moviepy.editor as mp

#Importing libraries
import os
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium import webdriver
import time
import requests

def run_tiktok_scraper(url):
    #defining the download path 
    download_path = 'tmp'
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    #chrome_options.add_argument("--headless")
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(5)
    captcha_close_button = driver.find_element(By.ID, "verify-bar-close")
    captcha_close_button.click()
    continue_as_guest_button = driver.find_element(By.CLASS_NAME, "[class*='DivGuestModeContainer']")
    continue_as_guest_button.click()
    time.sleep(5)
    setting_button = driver.find_element(By.CSS_SELECTOR, "[class*='DivVideoSwiperControlContainer']")
    actions = ActionChains(driver)
    actions.context_click(setting_button).perform()
    download_b = driver.find_element(By.CSS_SELECTOR, "[class*='LiItemWrapper']")
    download_b.click()
    time.sleep(5)
    driver.quit()
    print("Video downloaded successfully in:", download_path)



def run_youtube_scraper(url):
    YouTube(url).streams.first().download()
    yt = YouTube(url)
    stream = yt.streams.first()
    stream.download()
    mp4_streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')
    if mp4_streams:
        mp4_streams.first().download(output_path="tmp")
    else:
        print("No mp4 streams available.")

#Loading the English Language Model
nlp = spacy.load("en_core_web_sm")

#Remove Stopwords
def remove_stopwords(text):
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if token.text.lower() not in stop_words]
    return " ".join(filtered_tokens)

#Check for given url is a tik-tok video
def tiktok_validation(url):
    tiktok_regex = r'(https?://)?(www\.)?tiktok\.com/.+'
    match = re.match(tiktok_regex, url)
    return bool(match)

#Check for given url is a youtube video
def youtube_validation(url):
    youtube_regex = r'(https?://)?(www\.)?''(youtube|youtu|youtube-nocookie)\.(com|be)/''(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    match = re.match(youtube_regex, url)
    return bool(match)


def is_api_key_valid(api_key, client):
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Hello World!"},
        ]
        )
    except:
        return False
    else:
        return True
    
def get_model_selection():
    # Display model selection after API key check passed
    model = st.selectbox("Select the AI model to use:", ("gpt-3.5-turbo", "gpt-3.5-turbo-0125",  "gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview", "gpt-4-0125-preview"))
    return model


def transcribe(audio_file_path,client):
    with open(audio_file_path, "rb") as audio_file_object:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_object,  # Pass the file-like object directly
            response_format="text"
            )
        return transcript

def tokenizer(string):
    #Returns the number of tokens in the transcript.
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def tokens_check(tokens, model):
    model_dict = {'gpt-3.5-turbo': 4096, 'gpt-3.5-turbo-0125' : 16385, 'gpt-3.5-turbo-1106' : 16385, 'gpt-4' : 8192, 'gpt-4-1106-preview': 128000, 'gpt-4-0125-preview' : 128000}

    #Check the limit of the specified model
    model_limit = model_dict.get(model)
    # Check if the tokens exceed the limit for any model
    exceeding_models = [m for m, limit in model_dict.items() if tokens > model_limit]

    if exceeding_models:
        st.error(f"Warning: Token usage ({tokens}) is too big for any GPT models to handle. Unfortunately, this video cannot be used. Please choose a different video!")
    
    if model_limit is not None:
            # Check if the number of tokens exceeds the limit
            if tokens > model_limit:
                st.error(f"Warning: Token usage ({tokens}) exceeds the limit ({model_limit}) for model {model}. Please switch your model to one that can handle more tokens.")
            else:
                st.success(f"Token usage ({tokens}) is within the limit ({model_limit}) for model {model}.")
    else:
        st.error(f"Error: Model {model} not found in the dictionary.")

def analyze(transcript, model, client, container):
    system_msg1 = "Your task is to extract up to six keywords from the text provided to you, sorted in order of criticality. Follow the format \"Keywords: ...\""

    system_msg2 = "Your next task is to determine if the text contains any misinformation. You only could say \'May contain misinformation\', \'Cannot be recognized\' or \'No misinformation detected\'. Follow the format \"Classification Status: ...\""

    system_msg3 = "Lastly, you must briefly summarize the reasons for determining whether the statement contains misinformation. Provide three or less reasons of no more than 50 words each."

    chat_sequence = [
        {"role": "system", "content": "You are an experienced scientist and medical doctor. You need to fully read and understand the text paragraph given below. Then complete the requirements based on the contents therein." + transcript},
        {"role": "user", "content": system_msg1},
        {"role": "user", "content": system_msg2},
        {"role": "user", "content": system_msg3}
    ]   

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoding.encode(transcript))
    model_dict = {'gpt-3.5-turbo': 4096, 'gpt-3.5-turbo-0125' : 16385, 'gpt-3.5-turbo-1106' : 16385, 'gpt-4' : 8192, 'gpt-4-1106-preview': 128000, 'gpt-4-0125-preview' : 128000}
    model_limit = model_dict.get(model)
    exceeding_models = [m for m, limit in model_dict.items() if tokens > limit]
    if exceeding_models:
        container.update(label = f"Warning: This video's token usage of {tokens} tokens is too big for any GPT models to handle. Unfortunately, this video cannot be used. Please choose a different video!", state="error", expanded=False)
    if tokens > model_limit:
        container.update(label = f"Warning: This video's token usage of {tokens} tokens exceeds the limit of {model_limit} for model {model}. Please switch your model to one that can handle more tokens.", state="error", expanded=False)
    else:
        container.update(label = f"Token Usage within Model Limit", state="running", expanded=True)

    try:
        response = client.chat.completions.create(
            model=model,
            #response_format={ "type": "json_object" },
            messages=chat_sequence
        )
        gpt_response = response.choices[0].message.content
    except:
        st.error(f"Your video exceeds the token limit for your selected model. To analyze this video you need a model that is capable of handling ({tokens}) tokens.")
    return gpt_response

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def keyword(keywords):
    keywords_list = [keyword.strip() for keyword in keywords.split(',')]

    # Count occurrences of each unique keyword
    keyword_counts = {keyword: keywords_list.count(keyword) for keyword in set(keywords_list)}

    # Create a dataframe from the results
    df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Occurrences'])
    st.dataframe(df)

def split_video(input_file, output_prefix, num_parts):
    video_clip = VideoFileClip(input_file)
    total_duration = video_clip.duration
    part_duration = total_duration / num_parts

    output_files = []

    for part_number in range(num_parts):
        start_time = part_number * part_duration
        end_time = min((part_number + 1) * part_duration, total_duration)

        subclip = video_clip.subclip(start_time, end_time)

        output_file = f"tmp/{output_prefix}_part{part_number + 1}.mp4"
        output_files.append(output_file)
        subclip.write_videofile(output_file, codec="libx264", audio_codec="aac")

    video_clip.close()

    return output_files

def display_split_files(split_files):
    st.write("### Split Files:")
    for file_path in split_files:
        st.markdown(f"**{file_path}**")
        st.video(file_path)

def transcribe2(video_file):
    model = whisper.load_model('base')
    transcript = model.transcribe(video_file)
    return transcript["text"]

def transcriber(video_file_path, client):
    print(video_file_path)
    # Check if the file exists
    if not os.path.exists(video_file_path):
        raise FileNotFoundError("The specified video file does not exist.")
    
    # Get the size of the video file
    file_size = os.path.getsize(video_file_path) / (1024 * 1024)
    
    # Set the threshold for splitting
    threshold = 20  # MB
    
    if file_size >= threshold:
        st.info(
            "One of the files you have uploaded is too big for current model capabilities. Attempting to split file. This may take a few minutes."
        )
        # Split the video file into parts
        input_path = "tmp/uploaded_video.mp4"
        with open(video_file_path, "rb") as f:
            bytes_data = f.read()
        with open(input_path, "wb") as f:
            f.write(bytes_data)
        split_files = split_video(input_path, "output_part", math.ceil(file_size / threshold))
        st.success("Splitting complete. Check the generated files.")
        st.session_state.split_files = split_files
        # Transcribe each part separately
        transcript = ""
        for file in split_files:
            audio_temp_file = convert_to_audio(file)
            temp_transcript = transcribe(file,client)
            transcript += temp_transcript
    else:
        # Convert video to audio and transcribe
        audio_temp_file = convert_to_audio(video_file_path)
        transcript = transcribe(audio_temp_file,client)
    return transcript

def convert_to_audio(video_file_path):
    # Read the video file
    video_clip = mp.VideoFileClip(video_file_path)
    # Create a temporary file to store the audio
    audio_temp_file = "tmp/temp_audio.wav"
    # Write the audio to the temporary file
    video_clip.audio.write_audiofile(audio_temp_file)
    video_clip.close()
    return audio_temp_file

def split_video(video_file_path, output_prefix, num_parts):
    video = mp.VideoFileClip(video_file_path)
    duration_per_part = video.duration / num_parts

    split_files = []
    for i in range(num_parts):
        start_time = i * duration_per_part
        end_time = min((i + 1) * duration_per_part, video.duration)
        subclip = video.subclip(start_time, end_time)
        output_path = f"{output_prefix}_{i}.mp4"
        subclip.write_videofile(output_path, codec="libx264")
        split_files.append(output_path)
    return split_files

def main():
    st.title("Automatic Misinformation Analysis")
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    if "check_done" not in st.session_state:
        st.session_state["check_done"] = False

    if not st.session_state["check_done"]:
        # Display input for OpenAI API key
        st.session_state["api_key"] = st.text_input('Enter the OpenAI API key', type='password')
    
    if st.session_state["api_key"]:
        # On receiving the input, perform an API key check
        client = OpenAI(api_key=st.session_state["api_key"])
        if is_api_key_valid(st.session_state["api_key"],client):
            # If API check is successful
            st.success('OpenAI API key check passed')
            st.session_state["check_done"] = True
            client = OpenAI(api_key=st.session_state["api_key"])
        else:
            # If API check fails
            st.error('OpenAI API key check failed. Please enter a correct API key')
    
    if st.session_state["check_done"]:
        # Display model selection after API key check passed
        selected_model = get_model_selection()

     #Adding url scraper section
    url= st.text_input("Enter URL", "").split(',')
    if url:
        for singleurl in url:
            if singleurl:
                if tiktok_validation(singleurl):
                    run_tiktok_scraper(singleurl)
                if youtube_validation(singleurl):
                    run_youtube_scraper(singleurl)
                else:
                    st.warning("Please enter a URL")

    if st.button('Transcribe and Analyze Videos'):
        data = []
        keywords = ""
        container_array = []
        folder_path = "tmp"
        for i, filename in enumerate(os.listdir(folder_path)):
            if filename.endswith(".mp4"):
                container_name = f"container{i+1}"
                container_array.append(container_name)
                video_file_path = os.path.join(folder_path, filename)
                container_array[i] = st.status(f"Processing ({filename})")
                time.sleep(0.7)
                with container_array[i] as container:
                    container.update(label=f"Transcribing ({filename})...", state="running", expanded=False)
                    transcript = transcriber(video_file_path, client)
                    st.markdown(filename + ": " + transcript)
                    container.update(label=f"Transcribed ({filename})", state="running", expanded=True)
                    time.sleep(0.8)
                    container.update(label=f"Analyzing ({filename})...", state="running", expanded=True)
                    analysis = analyze(transcript, selected_model, client, container)
                    st.markdown(analysis)
                    analysis = analysis.split('\n')
                    misinformation_status = analysis[0]
                    keywords_index = [i for i, element in enumerate(analysis) if 'Keywords:' in element]
                    if keywords_index:
                        video_keywords = analysis[keywords_index[0]]
                        video_keywords = video_keywords.split("Keywords:")[1].strip()
                        print(video_keywords)
                        keywords += video_keywords
                        print(video_keywords)
                        keyword(video_keywords)
                    else:
                        video_keywords = "Not found"
                        d = {
                        "video_file": filename,
                        "transcript": transcript, 
                        "misinformation_status": misinformation_status, 
                        "keywords": video_keywords
                        }
                        data.append(d)
                        container.update(label=f'{filename}', state="complete", expanded=False)
            st.session_state.data = data
            df = pd.DataFrame(data)
            keyword(keywords)
            csv = convert_df(df)
            date_today_with_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name = f'results_{date_today_with_time}.csv',
                mime='text/plain')
if __name__ == "__main__":
    main()