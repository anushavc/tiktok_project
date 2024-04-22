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


def transcribe(video_file,client):
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=video_file, 
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

        output_file = f"{output_prefix}_part{part_number + 1}.mp4"
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

def transcriber(video_file, client):
    bytes_data = video_file.getvalue()
    file_size = len(bytes_data)/ (1024 * 1024)
    threshold = 20 * 1024 * 1024
    num_parts = math.ceil(file_size/threshold)
    print(file_size)
    if file_size >= 20:
        st.info("One of the files you have uploaded is too big for current model capabilities. Attempting to split file. This may take a few minutes.")
        input_path = f"uploaded_video.mp4"
        with open(input_path, 'wb') as f:
            f.write(video_file.read())
        split_files = split_video(input_path, "output_part", num_parts)
        st.success("Splitting complete. Check the generated files.")
        st.session_state.split_files = split_files
        transcript = ""
        for file in split_files:
            temp_transcript = transcribe2(file)
            transcript += temp_transcript
    else:
        temp_video_file = video_file
        transcript = transcribe(temp_video_file, client)
    return transcript


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

    video_files = st.file_uploader('Upload Your Video File', type=['wav', 'mp3', 'mp4'], accept_multiple_files=True)

    if st.button('Transcribe and Analyze Videos'):
        data = []
        keywords = ""
        container_array = []
        for i in range(1, len(video_files) + 1):
            container_name = f"container{i}"
            container_array.append(container_name)
        for i, video_file in enumerate(video_files):
            container_array[i] = st.status(f"Processing ({video_file.name})")
            time.sleep(0.7)
            with container_array[i] as container:
                if video_file is not None:
                    container.update(label = f"Transcribing ({video_file.name})...", state="running", expanded=False)
                    transcript = transcriber(video_file, client)
                    st.markdown(video_file.name + ": " + transcript)
                    container.update(label=f"Transcribed ({video_file.name})", state="running", expanded=True)
                    time.sleep(0.8)
                    container.update(label = f"Analyzing ({video_file.name})...", state="running", expanded=True)
                    analysis = analyze(transcript, selected_model, client, container)
                    st.markdown(analysis)
                    analysis = analysis.split('\n')
                    misinformation_status = analysis[0]
                    keywords_index = [i for i, element in enumerate(analysis) if 'Keywords:' in element]
                    if keywords_index:
                        video_keywords = analysis[keywords_index[0]]
                        video_keywords = video_keywords.split("Keywords:")[1]
                        print(video_keywords)
                        keywords += video_keywords
                        print(video_keywords)
                        keyword(video_keywords)
                    else:
                        video_keywords = "Not found"
                    #reasons = ' '.join(analysis[4:])
                    d = {
                        "video_file": video_file.name,
                        "transcript": transcript, 
                        "misinformation_status": misinformation_status, 
                        "keywords": video_keywords, 
                        #"reasons": reasons
                    }
                    data.append(d)
                    #print('"{}"'.format(transcript)    
                else:
                    st.sidebar.error("Please upload a video file")
                container.update(label=f'{video_file.name}', state="complete", expanded=False)
        st.session_state.data = data
        df = pd.DataFrame(data)
        keyword(keywords)
        csv = convert_df(df)
        date_today_with_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name = f'results_{date_today_with_time}.csv',
            mime='text/plain',
        )
if __name__ == "__main__":
    main()