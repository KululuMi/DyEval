import streamlit as st
import torch
import os
import json
from datetime import datetime
import logging
from experiment_data import ExperimentData, load_t2i_model, setup_seed, generate_images, MaxDepthException  # 确保这个类在experiment_data.py文件中
import openai
import sys

import time

os.environ['http_proxy'] = 'http://127.0.0.1:7891'
os.environ['https_proxy'] = 'http://127.0.0.1:7891'
keys = [
    "OPENAI-TOKEN" 
]
openai.api_key = keys[0]

torch.cuda.set_device(7)
setup_seed(20)
save_path = 'save'

# 获取已存在的用户文件列表
def get_existing_user_files(save_path):
    user_files = []
    for root, dirs, files in os.walk(save_path):
        for file in files:
            if file.endswith('.json'):
                user_files.append(os.path.join(root, file))
    return user_files

# 初始化实验数据
def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# 记录一个事件的函数
def log_event(event_description):
    logging.info(event_description)

# 函数：根据选择的语言显示文本
def localize(text_en, text_cn):
    if language == 'English':
        return text_en
    else:
        return text_cn

# 选择语言
language = st.sidebar.selectbox(
    '选择语言 / Choose Language',
    ('中文', 'English')
)

# 设置标题

def Initialization():
    st.title(localize('Initialization Page', '初始化页面'))

    if not st.session_state['initialized']:
        with st.form(key='init_form'):
            st.session_state['username'] = st.text_input(localize('Username', '用户名'), value="xiaosun")
            st.session_state['contact'] = st.text_input(localize('Contact Information', '联系方式'), value="mxy")
            st.session_state['test_theme'] = st.text_input(localize('Test Theme', '测试主题'), value="different style in images")
            options = ['sd1_4', 'sd1_5', 'sd2_1', 'sdxl', 'sdxl_b', 'sd3']
            st.session_state['tested_model'] = st.selectbox(localize('Tested Model', '被测模型'), options, index=5)
            st.session_state['max_test_depth'] = st.number_input(localize('Max Test Depth', '最大测试深度'), min_value=1, value=3)
            st.session_state['topics_per_test'] = st.number_input(localize('Number of Topics per Test', '每次生成的测试topic数量'), min_value=1, value=3)
            st.session_state['test_input_count'] = st.number_input(localize('Number of Test Inputs', '测试Input数量'), min_value=1, value=5)
            st.session_state['image_count'] = st.number_input(localize('Number of Images Generated', '生成图片数量'), min_value=1, value=4)

            submit_button = st.form_submit_button(label=localize('Submit', '提交'))

        if submit_button and st.session_state['username']:
            user_folder = os.path.join(save_path, st.session_state['username'])
            user_file = os.path.join(user_folder, f"{st.session_state['username']}.json")

            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            record = {
                'username': st.session_state['username'],
                'contact': st.session_state['contact'],
                'test_theme': st.session_state['test_theme'],
                'tested_model': st.session_state['tested_model'],
                'max_test_depth': st.session_state['max_test_depth'],
                'topics_per_test': st.session_state['topics_per_test'],
                'test_input_count': st.session_state['test_input_count'],
                'image_count': st.session_state['image_count'],
                'created_at': datetime.now().isoformat()
            }

            if os.path.exists(user_file):
                with open(user_file, 'r+', encoding='utf-8') as file:
                    data = json.load(file)
                    data.append(record)
                    file.seek(0)
                    json.dump(data, file, ensure_ascii=False, indent=4)
            else:
                with open(user_file, 'w', encoding='utf-8') as file:
                    json.dump([record], file, ensure_ascii=False, indent=4)

            st.success(localize('Information submitted successfully!', '信息已提交成功！'))
            st.session_state['user_records_file_path'] = os.path.join(user_folder, f"{st.session_state['username']}_{st.session_state['test_theme']}_{st.session_state['tested_model']}.json")

            # 确保模型只加载一次
            if 'model' not in st.session_state:
                st.session_state['model'] = load_t2i_model(model_name=st.session_state['tested_model'], 
                                                           model_path='/data2/mixiaoyue/mixiaoyue/diffusion_models/stable-diffusion-v1-5',
                                                           device='cuda')

            st.session_state['test_inputs'] = None
            st.session_state['MiniInput_state'] = False

            st.session_state['initialized'] = True
        else:
            st.warning(localize('Please enter a username to continue.', '请先输入一个用户名才能继续'))
            e = RuntimeError('This is an exception of type RuntimeError')
            st.exception(e)

def annotation():
    pass

def get_index(inp_list, key):
    for i, k in enumerate(inp_list):
        if key == k:
            return i
    return 0

def app(lang):
    st.title("Image Generation Testing")

    topic_text = "Select Test Topic"
    input_text = "Select Test Input"
    image_section = "Generated Images"
    annotation_section = "Image Annotation"
    next_iteration_button = "Next Iteration"
    miniinput_finish = 'MiniInput Finish, Please GO ON!'
    Waiting_Image_Generation = 'Please wait for Image Generation'

    if not st.session_state['MiniInput_state']:
        if st.button("Test Topic Generation"):
            try:
                st.session_state['experiment_data'].generate_test_topics(num_topics=st.session_state['topics_per_test'])
            except MaxDepthException as e:
                st.success('该topic已经达到最大深度')

        current_topics, test_topics_offline = st.session_state['experiment_data'].return_current_topics(offline=True)
        current_topic_index = get_index(test_topics_offline, st.session_state['experiment_data'].current_topic)

        topic = st.selectbox(topic_text, current_topics, index=current_topic_index)
        topic = topic.replace("* ", "").replace("\"", "")
        if topic != st.session_state['experiment_data'].current_topic:
            st.session_state['experiment_data'].current_topic = topic
            st.session_state['test_inputs'] = None

        st.text('Current Topic is: {}'.format(st.session_state['experiment_data'].current_topic))
        st.text('Father Topic is: {}'.format(st.session_state['experiment_data'].test_topics[st.session_state['experiment_data'].current_topic]['Father_topic']))
        st.text('Child Topic is: {}'.format(st.session_state['experiment_data'].test_topics[st.session_state['experiment_data'].current_topic]['Child_topics']))
        st.text('Current Depth is: {}'.format(st.session_state['experiment_data'].test_topics[st.session_state['experiment_data'].current_topic]["Depth"]))

        if st.button("Test Input Generation"):
            st.session_state['test_inputs'] = st.session_state['experiment_data'].generate_test_inputs(topic, num_inputs=st.session_state['test_input_count'])
            st.rerun()

        selected_input = None
        current_test_inputs, current_test_inputs_offline = st.session_state['experiment_data'].return_current_inputs(st.session_state['experiment_data'].current_topic, offline=True)
        if current_test_inputs is not None:
            if 'current_test_input' in st.session_state:
                current_test_input_index = get_index(current_test_inputs_offline, st.session_state['current_test_input'])
            else:
                current_test_input_index = 0
            st.session_state['test_inputs'] = current_test_inputs_offline
            selected_input = st.selectbox(input_text, current_test_inputs, index=current_test_input_index)
            if selected_input is not None:
                st.session_state['current_test_input'] = selected_input.replace("* ", "").replace("\"", "")
            st.text('Current Text Input is: {}'.format(st.session_state['current_test_input']))

        if selected_input is not None:
            st.subheader(image_section)
            images_save_path = os.path.join(save_path, st.session_state['username'], topic, st.session_state['tested_model'])
            if not os.path.exists(images_save_path):
                os.makedirs(images_save_path, exist_ok=True)

            # 只生成图像一次，并存储在 session_state 中
            if selected_input not in st.session_state :
                with st.spinner(Waiting_Image_Generation):
                    st.session_state['images_generated'] = generate_images(st.session_state['model'], st.session_state['image_count'], selected_input, images_save_path)
                    st.session_state["* "+selected_input]=True
                    print(selected_input)
                    # 清理未使用的显存
                    torch.cuda.empty_cache()

            images_generated = st.session_state['images_generated']
            with st.form(key='annotation_form'):
                st.subheader(annotation_section)
                annotations = {}
                for idx, image in enumerate(images_generated):
                    st.image(image, caption=f"Image {idx+1}", use_column_width=True)
                    a = time.time()
                    label = st.radio(f"Label for Image {idx+1}:", ('Pass', 'Fail', 'Off-topic'), key=f"label_{idx}")
                    a_after = time.time() - a
                    annotations[f"image_{idx+1}"] = {"label": label, "image": image, "time": a_after}

                submitted = st.form_submit_button("Submit Annotations")
                if submitted:
                    acc = sum(1 for label in annotations.values() if label['label'] == "Pass") / st.session_state['image_count']
                    for idx, annotation_ in annotations.items():
                        st.session_state['experiment_data'].add_image_annotations(topic, selected_input, idx, annotation_["label"], annotation_["image"], annotation_["time"])
                    st.success(f"Annotations submitted successfully! Pass Rate: {acc}")
                    st.session_state['experiment_data'].add_image_pass_rate(topic, selected_input, acc)
                    st.session_state['experiment_data'].save_to_file()

                    if acc < 0.75:
                        st.session_state['MiniInput_state'] = True
                        st.session_state['miniinput_process_state'] = 0
                    st.rerun()

    else:  # MiniInput
        ed = st.session_state['experiment_data']
        if st.session_state['miniinput_process_state'] == 0:
            st.session_state['MiniInput'] = ed.minimize_input_generation(st.session_state['current_test_input'])
            st.session_state['miniinput_process_state'] = 1
            st.rerun()

        if st.session_state['miniinput_process_state'] == 1:
            miniinput = next(st.session_state['MiniInput'])
            st.session_state['miniinput_texts'] = miniinput
            st.session_state['miniinput_process_state'] = 2
            st.rerun()

        if st.session_state['miniinput_process_state'] == 2:
            miniinput = st.session_state['miniinput_texts']
            topic = st.session_state['experiment_data'].current_topic
            images_save_path = os.path.join(save_path, st.session_state['username'], topic, st.session_state['tested_model'])
            os.makedirs(images_save_path, exist_ok=True)

            images_generated = {}
            with st.spinner(Waiting_Image_Generation):
                for m_input in miniinput:
                    images_generated[m_input] = generate_images(st.session_state['model'], st.session_state['image_count'], m_input, images_save_path)
                     # 清理未使用的显存
                    torch.cuda.empty_cache()

            with st.form(key='annotation_form'):
                st.subheader(annotation_section)
                annotations = {}
                for m_idx, m_input in enumerate(miniinput):
                    annotations[m_input] = {}
                    images = images_generated[m_input]
                    for idx, image in enumerate(images):
                        st.caption(m_input)
                        st.image(image, caption=f"{m_idx} Image {idx+1}", use_column_width=True)
                        a = time.time()
                        label = st.radio(f"Label for Image {idx+1}:", ('Pass', 'Fail', 'Off-topic'), key=f"{m_idx}_label_{idx}")
                        a_after = time.time() - a
                        annotations[m_input][f"image_{idx+1}"] = {"label": label, "image": image, "time": a_after}

                submitted = st.form_submit_button("Submit Annotations")
                if submitted:
                    labels = []
                    for m_input in miniinput:
                        acc = sum(1 for label in annotations[m_input].values() if label['label'] == "Pass") / st.session_state['image_count']
                        for idx, annotation_ in annotations[m_input].items():
                            st.session_state['experiment_data'].add_image_annotations(topic, m_input, idx, annotation_["label"], annotation_["image"], annotation_["time"])
                        st.success(f"Annotations submitted successfully! Pass Rate: {acc}")
                        st.session_state['experiment_data'].add_image_pass_rate(topic, m_input, acc)
                        labels.append(acc >= 0.75)
                    st.session_state['experiment_data'].save_to_file()

                    try:
                        miniinput = st.session_state['MiniInput'].send(labels)
                    except Exception as e:
                        st.session_state['miniinput_process_state'] = 3
                        st.rerun()

                    st.session_state['miniinput_texts'] = miniinput
                    st.rerun()

        if st.session_state['miniinput_process_state'] == 3:
            st.balloons()
            if st.button(miniinput_finish):
                st.session_state['MiniInput_state'] = False
                st.rerun()

if __name__ == "__main__":
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = False

    if not st.session_state['initialized']:
        Initialization()
        
        if st.session_state['initialized']:
            st.success(localize('Thank you for entering your username, you may proceed.', '感谢您的输入，现在您可以继续操作了'))
            log_save_file = st.session_state['username'] + "_" + st.session_state['test_theme'] + "_max_test_depth_" + str(st.session_state['max_test_depth']) + "_topics_per_test_" + str(st.session_state['topics_per_test']) + "_test_input_count_" + str(st.session_state['test_input_count']) + "_image_count_" + str(st.session_state['image_count']) + '_app.log'
            logger = set_logger(log_save_file)
            st.session_state['experiment_data'] = ExperimentData(st.session_state['user_records_file_path'], st.session_state['test_theme'], logger, openai.api_key)
            if os.path.exists(st.session_state['user_records_file_path']):
                st.session_state['experiment_data'].load_from_file()
            else:
                st.session_state['experiment_data'].generate_test_topics(num_topics=st.session_state['topics_per_test'])

            st.session_state['experiment_data'].max_test_depth = st.session_state['max_test_depth']

    if st.session_state['initialized']:
        app(language)
