import streamlit as st
import torch
import os
import json
from datetime import datetime
import logging
from experiment_data import ExperimentData,load_t2i_model,setup_seed,generate_images  # 确保这个类在experiment_data.py文件中
import openai
import sys
from random import choice

os.environ['http_proxy'] = 'http://127.0.0.1:7891'
os.environ['https_proxy'] = 'http://127.0.0.1:7891'
keys = [
    "OPENAI-TOKEN" #OPENAI-TOKEN
]
openai.api_key= choice(keys)
torch.cuda.set_device(4)
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
st.title(localize('Initialization Page', '初始化页面'))

# @st.cache(allow_output_mutation=True,suppress_st_warning=True)  # 允许缓存的输出可变
def Initialization():
    # user_files = get_existing_user_files(save_path)
    # if not user_files:
    #     st.info('No initialization data files found.')
    
    # if 'initialized' not in st.session_state:
    #     st.session_state['initialized'] = False  # 用于跟踪是否已完成初始化
    
    if not st.session_state['initialized']:
        # 表单输入
        with st.form(key='init_form'):
            st.session_state['username'] = st.text_input(localize('Username', '用户名'),value="mxy")
            st.session_state['contact'] = st.text_input(localize('Contact Information', '联系方式'),value="mxy")
            st.session_state['test_theme']  = st.text_input(localize('Test Theme', '测试主题'),value="cat")
            options  = ['sd1_4','sd1_5','sd2_1','sdxl','sdxl_b']
            st.session_state['tested_model']  = st.selectbox(localize('Tested Model', '被测模型'),options,index=1)
            st.session_state['max_test_depth']  = st.number_input(localize('Max Test Depth', '最大测试深度'), min_value=1,value=3)
            st.session_state['topics_per_test']  = st.number_input(localize('Number of Topics per Test', '每次生成的测试topic数量'), min_value=1,value=10)
            st.session_state['test_input_count']  = st.number_input(localize('Number of Test Inputs', '测试Input数量'), min_value=1,value=10)
            st.session_state['image_count']  = st.number_input(localize('Number of Images Generated', '生成图片数量'), min_value=1,value=4)

            submit_button = st.form_submit_button(label=localize('Submit', '提交'))
            
        
            
        # 用户提交表单后的操作
        if submit_button and st.session_state['username']:
            print('hello world')
            user_folder = os.path.join(save_path, st.session_state['username'])
            user_file = os.path.join(user_folder, f"{st.session_state['username']}.json")


            # 检查用户名文件夹是否存在，如果不存在则创建
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            # 准备要保存的数据
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

            # 如果json文件存在，则读取内容并追加记录；如果不存在，则创建新文件
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
            st.session_state['user_records_file_path']= os.path.join(user_folder, f"{st.session_state['username']}+_+{st.session_state['test_theme']}.json")
            
            st.session_state['model'] = load_t2i_model(model_name=st.session_state['tested_model'], 
                        model_path='/data2/mixiaoyue/mixiaoyue/diffusion_models/stable-diffusion-v1-5',
                        device = 'cuda'
                        )
            
            st.session_state['test_inputs'] = None
            
            st.session_state['initialized'] = True
            print(st.session_state['initialized'])
        else:
            st.warning(localize('Please enter a username to continue.','请先输入一个用户名才能继续'))
            e = RuntimeError('This is an exception of type RuntimeError')
            st.exception(e)

    
# Streamlit应用界面
def app(lang):
    st.title("Image Generation Testing / 图像生成测试")

    
    # 根据选择的语言显示文本
    if lang == "English / 英文":
        topic_text = "Select Test Topic"
        input_text = "Select Test Input"
        image_section = "Generated Images"
        annotation_section = "Image Annotation"
        next_iteration_button = "Next Iteration"
    else:
        topic_text = "选择测试主题"
        input_text = "选择测试输入"
        image_section = "生成的图像"
        annotation_section = "图像标注"
        next_iteration_button = "下一轮迭代"

    # 选择测试主题
    

    # hzy：显示所有可选topics，father、current和children
    # hzy：这里是否需要维护一个test_topics？还是说所有的topic放进一个list就行
    # print(st.session_state['experiment_data'].test_topics)

    
    # hzy：感觉先选择再生成更有逻辑→我错了
    if st.button("Test Topic Generation"):
        # 根据当前test records生成topic
        st.session_state['experiment_data'].generate_test_topics(num_topics=st.session_state['topics_per_test'])


    topic = st.selectbox(topic_text, st.session_state['experiment_data'].test_topics)
    # hzy：这里是用户选择一个topic。考虑到重选topic的问题，需要初始化test_inputs
    if topic != st.session_state['experiment_data'].current_topic:
        st.session_state['experiment_data'].current_topic=topic
        st.session_state['test_inputs'] = None

    st.text('Current Topic is: {}'.format(st.session_state['experiment_data'].current_topic))
    st.text('Father Topic is: {}'.format(st.session_state['experiment_data'].test_topics[st.session_state['experiment_data'].current_topic]['Father_topic']))


    # 显示测试输入
    # hzy：点击按钮后按当前topic进行生成input，不点击就不生成
    if st.button("Test Input Generation"):
        st.session_state['test_inputs'] = st.session_state['experiment_data'].generate_test_inputs(topic,num_inputs=st.session_state['test_input_count'])
        # test_inputs = experiment_data.test_inputs[topic]
    # hzy：选择一个input
    selected_input = None
    # if 'test_inputs' not in st.session_state: # hzy：这个应该不需要了，因为上面会对test_inputs置None。想一想好像还是会的。我改成了在initializtion里会置None
    #     selected_input = st.selectbox(input_text, st.session_state['test_inputs'])
    if st.session_state['test_inputs'] is not None: # hzy：如果没有test_inputs，那么需要选一个
        selected_input = st.selectbox(input_text, st.session_state['test_inputs'])

    # hzy：有select_input才允许进行图像生成与标注
    if selected_input is not None:
        # 图像生成和标注部分（这里需要进一步的实现）
        st.subheader(image_section)
        # 这里可以集成图像生成模型，生成图像并展示
        print(save_path, st.session_state['username'], topic)
        images_save_path =os.path.join(save_path, st.session_state['username'],topic)
        print("190", images_save_path)
        if not os.path.exists(images_save_path):
            os.mkdir(images_save_path)
        # 生成图片
        if st.button("Image Generation"):
            # hzy：现在同一prompt每次生成的结果都是相同的，后续需要修改
            images_generated = generate_images(st.session_state['model'],st.session_state['image_count'],selected_input,images_save_path)
                
                # 显示生成的图像并提供标注选项
            with st.form(key='annotation_form'):
                st.subheader(annotation_section)
                annotations = {}
                for idx, image in enumerate(images_generated):
                    # 显示图像
                    st.image(image, caption=f"Image {idx+1}", use_column_width=True)
                    # 为每张图像创建一个单选按钮组
                    # label = st.radio(f"Label for Image {idx+1}:", ('Pass', 'Fail', 'Off-topic'), key=f"label_{idx}")
                    label = st.radio(f"Label for Image {idx+1}:", ('Pass', 'Fail', 'Off-topic'), key=f"label_{idx}")
                    annotations[f"image_{idx+1}"] = label

                # 提交按钮
                submitted = st.form_submit_button("Submit Annotations")
                if submitted:
                    acc = sum(1 for label in annotations.values() if label == "Pass")/st.session_state['image_count']
                    # 将标注添加到experiment_data中
                    for idx, label in annotations.items():
                        st.session_state['experiment_data'].add_image_annotations(topic, selected_input, st.session_state['image_count'], label)
                    st.success(f"Annotations submitted successfully! Pass Rate: {acc}")
                    # st.session_state['experiment_data'].add_image_annotations(topic, selected_input, acc)
                    st.session_state['experiment_data'].add_image_pass_rate(topic, selected_input, acc) # hzy
                    # 处理其他逻辑，比如保存数据或者准备下一个迭代的测试数据
                # st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

                # # 用户对生成的图像进行标注
                # st.subheader(annotation_section)
                # # 提供标注的选项，例如：通过（pass）、失败（fail）或偏题（off-topic）
                # acc=0
                # for image in images_generated:
                #     label = st.radio(f"Label for {image}:", ('Pass', 'Fail'), key=f"label_{selected_input}_{image}")
                #     if label =="Pass":
                #         acc=acc+1
                #     experiment_data.add_image_annotations(topic, selected_input,st.session_state['image_count'], label)

    # 反思与迭代
    if st.button(next_iteration_button):
        st.success("Annotations submitted and recorded.")
        st.balloons()
        st.session_state['experiment_data'].next_iteration(num_topics=st.session_state['topics_per_test'])

    # 记录实验过程
    # 这部分可以添加代码以记录用户的选择和标注


if __name__ == "__main__":
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = False  # 用于跟踪是否已完成初始化
    if not st.session_state['initialized']:
        print(st.session_state['initialized'])
        Initialization()
        
        if st.session_state['initialized']:
            # 如果有用户名，继续执行其他部分
            st.success(localize('Thank you for entering your username, you may proceed.', '感谢您的输入，现在您可以继续操作了'))

            # 在此处添加应用的其余部分...
            #标注页面
            # 设置日志记录
            log_save_file = st.session_state['username']+"_"+st.session_state['test_theme']+"_max_test_depth_"+str(st.session_state['max_test_depth'])+"_topics_per_test_"+str(st.session_state['topics_per_test'])+"_test_input_count_"+str(st.session_state['test_input_count'])+"_image_count_"+str(st.session_state['image_count'])+'_app.log'
            # logging.basicConfig(filename=log_save_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger = set_logger(log_save_file)
            # 随机生成Topic和Test Input的函数
            # load the model,此处还没有修改
            # import pdb; pdb.set_trace()
            print(st.session_state['user_records_file_path'], st.session_state['test_theme'])
            st.session_state['experiment_data']= ExperimentData(st.session_state['user_records_file_path'],st.session_state['test_theme'],logger,openai.api_key) 
            st.session_state['experiment_data'].generate_test_topics(num_topics=st.session_state['topics_per_test'])
            # print('i am free!')

    if st.session_state['initialized']:
        app(language)
    # print('ok')