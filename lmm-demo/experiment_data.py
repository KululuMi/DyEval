import json
import random
import openai
import os
import pdb
import torch
from diffusers import DiffusionPipeline,StableDiffusionXLPipeline
from diffusers.utils import load_image
import torch
import safetensors

import numpy as np
import time
import re
import string
import logging
import sys
import open_clip
from random import choice
def Initialize_clip(clip_model_name,clip_pretrain,device):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrain, device=device, jit=True)
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    return tokenizer,clip_preprocess,clip_model

def get_text_img_similarity(init_input, generated_image,device,tokenizer,clip_preprocess,clip_model):
    text = tokenizer([init_input]).to(device)

    gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [generated_image]]
    gen_batch = torch.concatenate(gen_batch).to(device)

    gen_feat = clip_model.encode_image(gen_batch)
    text_feat = clip_model.encode_text(text)
    
    gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)

    return (gen_feat @ text_feat.t()).mean().item()



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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_t2i_model(model_name='sd1_5', model_path='/data2/mixiaoyue/mixiaoyue/diffusion_models/stable-diffusion-v1-5',device = 'cuda', **model_kwargs):
    #TODO: Adding different hyter-parameter records

    if model_name  not in ['sd1_4','sd1_5','sd2_1','sdxl','sdxl_b','sd3']:
        print("ERROR: Unknown model! ")
        return
    else:
        if model_name =='sd1_5':
            # torch.cuda.set_device(0)
            pipe = DiffusionPipeline.from_pretrained("/data2/mixiaoyue/mixiaoyue/diffusion_models/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None,use_safetensors=False, variant="fp16")
            pipe.to(device)
            # model_kwargs['torch_dtype'] ="torch.float16" 
            # model_kwargs['use_safetensors'] ="False"
            model_kwargs['variant'] ="fp16"
        if model_name =='sdxl':
            # torch.cuda.set_device(1)
            pipe = StableDiffusionXLPipeline.from_pretrained(
    "/data/huangziyao/projects/diffusion/diffusers/hzy_examples/good_weights/stable-diffusion-xl-base-1.0/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to(device)
        if model_name =='sd2_1':
            # torch.cuda.set_device(2)
            pipe = DiffusionPipeline.from_pretrained("/data2/mixiaoyue/mixiaoyue/Anti-DreamBooth/stable-diffusion-2-1-base", torch_dtype=torch.float16, safety_checker=None,use_safetensors=False, variant="fp16")
            pipe.to(device)
            # model_kwargs['torch_dtype'] ="torch.float16" 
            # model_kwargs['use_safetensors'] ="False"
            model_kwargs['variant'] ="fp16"
        if model_name =='sd3':
           
            from diffusers import StableDiffusion3Pipeline

            pipe = StableDiffusion3Pipeline.from_pretrained("/data3/wuyou/wuyou/workspace/SD_models_checkpoints/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
            pipe = pipe.to(device)

            # image = pipe(
            #     "A cat holding a sign that says hello world",
            #     negative_prompt="",
            #     num_inference_steps=28,
            #     guidance_scale=7.0,
            # ).images[0]
            # image.save("./output_sd3/test.png")



        # if using torch < 2.0
        # pipe.enable_xformers_memory_efficient_attention()

        return pipe

def replace_punctuation_and_spaces(lst):
    # Create a translation table: map punctuation to underscore
    trans_table = str.maketrans(string.punctuation + ' ', '_' * (len(string.punctuation) + 1))
    return [s.translate(trans_table) for s in lst]
def generate_images(image_pipe,num_images_per_prompt,prompt,save_path):
    import hashlib
    import random
    random.seed(123)
    image_paths = []
    if type(prompt) is not list:
        prompt = [prompt]
    prompt = replace_punctuation_and_spaces(prompt)
    skip = [False] * num_images_per_prompt
    for i in range(num_images_per_prompt):
        image_name = prompt[0]+str(int(i%num_images_per_prompt))
        image_name = hashlib.md5(image_name.encode())
        image_name = image_name.hexdigest()

        image_path = os.path.join(save_path, image_name + ".png")
        image_paths.append(image_path)
        if os.path.exists(image_path):
            skip[i] = True
    skip_count = sum(skip)

    if skip_count >=num_images_per_prompt:
        out_images = []
    else:
        # import pdb;pdb.set_trace()
        #sd1-5/2-1/xl
        # out_images = image_pipe(prompt,num_inference_steps=25, num_images_per_prompt=num_images_per_prompt,safety_checker=False).images
        #sd3
        out_images = image_pipe(prompt, negative_prompt=[""] * len(prompt),num_inference_steps=28,guidance_scale=7.0,num_images_per_prompt=num_images_per_prompt).images

    # for i, out_image in enumerate(out_images):
    oid=0
    for i, image_path in enumerate(image_paths):
        # image_path = os.path.join(save_path,prompt[0]+str(int(i%num_images_per_prompt)) + ".png")
        # print(image_path)
        # image_path = image_paths[i]
        # image_paths.append(image_path)
        if not skip[i]:
            out_image = out_images[oid]
            oid+=1
            out_image.save(image_path)
    return image_paths

def post_message(messages, tokens, logger):
    # 调用接口

    # openai.api_key="YOUR_OPENAI_API_KEY_HERE"    
    stime = 5

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )  

    tokens += response['usage']['total_tokens']
    messages.append({'role': response['choices'][0]['message']['role'], 'content': response['choices'][0]['message']['content']})
    time.sleep(stime)

    return messages, response['choices'][0]['message']['content'], tokens

def generate_test_inputs(num,is_input):
    test_inputs = []
    
    for i in range(1, num+1):
        if is_input:
            test_input = f"[Test Input]: <OUTPUT{i}>"
        else:
            test_input = f"[Next Test Topic]: <OUTPUT{i}>"
        test_inputs.append(test_input)
    return "\n".join(test_inputs)

def test_topics_to_prompts(topics):
    output_string = ""
    for i, item in enumerate(topics):
        output_string += f"[Next Test Topic]: '{item}'"
        if i != len(topics) - 1:
            output_string += ",\n"

    formatted_output = f"""\n{output_string}\n"""
    return formatted_output
#目前是全部测试记录都拿来了
def format_records_output(test_records, test_inputs, image_annotations):
    output_string = ""
    for topic, topic_data in test_records.items():
        output_string += f"**Test Records: {topic}**\n"
        child_topics = topic_data.get("Child_topics", [])
        
        for child_topic in child_topics:
            output_string += f"***Test Topic***: \"{child_topic}\".\n"
            inputs = test_inputs.get(child_topic, [])
            
            for input_text in inputs:
                output_string += f"[Test Input]: \"{input_text}\".\n"
                pass_rate = image_annotations.get(child_topic, {}).get(input_text, {}).get("Pass Rate", 0)
                output_string += f"[Pass Rate]: {pass_rate}\n\n"
        break
                
    return output_string
#只有当前topic下测试记录的版本+MiniInput作为输入,待修改
def format_records_output_current_topic_miniinput(test_records, test_inputs, image_annotations,current_topic):
    output_string = ""
    # for topic, topic_data in test_records[current_topic].items():
    #     output_string += f"**Test Records: {topic}**\n"
    if not bool(image_annotations):
        return output_string
    topic_data = test_records[current_topic]
    child_topics=[]
    child_topics.append(current_topic)
    child_topics.extend( topic_data.get("Child_topics", []))
    
    for child_topic in child_topics:
        output_string += f"***Test Topic***: \"{child_topic}\".\n"
        inputs=[]
        mini_inputs=[]
        inputs_oral=test_inputs.get(child_topic, {}).get("input_lists", [])
        for input_oral in inputs_oral:
            mini_inputs.append(input_oral)
            mini_inputs.extend(test_inputs.get(child_topic, {}).get(input_oral, {}).get("MiniInputs", []))
            # mini_inputs.extend(test_inputs.get(child_topic, {}).get(input_text, {}).get("MiniInputs", []))
        
        inputs.extend(mini_inputs)
        
        for input_text in inputs:
            output_string += f"[Test Input]: \"{input_text}\".\n"
            pass_rate = image_annotations.get(child_topic, {}).get(input_text, {}).get("Pass Rate", 0)
            output_string += f"[Pass Rate]: {pass_rate}\n\n"
    return output_string
import copy
import pprint

def empty_graph(graph):
    _es = [k for k in graph["entities"]]
    if len(graph["context"]) == 0 and len(graph["entities"]) == 1 and len(graph["relations"]) == 0 and len(graph["entities"][_es[0]]["attributes"]) == 0:
        return True
    else:
        return False


# def split_scene_graph(input_graph):

#     if empty_graph(input_graph):
#         return None, None, None
    
#     graph = copy.deepcopy(input_graph)

#     fst_graph = {k : dict() for k in graph}
#     snd_graph = {k : dict() for k in graph}
#     trd_graph = {k : dict() for k in graph}

#     fst_graph["context"] = []
#     snd_graph["context"] = []
#     trd_graph["context"] = []

    
#     r = copy.deepcopy(graph["relations"])
#     r_keys = [k for k in r]

#     if len(r) > 1:
    
#         fst_graph["context"] = copy.deepcopy(graph["context"])
#         snd_graph["context"] = copy.deepcopy(graph["context"])
#         trd_graph["context"] = copy.deepcopy(graph["context"])

#         e_used = {e: False for e in graph["entities"]}
        
#         for i in range(len(r)):
#             ri = r[r_keys[i]]
#             if i < len(r) // 2:
#                 fst_graph["relations"][r_keys[i]] = copy.deepcopy(ri)
                
#                 for e in ri["entities"]:
#                     fst_graph["entities"][e] = copy.deepcopy(graph["entities"][e])
#                     e_used[e] = True
                
#             else:
#                 snd_graph["relations"][r_keys[i]] = copy.deepcopy(ri)
        
#                 for e in ri["entities"]:
#                     snd_graph["entities"][e] = copy.deepcopy(graph["entities"][e])
#                     e_used[e] = True
        
#         for e in e_used:
#             if not e_used[e]:
#                 trd_graph["entities"][e] = copy.deepcopy(graph["entities"][e])
        
#         if len(trd_graph["entities"]) == 0:
#             return fst_graph, snd_graph, None
#         else:
#             return fst_graph, snd_graph, trd_graph

#     elif len(graph["entities"]) > 1:
        
#         fst_graph["context"] = copy.deepcopy(graph["context"])
#         snd_graph["context"] = copy.deepcopy(graph["context"])
        
#         e = graph["entities"]
#         e_keys = [k for k in e]
        
#         for i in range(len(e)):
#             ei = e[e_keys[i]]
#             if i < len(e) // 2:
#                 fst_graph["entities"][e_keys[i]] = copy.deepcopy(ei)
#             else:
#                 snd_graph["entities"][e_keys[i]] = copy.deepcopy(ei)

#         return fst_graph, snd_graph, None
    
#     elif len(graph["entities"]) == 1:
        
#         e = graph["entities"]
#         e_keys = [k for k in e]
#         e_k = e_keys[0]
        
#         graph["entities"][e_k]["attributes"].extend(copy.deepcopy(graph["context"]))
        
#         a = copy.deepcopy(graph["entities"][e_k]["attributes"])
        
#         if len(a) <= 1:
#             graph["entities"][e_k]["attributes"] = []
#             return graph, None, None
#         else:
#             fst_graph["entities"][e_k] = copy.deepcopy(graph["entities"][e_k])
#             snd_graph["entities"][e_k] = copy.deepcopy(graph["entities"][e_k])
            
#             fst_graph["entities"][e_k]["attributes"] = fst_graph["entities"][e_k]["attributes"][: len(a)//2]
#             snd_graph["entities"][e_k]["attributes"] = snd_graph["entities"][e_k]["attributes"][len(a)//2 :]

#             return fst_graph, snd_graph, None

#     else:
#         return None, None, None

def split_scene_graph(input_graph):

    if empty_graph(input_graph):
        return None, None, None
    
    graph = copy.deepcopy(input_graph)

    fst_graph = {k : dict() for k in graph}
    snd_graph = {k : dict() for k in graph}
    trd_graph = {k : dict() for k in graph}

    fst_graph["context"] = []
    snd_graph["context"] = []
    trd_graph["context"] = []

    
    r = copy.deepcopy(graph["relations"])
    r_keys = [k for k in r]

    if len(r) > 1:
    
        fst_graph["context"] = copy.deepcopy(graph["context"])
        snd_graph["context"] = copy.deepcopy(graph["context"])
        trd_graph["context"] = copy.deepcopy(graph["context"])

        e_used = {e: False for e in graph["entities"]}
        
        for i in range(len(r)):
            ri = r[r_keys[i]]
            if i < len(r) // 2:
                fst_graph["relations"][r_keys[i]] = copy.deepcopy(ri)
                
                for e in ri["entities"]:
                    fst_graph["entities"][e] = copy.deepcopy(graph["entities"][e])
                    e_used[e] = True
                
            else:
                snd_graph["relations"][r_keys[i]] = copy.deepcopy(ri)
        
                for e in ri["entities"]:
                    snd_graph["entities"][e] = copy.deepcopy(graph["entities"][e])
                    e_used[e] = True
        
        for e in e_used:
            if not e_used[e]:
                trd_graph["entities"][e] = copy.deepcopy(graph["entities"][e])
        
        if len(trd_graph["entities"]) == 0:
            return fst_graph, snd_graph, None
        else:
            return fst_graph, snd_graph, trd_graph

    elif len(graph["entities"]) > 1:
        
        fst_graph["context"] = copy.deepcopy(graph["context"])
        snd_graph["context"] = copy.deepcopy(graph["context"])
        
        e = graph["entities"]
        e_keys = [k for k in e]
        
        for i in range(len(e)):
            ei = e[e_keys[i]]
            if i < len(e) // 2:
                fst_graph["entities"][e_keys[i]] = copy.deepcopy(ei)
            else:
                snd_graph["entities"][e_keys[i]] = copy.deepcopy(ei)

        return fst_graph, snd_graph, None
    
    elif len(graph["entities"]) == 1:
        
        e = graph["entities"]
        e_keys = [k for k in e]
        e_k = e_keys[0]
        
        graph["entities"][e_k]["attributes"].extend(copy.deepcopy(graph["context"]))
        
        a = copy.deepcopy(graph["entities"][e_k]["attributes"])
        if len(a) <= 1:
            graph["entities"][e_k]["attributes"] = []
            len_context = len(graph["context"])
            snd_graph = copy.deepcopy(graph)
            if len_context>1:
                
                graph["entities"][e_k]["attributes"].extend(graph["context"][:len_context/2])
                snd_graph["entities"][e_k]["attributes"].extend(graph["context"][len_context/2:])
                graph["context"] = []
                snd_graph["context"] = []
                return graph, snd_graph,None
            else:
                # graph["entities"][e_k]["attributes"].extend(graph["context"])
                graph["context"] = []
                return graph, None, None
        else:
            fst_graph["entities"][e_k] = copy.deepcopy(graph["entities"][e_k])
            snd_graph["entities"][e_k] = copy.deepcopy(graph["entities"][e_k])
            
            fst_graph["entities"][e_k]["attributes"] = fst_graph["entities"][e_k]["attributes"][: len(a)//2]
            snd_graph["entities"][e_k]["attributes"] = snd_graph["entities"][e_k]["attributes"][len(a)//2 :]

            return fst_graph, snd_graph, None

    else:
        return None, None, None
def split_recursive(input_graph):
    
    graph_tree = dict()
    graph_list = []
    
    graph_list.append(input_graph)
    
    _a, _b = 0, len(graph_list)
    
    while True:
        new_gs = []
        for gi in range(_a, len(graph_list)):
            child_gs = split_scene_graph( graph_list[gi] )

            for cg in child_gs:
                if cg is not None:
                    graph_tree[gi] = graph_tree.get(gi, []) + [_b + len(new_gs)]
                    new_gs.append(cg)

        if len(new_gs) == 0:
            break
        # else:
        #     if all([empty_graph(g) for g in new_gs]):
        #         break

        _a = len(graph_list)
        graph_list.extend(new_gs)
        _b = len(graph_list)

        # pprint.pprint(graph_list)
        # input()

    return graph_list, graph_tree

class MaxDepthException(Exception):
    def __init__(self, message):
        self.message = message

class ExperimentData:
    def __init__(self,save_path,init_topic,logger,api_key):
        self.test_inputs = {}
        self.image_annotations = {}
        self.iteration = 0
        self.max_test_depth = 0
        self.filename = save_path
        self.current_topic=init_topic
        self.messages =[]
        self.reflections_records = {}
        self.test_topics={}
        self.logger = logger
        self.tokens=0
        self.current_reflection=""
        self.minimize_inputs={}
        self.minimize_inputs={}
        self.current_inputs_image_annotations={}
        self.init_topic=init_topic
        openai.key = api_key
        self.c={"context": [],
            "entities": {},
            "relations": {}
            }
        self.r={"context": [],
            "entities": {},
            "relations": {}
            }
        self.num_topics=0
        self.num_inputs=0


    def generate_test_topics(self,num_topics):
        #initialization
        if  not bool(self.test_topics):#如果之前没记录过这个Topic
            if self.current_topic in self.image_annotations:
                test_records_prompt=format_records_output_current_topic_miniinput(self.test_topics, self.test_inputs, self.image_annotations,self.current_topic)#Need to change(是否需要改成只加载当前topic?)
                self.reflections_records[self.current_topic] = self.reflection()
                reflection_prompt="Based on the provided test records, we can observe the following patterns:\n"+self.reflections_records[self.current_topic]
                print("412")
            else:
                test_records_prompt="Test Record:\nN/A"
                reflection_prompt=""
                self.reflections_records[self.current_topic]=""
        # elif self.current_topic==self.init_topic: #如果已经有过记录了，但是
        #     test_records_prompt=format_records_output_current_topic_miniinput(self.test_topics, self.test_inputs, self.image_annotations,self.current_topic)#Need to change(是否需要改成只加载当前topic?)
        #     reflection_prompt = "Based on the provided test records, we can observe the following patterns:\n"+self.reflections_records[self.current_topic]
        #     self.reflections_records[self.current_topic]=
        elif self.test_topics[self.current_topic]['Depth'] >= self.max_test_depth-1:# 如果topic大于最大深度，则抛一个异常不生成
            raise MaxDepthException('max depth touched')
        elif self.current_topic in self.reflections_records:#如果之前记录过reflection
            test_records_prompt=format_records_output_current_topic_miniinput(self.test_topics, self.test_inputs, self.image_annotations,self.current_topic)#Need to change(是否需要改成只加载当前topic?)
            self.reflections_records[self.current_topic] = self.reflection()
            reflection_prompt="Based on the provided test records, we can observe the following patterns:\n"+self.reflections_records[self.current_topic]
            print("409")
        else:#如果之前记录过这个Topic 但没有记录过reflection
            test_records_prompt=format_records_output_current_topic_miniinput(self.test_topics, self.test_inputs, self.image_annotations,self.current_topic)#Need to change(是否需要改成只加载当前topic?)
            self.reflections_records[self.current_topic] = self.reflection()
            reflection_prompt="Based on the provided test records, we can observe the following patterns:\n"+self.reflections_records[self.current_topic]
            print("414")
        if self.current_topic not in self.test_topics:
            self.test_topics[self.current_topic] = {"Father_topic": "", "Child_topics": [],"Father_Reflection":"","Depth":0}
        #如果当前topic还没有生成过测试样本
        # if self.current_topic not in self.test_topics:
        #     self.test_topics[self.current_topic] = {}

        text_prompts = generate_test_inputs(num_topics,is_input=False)
        system_prompt1=f"""As a professional testing expert, your task is to test a text-to-image generation model.
        The current focus is on the topic of '{self.current_topic}'   Please provide structured next test topic that explore finer details of the topic itself, as well as combinations or different relationships of the topic with other objects.
        Your objective is to generate new test topics based on based on the test record to uncover as many errors in the model as possible. In the test records, topic indicates the test topic, text prompt indicates the actual input to the tested model, and Score indicates whether the test is passed (0 fail,1 pass).
        {test_records_prompt}
        {reflection_prompt}
        Ensure that each output is relevant and distinct.
        Please keep the format and fill all the <OUTPUT>.
        Current test topic: {self.current_topic}
        """
        messages = [{'role': 'system', 'content': system_prompt1+text_prompts}]
        messages,response,self.tokens,=post_message(messages, self.tokens, self.logger)
        self.messages=self.messages + messages
        # start_index = response.find("[Next Test Topic]")
        # # 使用切片保留从该索引位置到最后的部分
        # desired_string = response[start_index:]
        # # reranking
        # response = self.topic_curation(self.current_topic,desired_string)
        topics = [line.split("]: ")[1] for line in response.strip().split("\n") if line.startswith("[Next Test Topic]")]
        # new_topics = random.sample(topics, 10)
        for topic in topics:
            if topic not in self.test_topics:
                topic = topic.replace("* ","").replace("\"","")
                self.test_topics[self.current_topic]["Child_topics"].append(topic)
                self.test_topics[topic]={"Father_topic":self.current_topic,
                                         "Child_topics":[],
                                         "Father_Reflection":self.reflections_records[self.current_topic],
                                         "Depth":self.test_topics[self.current_topic]["Depth"]+1}
        self.num_topics=num_topics
        return self.test_topics
    def return_current_topics(self, offline=False):
        test_topics_online=[]
        test_topics_offline=[]
        for topic_name in self.test_topics.keys():
            if topic_name in self.image_annotations and  len(self.image_annotations[topic_name].keys())>=self.num_inputs:
                test_topics_online.append("* "+topic_name)
            else: 
                test_topics_online.append(topic_name)
            test_topics_offline.append(topic_name)
        if offline:
            return test_topics_online, test_topics_offline
        else:
            return test_topics_online
    
    def return_current_inputs(self, topic_name, offline=True):
        test_inputs_online=[]
        test_inputs_offline=[]

        if topic_name not in self.test_inputs: 
            if offline:
                return None, None
            else:
                return None

        for input_name in self.test_inputs[topic_name]["input_lists"]:
            if topic_name in self.image_annotations:
                if input_name in self.image_annotations[topic_name]:#
                    test_inputs_online.append("* "+input_name)
                else:
                    test_inputs_online.append(input_name)
            else: 
                test_inputs_online.append(input_name)
            test_inputs_offline.append(input_name)
        if offline:
            return test_inputs_online, test_inputs_offline
        else:
            return test_inputs_online

    def generate_test_inputs(self, topic,num_inputs):
        self.num_inputs=num_inputs
        # import pdb;pdb.set_trace()
        if not bool(self.test_inputs) or not bool(self.reflections_records[self.test_topics[self.current_topic]["Father_topic"]]):
            test_records_prompt="Test Record:\n N/A"
            reflection_prompt=""
        else :
            test_records_prompt=format_records_output_current_topic_miniinput(self.test_topics, self.test_inputs, self.image_annotations,self.current_topic) #need to change
            reflection_prompt="Based on the provided test records, we can observe the following patterns:\n"+self.reflections_records[self.test_topics[self.current_topic]["Father_topic"]]

        text_prompts = generate_test_inputs(num_inputs,is_input=True)
        system_prompt2=f"""
As a professional testing expert, your task is to test a text-to-image generation model focusing on the theme of '{self.current_topic}'.
Provide specific test inputs that align with the theme, exploring finer details/contexts/relations/actions of the theme itself.
Your goal is to generate new test inputs based on the Test Record to uncover as many errors in the model as possible.
In the Test Record, topic indicates the test topic, text input indicates the actual input to the tested model, and Score indicates whether the test is passed (0 fail,1 pass), N/A means Test record is empty.
{test_records_prompt}
{reflection_prompt}
Ensure that each output is relevant to the current test topic, **suitable for image to display**, and distinct to each other. Please keep the format and fill all the <OUTPUT>. Remember to ensure the maximum input length is **40** words.
**Increase the difficulty or lengths of your generated new test inputs progressively based on Test Record.**
Current test topic: {self.current_topic}
        """
        messages=[{'role': 'system', 'content': system_prompt2+text_prompts}]
        messages,response,self.tokens=post_message(messages, self.tokens, self.logger)
        self.messages=self.messages+messages
        
        # start_index = response.find("[Test Input]")
        # # 使用切片保留从该索引位置到最后的部分
        # desired_string = response[start_index:]
        # # reranking
        # response = self.test_inputs_curation(self.current_topic,desired_string)
        inputs = [line.split("]: ")[1] for line in response.strip().split("\n") if line.startswith("[Test Input]")]
        
        if topic not in self.test_inputs:
            self.test_inputs[topic] ={"input_lists":[]}
        
        for input in inputs:
            if input not in self.test_inputs[topic]:
                input = input.replace("* ","").replace("\"","")
                self.test_inputs[topic]["input_lists"].append(input)
                if bool(reflection_prompt):
                    self.test_inputs[topic][input]={"MiniInputs":[],"MiniInputs_Scene_Graphs":[],
                                         "Father_Reflection":self.reflections_records[self.test_topics[self.current_topic]["Father_topic"]]}
                else:
                    self.test_inputs[topic][input]={"MiniInputs":[],"MiniInputs_Scene_Graphs":[],
                                         "Father_Reflection":""}
        return self.test_inputs[topic]["input_lists"]

    def add_image_annotations(self, topic, input_description,image_count, annotations,image_path,times):
        if topic not in self.image_annotations:
            self.image_annotations[topic] = {}
        if input_description not in self.image_annotations[topic]:
            self.image_annotations[topic][input_description] = {}
        # if not bool(self.image_annotations[topic][input_description][str(image_count)]):
        self.image_annotations[topic][input_description][str(image_count)]={"Annotations":annotations,"image_path":image_path,"time":times}

    def add_image_pass_rate(self, topic, input_description,annotations):
        if topic not in self.image_annotations:
            self.image_annotations[topic] = {}
        if input_description not in self.image_annotations[topic]:
            self.image_annotations[topic][input_description] = {}
        # if not bool(self.image_annotations[topic][input_description]["Pass Rate"]):
        self.image_annotations[topic][input_description]["Pass Rate"] = annotations
    def generate_random_binary_array(self,Num):
    # 使用列表推导式生成一个长度为Num的随机0/1数组
        return [random.randint(0, 1) for _ in range(Num)]
    def minimize_input_generation(self, prompt):
        test_labels=[]
        labels=[]
        c = self.generate_scene(prompt)
        print(c)
        r = {"context": [],
            "entities": {},
            "relations": {}
            }
        while(not empty_graph(c)):
            current_tests_inputs=[]
            # import pdb;pdb.set_trace()
            split_sets=split_scene_graph(c)
            # import pdb;pdb.set_trace()
            split_sets = list(filter(None, split_sets)) 
            nums_labels=len(split_sets)
            print ("r", r)
            current_tests_inputs_scenes=[]
            
            for i in range(nums_labels):
                x = self.merge_scene_graphs(r,split_sets[i])
                x2 = self.generate_scene_to_text(total_text=prompt,scene_graph=x)
                current_tests_inputs.append(x2)
                current_tests_inputs_scenes.append(x)
                pprint.pprint (f"current_tests_inputs:{x}")
                pprint.pprint (f"current_tests_inputs_scene_graph:{x2}")
                #如果这个input之前标记过
                # print(self.image_annotations, type(self.image_annotations))
                # print(self.current_topic, type(self.current_topic))
                # if x2 in self.image_annotations[self.current_topic]:
                #     if  self.image_annotations[self.current_topic][x2]["Pass Rate"] >0.75:
                #         labels[i]=1
                #     else:
                #         labels[i]=0
            self.test_inputs[self.current_topic][prompt]["MiniInputs"].extend(current_tests_inputs)
            self.test_inputs[self.current_topic][prompt]["MiniInputs_Scene_Graphs"].extend(current_tests_inputs_scenes)
            
            #剩下的没有被标记过的Input需要人来标注，但这里不知道写啥
            labels = yield current_tests_inputs
            
            #测试
            # labels=self.generate_random_binary_array(nums_labels)
            print("labels",labels)
            test_labels.extend(labels)
            next_iter=False
            for i in range(nums_labels):
                if labels[i] ==0 and not next_iter:
                    
                    c=split_sets[i]#i不对的话
                    next_iter=True
            if not next_iter:
                r = self.merge_scene_graphs(r,split_sets[0])
                if nums_labels==2:
                    c = split_sets[1]
                elif nums_labels ==3:
                    c =  self.merge_scene_graphs(split_sets[1],split_sets[2])
                else:#c就拆分出一个来，并且这一个也是对的
                    c = None 
                    break
        return None
    def merge_scene_graphs(self,scene1, scene2):
        """
        Merge two scene graphs into a single graph.

        :param scene1: First scene graph.
        :param scene2: Second scene graph.
        :return: Merged scene graph.
        """
        # Initialize the merged scene with the context from the first scene
        merged_scene = {'context': list(set(scene1['context']+scene2['context'])), 'entities': {}, 'relations': {}}

        # Merge entities
        for scene in [scene1, scene2]:
            for entity, details in scene['entities'].items():
                if entity in merged_scene['entities']:
                    # Merge attributes of the same entity
                    merged_scene['entities'][entity]['attributes'] = list(set(merged_scene['entities'][entity]['attributes']) | set(details['attributes']))
                else:
                    merged_scene['entities'][entity] = details

        # Merge relations
        for scene in [scene1, scene2]:
            for relation, details in scene['relations'].items():
                if relation in merged_scene['relations']:
                    # Merge entities of the same relation
                    merged_scene['relations'][relation]['entities'] = list(set(merged_scene['relations'][relation]['entities']) | set(details['entities']))
                    # Merge attributes of the same relation
                    merged_scene['relations'][relation]['attributes'] = list(set(merged_scene['relations'][relation]['attributes']) | set(details['attributes']))
                else:
                    merged_scene['relations'][relation] = details
        return merged_scene
    def generate_scene(self,test_input):
        system_prompt= """ Task: given input prompts, and transformed into scene graph. Do not generate nodes or edges that are not explicitly described in the prompts and do not lose key information in prompts! The nodes fall into five categories: Objects, Relations, Object_Attributions, Relation_Attributions, and Context.
output format:{
"context": ["context1","context2",...],
    "entities": {
        "entity1": {
            "attributes": []
        },
        "entity2": {
            "attributes": []
        },
        ...
    },
    "relations": [
        {
            "relation1": {
            "entities": ["entity1","entity2"],
            "attributes": []
        },
        "relation2": {
            "entities": [...],
            "attributes": []
        },
        ...
        }
    ]
}

input: "Two sleek and elegant Greyhound with a slender body and long legs and a friendly and intelligent Golden Retriever with a beautiful golden coat and a wagging tail are playing in the park."

output: {
"context": ["in the park"],
    "entities": {
        "Greyhound": {
            "attributes": ["Two ", "sleek", "elegant",]
        },
"body": {
            "attributes": ["slender"]
        },
"legs": {
            "attributes": ["long"]
        },
        "Golden Retriever": {
            "attributes": ["friendly", "intelligent"]
        },
        "coat": {
            "attributes": ["beautiful", " golden "]
        },
        "tail": {
            "attributes": ["wagging"]
        },
    },
    "relations": 
        {
"with": {
            "entities": ["Greyhound", "body"],
            "attributes": []
        },
"with": {
            "entities": ["Greyhound", "legs"],
            "attributes": []
        },
"with": {
            "entities": ["Golden Retriever", "coat"],
            "attributes": []
        },
"with": {
            "entities": ["Golden Retriever","tail"],
            "attributes": []
        },
            "playing together": {
            "entities": ["Greyhound", "Golden Retriever"],
            "attributes": []
        }
        }
}

input : "A dog sitting patiently beside a human, waiting for a treat and wagging its tail in anticipation."
output:
{
    "context": [],
    "entities": {
        "dog": {
            "attributes": ["sitting patiently"]
        },
        "human": {
            "attributes": []
        },
        "treat": {
            "attributes": []
        },
"tail": {
"attributes": []
}
    },
    "relations": 
        {
        "beside": {
                "entities": ["dog", "human"],
                "attributes": []
            },
            "waiting for": {
                "entities": ["dog", "treat"],
                "attributes": ["in anticipation"]
            },
"wagging": {
                "entities": ["dog", "tail"],
                "attributes": ["in anticipation"]
            },
        }
}


input: "A fluffy white cat with bright blue eyes sitting on a windowsill, watching birds outside."
output: 
{
    "context": {
    },
    "entities": {
        "cat": {
            "attributes": ["fluffy", "white"]
        },
        "eyes": {
            "attributes": ["bright blue"]
        },
        "windowsill": {
            "attributes": []
        },
        "birds": {
            "attributes": []
        }
    },
    "relations": {
        "with": {
            "entities": ["cat", "eyes"],
            "attributes": []
        },
        "sitting on": {
            "entities": ["cat", "windowsill"],
            "attributes": []
        },
        "watching": {
            "entities": ["cat", "birds"],
            "attributes": ["outside"]
        }
    }
}

input: "A dog eagerly jumping into its owner's arms for a hug."
output:
{
    "context": [],
    "entities": {
        "dog": {
            "attributes": ["eagerly"]
        },
        "dog's owner's arms": {
            "attributes": []
        },
        "hug": {
            "attributes": []
        }
    },
    "relations": {
        "jumping into": {
            "entities": ["dog", "dog's owner's arms"],
            "attributes": []
        },
        "for": {
            "entities": ["dog", "hug"],
            "attributes": []
        }
    }
}

input: "The concept of time as a flowing river, with past, present, and future merging together."
output:
{'context': [],
 'entities': {'future': {'attributes': []},
              'past': {'attributes': []},
              'present': {'attributes': []},
              'time': {'attributes': ['concept of']},
              'flowing river':{'attributes': []}},
 'relations': {'as': {'attributes': [], 'entities': ['time', 'flowing river']},
               'merging together': {'attributes': [],
                                    'entities': ['past', 'present', 'future']}}}

input: "A fluffy golden retriever playing fetch in a lush green park."
output:
{
    "context": ["in a lush green park"],
    "entities": {
        "golden retriever": {
            "attributes": ["fluffy"]
        },
        "fetch": {
            "attributes": []
        }
    },
    "relations": {
        "playing": {
            "entities": ["golden retriever", "fetch"],
            "attributes": []
        }
    }
}
input: "A dog chasing its tail in circles, looking playful and energetic."
output: 
    "context": [],
    "entities": {
        "dog": {
            "attributes": []
        },
        "tail": {
            "attributes": []
        }
    },
    "relations": {
        "chasing": {
            "entities": ["dog", "tail"],
            "attributes": ["in circles"]
        },
        "looking": {
            "entities": ["dog"],
            "attributes": ["playful", "energetic"]
        }
    }
}
"""
        input_prompt=f"""
        input: {test_input}
        output:
        """
        messages = [{'role': 'system', 'content': system_prompt+input_prompt}]
        messages,response,self.tokens,=post_message(messages, self.tokens, self.logger)
        self.messages=self.messages + messages
        scene_graph=json.loads(response)
        return scene_graph
    def generate_scene_to_text(self,total_text,scene_graph):
#         system_prompt= """
#         Task: Given a text T, and part of scene graph G of T. describe G accurately in text. Do not output any Entity/Relation/Context that is not in G, especially DO NOT OUTPUT any information that is not in the G but is in T, and do not omit any nodes in G. Be precise and concise.

# Input T: "Two sleek and elegant Greyhound with a slender body and long legs and a friendly and intelligent Golden Retriever with a beautiful golden coat and a wagging tail are playing in the park."

# Input G: {
# "context": {
# }
#     "entities": {
#         "Greyhound": {
#             "attributes": ["sleek", "elegant",]
#         },
# "body": {
#             "attributes": ["slender"]
#         },
# "legs": {
#             "attributes": ["long"]
#         }
#     },
#     "relations": 
#         {
# "with": {
#             "entities": ["Greyhound", "body"],
#             "attributes": []
#          },
# "with": {
#             "entities": ["Greyhound", "legs"],
#             "attributes": []
#          }
#         }
# }
# Output: "A sleek and elegant Greyhound with a slender body and long legs."

# Input T: "A dog sitting patiently beside a human, waiting for a treat and wagging its tail in anticipation."
# Input G:
# {
#     "context": {},
#     "entities": {
#         "dog": {
#             "attributes": ["sitting patiently"]
#         },
#         "human": {
#             "attributes": []
#         }
#     },
#     "relations": 
#         {
#         "beside": {
#                 "entities": ["dog", "human"],
#                 "attributes": []
#             }
#             }
#         }
# }
# Output: 

# Input T: "A dog following closely behind its owner, matching their every step."
# Input G:
# {
#     "context": [],
#     "entities": {
#         "dog": {
#             "attributes": []
#         },
#         "step": {
#             "attributes": []
#         }
#     },
#     "relations": {
#         "matching": {
#             "entities": ["dog", "step"],
#             "attributes": ["every"]
#         }
#     }
# }
# Output: "A dog matching its owner's every step.

# Input T: "A fluffy white cat chasing a green ball across the room."
# Input G:
# {'context': ['across the room'],
#  'entities': {'cat': {'attributes': ['fluffy', 'white']}},
#  'relations': {}}
# Output: "A fluffy white cat across the room.

# Input T: "A fluffy white cat chasing a green ball across the room."
# Input G:
# {'context': ['across the room'],
#  'entities': {'ball': {'attributes': ['green']}},
#  'relations': {}}
# Output: "A green ball across the room."""
        system_prompt= """
        Task: Given a scene graph G of T. describe G accurately in text. Do not output any Entity/Relation/Context that is not in G, especially DO NOT OUTPUT any information that is not in the G, and do not omit any nodes in G. Be precise and concise.

Input G: {
"context": {
}
    "entities": {
        "Greyhound": {
            "attributes": ["sleek", "elegant",]
        },
"body": {
            "attributes": ["slender"]
        },
"legs": {
            "attributes": ["long"]
        }
    },
    "relations": 
        {
"with": {
            "entities": ["Greyhound", "body"],
            "attributes": []
         },
"with": {
            "entities": ["Greyhound", "legs"],
            "attributes": []
         }
        }
}
Output: "A sleek and elegant Greyhound with a slender body and long legs."

Input G:
{
    "context": {},
    "entities": {
        "dog": {
            "attributes": ["sitting patiently"]
        },
        "human": {
            "attributes": []
        }
    },
    "relations": 
        {
        "beside": {
                "entities": ["dog", "human"],
                "attributes": []
            }
            }
        }
}
Output: "A dog sitting patiently beside a human."

Input G:
{
    "context": [],
    "entities": {
        "dog": {
            "attributes": []
        },
        "step": {
            "attributes": []
        }
    },
    "relations": {
        "matching": {
            "entities": ["dog", "step"],
            "attributes": ["every"]
        }
    }
}
Output: "A dog matching its owner's every step.

Input G:
{'context': ['across the room'],
 'entities': {'cat': {'attributes': ['fluffy', 'white']}},
 'relations': {}}
Output: "A fluffy white cat across the room.

Input G:
{'context': ['across the room'],
 'entities': {'ball': {'attributes': ['green']}},
 'relations': {}}
Output: "A green ball across the room.

Input G:
{
    "context": ["in a lush green park"],
    "entities": {
        "golden retriever": {
            "attributes": ["fluffy"]
        }
    }, 
    'relations': {}
}
Output: "A fluffy golden retriever in a lush green park."

Input G:
{
    "context": ["in a lush green park"],
    "entities": {
        "fetch": {
            "attributes": []
        }
    },
    "relations": {
    }
}
Output: "A fetch in a lush green park."

Input G:{
    "context": [],
    "entities": {
        "dog": {
            "attributes": []
        },
        "tail": {
            "attributes": []
        }
    },
    "relations": {
        "chasing": {
            "entities": ["dog", "tail"],
            "attributes": ["in circles"]
        }
    }
Output: "A dog chasing its tail in circles."

Input G:{
    "context": [],
    "entities": {
        "dog": {
            "attributes": []
        }
    },
    "relations": {
        "looking": {
            "entities": ["dog"],
            "attributes": ["playful", "energetic"]
        }
    }
}
Output: "A dog looking playful and energetic."
"""
        input_prompt=f"""

Input G: {scene_graph}
Output:
        """
        messages = [{'role': 'system', 'content': system_prompt+input_prompt}]
        messages,response,self.tokens,=post_message(messages, self.tokens, self.logger)
        self.messages=self.messages + messages
        return response
    def reflection(self):
        test_records=format_records_output_current_topic_miniinput(self.test_topics, self.test_inputs, self.image_annotations,self.current_topic)
        
        system_prompt5 = f"""
You are an expert in text-generated image model testing and are good at finding error patterns in models. Below I will tell you the specific task objectives and the existing test records. In the test records, topic indicates the test topic, text prompt indicates the actual input to the tested model, and Score indicates whether the test is passed (0 fail,1 pass).

Analyze the performance of a text-to-image model based on provided test records: why some test cases in the test record generate failures (Score 0) and successes (Score 1), and summarize failure patterns where the model may underperform.  List by points.\n{test_records}
        """
        messages=[{'role': 'system', 'content': system_prompt5}]
        messages,response,self.tokens=post_message(messages, self.tokens, self.logger)
        self.messages=self.messages+messages
        self.reflections_records[self.current_topic] = response
        print("self.current_topic",self.current_topic)
        return response
    
    def topic_curation(self,current_topic,next_topics):
        system_prompt3=f"""
        As a professional testing expert, your task is to test a text-to-image generation model focusing on the theme of '{current_topic}'. Provide specific test inputs that align with the theme, exploring finer details of the theme itself.
        Your goal is to reorder the newly generated test topics based on existing test records. Re-rank the test topics according to their importance!
        The ranking principles are: 1) test topic is more able to find errors in the tested model, 2) Text content can be represented entirely in images, and can be image captions. 
        Current test topic: {current_topic}
        {next_topics}
        Direct Output the final Orders and keep the format: [Next Test Topic]: <OUTPUT>!
        """
        messages=[{'role': 'system', 'content': system_prompt3}]
        messages,response,self.tokens =post_message(messages, self.tokens, self.logger)
        self.messages=self.messages + messages
        print(236,response)
        return response
    
    def test_inputs_curation(self,current_topic,next_inputs):
        system_prompt4=f"""
        As a professional testing expert, your task is to test a text-to-image generation model focusing on the theme of '{current_topic}'. Provide specific test inputs that align with the theme, exploring finer details of the theme itself.
        Your goal is to reorder the newly generated test inputs based on existing test records. Re-rank the test inputs according to their importance!
        The ranking principles are: 1) test input is more able to find errors in the tested model, 2) Text content can be represented entirely in images, and can be image captions. 
        Current test topic: {current_topic}
        {next_inputs}
        Direct Output the final Orders and keep the format: [Test Input]: <OUTPUT>!
        """
        messages=[{'role': 'system', 'content': system_prompt4}]
        messages,response,self.tokens =post_message(messages, self.tokens, self.logger)
        self.messages=self.messages + messages
        print(251,response)
        return response
    
    def Initialize_clip(self,clip_model_name,clip_pretrain,device):
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrain, device=device, jit=True)
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        return tokenizer,clip_preprocess,clip_model

    def get_text_img_similarity(self,init_input, generated_image,device,tokenizer,clip_preprocess,clip_model):
        text = tokenizer([init_input]).to(device)

        gen_batch = [clip_preprocess(i).unsqueeze(0) for i in [generated_image]]
        gen_batch = torch.concatenate(gen_batch).to(device)

        gen_feat = clip_model.encode_image(gen_batch)
        text_feat = clip_model.encode_text(text)
        
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)

        return (gen_feat @ text_feat.t()).mean().item()

    def auto_labeling(self,text, ori_image,device,tokenizer,clip_preprocess,clip_model,threshold):
        score = self.get_text_img_similarity(text, ori_image,device,tokenizer,clip_preprocess,clip_model)
        annotations=[]
        for i in score:
            if i>threshold:
                annotations.append(1)
            else:
                annotations.append(0)
        return annotations

    def next_iteration(self):
        self.save_to_file()
        self.iteration += 1
        self.generate_test_topics()
    def save_to_file(self):
        data = {
            "test_records": self.test_topics,
            "test_inputs": self.test_inputs,
            "image_annotations": self.image_annotations,
            "iteration": self.iteration,
            "messages":self.messages,
            "reflections":self.reflections_records,
            "current_topic":self.current_topic
        }
        with open(self.filename, 'w') as file:
            json.dump(data, file, indent=4)
            
    def load_from_file(self ):
        with open(self.filename, 'r') as file:
            data = json.load(file)

        # 根据文件中的数据更新类的属性
        self.test_inputs = data.get("test_inputs", {})
        self.image_annotations = data.get("image_annotations", {})
        self.iteration = data.get("iteration", 0)
        self.current_topic = data.get("current_topic", "")
        print('current topic:{}'.format(self.current_topic))
        self.messages = data.get("messages", [])
        self.reflections_records = data.get("reflections", {})
        self.test_topics = data.get("test_records", {})
#