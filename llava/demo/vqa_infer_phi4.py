import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import tokenizers
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from tqdm import tqdm
import shortuuid
import pandas as pd

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
import io
import base64
import pickle
import numpy as np
from PIL import Image
import math
import argparse
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square

## data load
def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def load_pkl_image(image_file):
    images = []
    for i in range(len(image_file)):
        rawbytes = base64.b64decode(image_file[i]) #!!tmp fix for one image case
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB") #load question from .pkl
        images.append(image)

    return images

def draw_bounding_box(image, bounding_box,width=2):
    from PIL import Image, ImageDraw, ImageFont
    """
    Draws a red bounding box on an image and saves it.
    
    Parameters:
    - image_path (str): Path to the input image.
    - box_coords (tuple): Bounding box coordinates in (x_min, y_min, x_max, y_max) 
    or (x_min,y_min,z_min,x_max,y_max,z_max) format.
    
    Returns:
    - image
    """
    draw = ImageDraw.Draw(image)
    if len(bounding_box) == 4:
        box_coords = bounding_box
    else:
        box_coords = (bounding_box[0], bounding_box[1], bounding_box[3], bounding_box[4])
    # Draw red bounding box
    draw.rectangle(box_coords, outline="red", width=width)
    
    return image

def _load_nii(nii_path, num_video_frames, center_image=None, bounding_box=None, pixel_spacing=None, key_in_center=True):
    import nibabel as nib
    try:
        if nii_path.endswith('.npy'):
            image_np = np.load(nii_path) #origin: H,W,D
        else:
            image_np = nib.load(nii_path).get_fdata() #origin: H,W,D
    except Exception as e:
        print(f"[DEBUG] Error processing {nii_path}: {e}")
        image_np = np.zeros((512,512,32),dtype=np.float32)
    # toTensor = transforms.ToTensor()
    draw_bbox = False
    if bounding_box is not None:
        center_image = int(round((bounding_box[2] + bounding_box[2 + 3] - 1) / 2.0))
        draw_bbox = True
    image = np.transpose(image_np,(2,0,1)) #to D,H,W
    depth = image.shape[0]
    if center_image==None or center_image<=0:
        if num_video_frames < depth:
            step = int(math.ceil(depth / num_video_frames))
            image = image[::step]
            step_ids = [i for i in range(0, depth, step)]
        else:
            image = image
            step_ids = [i for i in range(depth)]
        slice_ids = [i for i in step_ids] #reverse back to original order
    else:
        select_img = center_image #TODO: need check nii orientation
        if select_img < 0:
            image = image[:min(num_video_frames,depth)]
            step_ids = [i for i in range(num_video_frames)]
        else:
            if key_in_center:
                half_num_frames = num_video_frames/2 #center always the center image
            else:
                half_num_frames = np.random.uniform(1, num_video_frames-1) #center is random
            start, end = max(0, int(math.ceil(select_img-half_num_frames))), min(depth, int(math.ceil(select_img+(num_video_frames-half_num_frames))))
            image = image[start:end]
            step_ids = [i for i in range(start,end)]
        slice_ids = [i for i in step_ids] #TODO: need check nii orientation
    pil_imgs = [Image.fromarray(img).convert('RGB') for img in image]
    if draw_bbox: #TODO: consider pixel_spacing and z axis width
        for i in range(len(pil_imgs)):
            pil_imgs[i] = draw_bounding_box(pil_imgs[i], bounding_box)
            #tmp visualize
            #pil_imgs[i].save('pil_debug.png')
    frames_loaded = len(pil_imgs)
    #print(frames_loaded,step_ids)
    assert frames_loaded == len(slice_ids)

    return pil_imgs, slice_ids, frames_loaded

def load_meta_data(meta_path):
    sample_meta_data, question = [], None
    dict_meta_data = {}
    if meta_path and meta_path!='None':
        file_liat = os.listdir(meta_path)
        for file in file_liat:
            if file.endswith(".txt"): #text type meta data
                with open(os.path.join(meta_path,file),'r') as f:
                    question = f.read().strip('\n').strip()
            elif file.endswith(".jsonl"): #list type meta data
                sample_meta_data += [json.loads(q) for q in open(os.path.join(meta_path,file), "r")]
            elif file.endswith(".json"): #dict type meta data {col_name:{col_value:meta_text}}
                dict_meta_data = json.load(open(os.path.join(meta_path,file), "r"))
        if question!=None:
            print('Override old question with: ')
            print(question)
        if len(sample_meta_data)>0:
            print('Add each sample meta data')
    
    return sample_meta_data, dict_meta_data, question

def get_image_num(image_file):
    if isinstance(image_file,list):
        return len(image_file)
    else:
        return 1

def get_batch(questions, batch_size):
    out_list = [] #(N/B,B)
    questions_len = len(questions)
    start = 0
    while start<questions_len:
        end = start + batch_size
        out_list.append(questions[start:min(end,questions_len)])
        start = end
    return out_list

##data save
def save_jsonl(out_path,data):
    with open(out_path, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')

## process report
#json to formatted text
def note_dict_to_text(note_dict,list_format=False):
    template = 'There is a {size_w_unit} {characteristic} {density} nodule in the {location} [**Image**].'
    if isinstance(note_dict,str):
        note_dict = json.loads(note_dict.replace('None','"null"'))
    nodule_list = note_dict['nodule_info']
    out_text = []
    for nodule in nodule_list:
        size = nodule['size']
        if (isinstance(size,float) or isinstance(size,int)) and size > 0:
            size_w_unit = '%.1f %s'%(size,nodule['unit'])
        elif isinstance(size,str) and size != 'null':
            size_w_unit = '%s %s'%(size,nodule['unit'])
        else:
            size_w_unit = ''
        characteristic = nodule.get('characteristic','null')
        if characteristic == 'null' or not characteristic:
            characteristic = ''
        density = nodule.get('density','null')
        if density == 'null' or not density:
            density = ''
        location = nodule.get('location','null')
        if location == 'null' or not location:
            location = ''
        #series_images = '(Series %d Image %d)'%(nodule['series'],nodule['image'])
        e_nodule = template.format(size_w_unit=size_w_unit,characteristic=characteristic,density=density,location=location).replace('  ',' ').replace('  ',' ')
        out_text.append(e_nodule)
    if list_format:
        return out_text
    else:
        return ' '.join(out_text)

#process vqa to nodules report json format
def process_vqa_report(ans_data,ans_key='answer'):
    out_data = []
    for name, gp_data in ans_data.groupby('NOTE_ID'):
        note_id = name
        nodule_info = []
        for slice_name,nodule_data in gp_data.groupby('Slice_id'): ### !!! error when slice_id is not unique !!!
            if select_by_qtype(nodule_data,'exist',ans_key).lower() == 'no':
                continue
            nodule_dict = {}
            nodule_dict['density'] = select_by_qtype(nodule_data,['attenuation','density'],ans_key)
            nodule_dict['series'] = int(slice_name.split('_')[1])
            nodule_dict['image'] = int(slice_name.split('_')[2])
            nodule_dict['size'] = select_by_qtype(nodule_data,'size',ans_key)
            if nodule_dict['size'] < 0 or nodule_dict['size'] == 'null' or nodule_dict['series'] < 0 or nodule_dict['image'] < 0:
                continue
            nodule_dict['unit'] = 'mm'
            nodule_dict['location'] = select_by_qtype(nodule_data,'location',ans_key)
            nodule_dict['margin'] = select_by_qtype(nodule_data,['margin','shape'],ans_key)
            nodule_dict['characteristic'] = select_by_qtype(nodule_data,'characteristic',ans_key)
            nodule_info.append(nodule_dict)
        #check valid note
        valid_note = 'Yes'
        for nodule in nodule_info:
            if nodule['size'] < 0 or nodule['location'] == 'null':
                valid_note = 'No'
        nodule_json = {'valid_note':valid_note,'nodule_info':nodule_info}
        out_data.append({'NOTE_ID':note_id,'Format_report':nodule_json,'Report_text':note_dict_to_text(nodule_json)})
    return out_data

def select_by_qtype(data,qtypes,ans_key='answer'):  ### !!! error when slice_id is not unique !!!
    if isinstance(qtypes,str):
        qtypes = [qtypes]
    value = data[data['question_type'].isin(qtypes)][ans_key].values
    if len(value) == 0:
        value = 'null'
    else:
        value = value[0]
    if str(value) == 'unable to determine':
        value = 'null'
    if 'size' in qtypes:
        try:
            value = float(value)
        except:
            value = -1
    return value


def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--dtype', type=str, default='FP32')
    parser.add_argument('--attn_implementation', type=str, default=None)
    parser.add_argument('--hlora_r', type=int, default=16)
    parser.add_argument('--hlora_alpha', type=int, default=32)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--vq_idx_nums', type=int, default=1024)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')
    parser.add_argument('--vit_path', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--hlora_path', type=str, default=None)
    parser.add_argument('--fusion_layer_path', type=str, default=None)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--meta-path", type=str, default=None,help='metadata path .jsonl with each sample. .txt for all sample(will override origin qs)')
    parser.add_argument("--replace-q", action='store_true')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num_video_frames", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--rerun", action='store_true')
    
    #Var
    Use_pkl = False
    args = parser.parse_args()

    #Dataset
    if args.question_file.endswith('.pkl'):
        with open(os.path.expanduser(args.question_file), "rb") as f:
            data = pickle.load(f)
        questions = data #load question from .pkl
        Use_pkl = True
    else:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    if args.max_samples > 0:
        questions = questions[:args.max_samples]
    use_nii = False
    num_video_frames = args.num_video_frames
    question_keys = [k for k in questions[0].keys()]
    if 'image' in question_keys:
        img_key = 'image'
    elif 'filename' in question_keys:
        img_key = 'filename'
    elif "image:" in question_keys:
        img_key = 'image:'
    elif "nii" in question_keys:
        img_key = "nii"
        use_nii = True
    else:
        img_key = "image"
    
    if 'conversations' in question_keys:
        use_conv = True
    else:
        use_conv = False
    
    batch_size = args.batch_size
    #meta data
    sample_meta_data, dict_meta_data, new_question = load_meta_data(args.meta_path)
    use_meta_prefix = False
    use_meta_postfix = False
    if len(sample_meta_data)>0:
        assert len(sample_meta_data)==len(questions)
        use_meta_prefix = True
        if "question_id" in question_keys:
            meta_data_dict = {n["question_id"]: n["text"] for n in sample_meta_data}
        else:
            meta_data_dict = {i+1: sample_meta_data[i]["text"] for i in range(len(sample_meta_data))}
    if len(dict_meta_data)>0:
        use_meta_postfix = True
        meta_data_dict = {}
        for i,each_data in enumerate(questions):
            if 'question_id' in question_keys:
                idx = each_data["question_id"]
            elif 'image_id' in question_keys:
                idx = each_data["image_id"]
            else:
                idx = i
            col_name = sorted(dict_meta_data)[0]
            meta_data_dict[idx] = dict_meta_data[col_name].get(each_data[col_name],'')

    #Make batchs
    questions_batch = get_batch(questions, batch_size)
    print('Question batchs:')
    print(len(questions_batch))
    answers_file = os.path.expanduser(args.answers_file)
    gt_file = os.path.expanduser(args.answers_file.replace('.json','_gt.json'))
    gt_list = []
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    #check answer file or not
    answer_data = []
    skip_q_id = []
    if not args.rerun and os.path.isfile(answers_file):
        with open(answers_file, "r") as f:
            answer_data = [json.loads(n) for n in f]
            skip_q_id = [n["question_id"] for n in answer_data]
            print('Already have %d answers'%len(skip_q_id))
    ans_file = open(answers_file, "w")
    
    #Model
    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)

    model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=model_dtype
    )

    from llava.peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.hlora_r,
        lora_alpha=args.hlora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.hlora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
        lora_nums=args.hlora_nums,
    )
    model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, args.vq_idx_nums)
    print(f"Number of new tokens added for unified task: {num_new_tokens}")

    from utils import com_vision_args
    com_vision_args.model_name_or_path = args.model_name_or_path
    com_vision_args.vision_tower = args.vit_path
    com_vision_args.version = args.instruct_template

    model.get_model().initialize_vision_modules(model_args=com_vision_args)
    model.get_vision_tower().to(dtype=model_dtype)

    model = load_weights(model, args.hlora_path)
    model.eval()
    model.to(model_dtype).cuda()

    #For loop
    i = 0
    for batch in tqdm(questions_batch):
        idx_list = []
        image_tensor_list = []
        prompt_list = []
        input_ids_list = []
        for line in batch:
            i += 1
            if 'question_id' in question_keys:
                idx = line["question_id"]
            elif 'image_id' in question_keys:
                idx = line["image_id"]
            else:
                idx = i
            
            ## load image
            image_file = line[img_key]
            #tmp fix, transport to list
            if '[' in image_file and ']' in image_file and ', ' in image_file:
                image_file = image_file.strip('[').strip(']').split(', ')
            if Use_pkl:
                images = load_pkl_image(image_file)
                image_num = get_image_num(image_file)
            elif use_nii: #TODO: Change implementation
                images, step_ids, image_num = _load_nii(os.path.join(args.image_folder, image_file), num_video_frames,center_image=None,
                                                                                        bounding_box=None,pixel_spacing=None)
            else:
                if isinstance(image_file,list):
                    images = [load_image(os.path.join(args.image_folder, e_image_file)) for e_image_file in image_file]
                else:
                    images = [load_image(os.path.join(args.image_folder, image_file))]
                image_num = get_image_num(image_file)
            
            #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = [expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean)) for image in images]
            print('After expand2square:',images[0].size)
            image_tensor = model.get_vision_tower().image_processor.preprocess(images, return_tensors='pt')['pixel_values'][0]
            print('After preprocess:',image_tensor.shape)
            
            if not use_conv:
                ## load qs and save gt
                if Use_pkl:
                    qs = line['question']
                    gt_dict = {
                        "question_id": idx,
                        'prompt': qs,
                        'answer': line['answer']
                    }
                    gt_list.append(gt_dict)
                else:
                    qs = line["text"]
                #chnage qs if need
                if new_question!=None:
                    if args.replace_q:
                        qs = new_question
                    else:
                        qs = new_question + '\n' + qs
                
                if idx in skip_q_id:
                    ans_idx = skip_q_id.index(idx)
                    ans_dict = answer_data[ans_idx]
                    ans_file.write(json.dumps(ans_dict) + "\n")
                    ans_file.flush()
                    continue

                #add meta data if have
                if use_meta_prefix:
                    sample_meta = meta_data_dict[idx]
                    qs = sample_meta + '\n' + qs
                    if i==1:
                        print('Add meta data to question')
                        print(qs)
                if use_meta_postfix:
                    sample_meta = meta_data_dict[idx]
                    qs = qs + '\n' + sample_meta
                    if i==1:
                        print('Add meta data to question')
                        print(qs)
                #concat with image (add multi image case)
                cur_prompt = qs
                for im_i in range(image_num):
                    if use_nii:
                        qs = '%d : '%(step_ids[im_i]) + DEFAULT_IMAGE_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                conv = conversation_lib.conv_templates[args.instruct_template].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
            else:
                conv = conversation_lib.conv_templates[args.instruct_template]
                ## load conv
                ques_conv = line["conversations"]
                all_images = []
                #Assume last one is answer
                for j in range(len(ques_conv)-1):
                    question = ques_conv[j]["value"]
                    c_role = ques_conv[j]["from"]
                    if c_role == "human":
                        new_role = conv.roles[0]
                    else:
                        new_role = conv.roles[1] 
                    cur_prompt = question #last prompt will be the question
                    if "<image>" in question:
                        #default 3D load
                        add_images, step_ids, num_frames_loaded_successfully = _load_nii(os.path.join(args.image_folder, image_file), num_video_frames,center_image=None,
                                                                                            bounding_box=None,pixel_spacing=None)
                        question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
                        question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
                    elif "<boxed_image>" in question:
                        center_image = line.get("center_image",None) #if center_image in jsonl, use it
                        bounding_box = line.get("box",None) #if bounding_box in jsonl, use it
                        pixel_spacing = line.get("pixel_spacing",None) #if pixel_spacing in jsonl, use it
                        num_video_frames = 5 #TODO: explore longvila for more frames
                        add_images, step_ids, num_frames_loaded_successfully = _load_nii(os.path.join(args.image_folder, image_file), num_video_frames,center_image=center_image,
                                                                                    bounding_box=bounding_box,pixel_spacing=pixel_spacing)
                        question = question.replace("<boxed_image>\n", "").replace("\n<boxed_image>", "").replace("<boxed_image>", "")
                        question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
                    all_images.extend(add_images)
                    images_q = ''.join(['%d : %s\n'%(step_ids[i],DEFAULT_IMAGE_TOKEN) for i in range(num_frames_loaded_successfully)])
                    question = images_q + question
                    conv.append_message(new_role, question)
                conv.append_message(conv.roles[1], None) #add answer
                all_images = [expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean)) for image in all_images]
                image_tensor = model.get_vision_tower().image_processor.preprocess(all_images, return_tensors='pt')['pixel_values']
                #gt save
                gt_dict = {
                        "question_id": idx,
                        'prompt': cur_prompt,
                        'answer': ques_conv[-1]["value"]
                    }
                gt_list.append(gt_dict)
            
            prompt = conv.get_prompt()
            #print(prompt)
            #print(image_tensor.shape)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            #store to list
            idx_list.append(idx)
            image_tensor_list.append(image_tensor)
            prompt_list.append(cur_prompt)
            input_ids_list.append(input_ids)
        
        if len(idx_list)==0:
            continue
        
        #batch part
        input_ids_tensor = torch.cat(input_ids_list)
        image_tensors = torch.cat(image_tensor_list)
        print('input_ids_tensor & image_tensors:',input_ids_tensor.shape,image_tensors.shape)

        with torch.inference_mode():
            output_ids_list = model.generate(
                input_ids_tensor,
                images=[image_tensors.to(dtype=model_dtype, device='cuda', non_blocking=True)],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        outputs_batch = tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)
        

        for _i in range(len(idx_list)):
            idx = idx_list[_i]
            cur_prompt = prompt_list[_i]
            outputs = outputs_batch[_i]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": args.model_name_or_path,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
    
    #save gt if need
    if len(gt_list)>0:
        save_jsonl(gt_file, gt_list)
    ans_file.close()
    
    #process report if need
    req_cols = ['NOTE_ID','Slice_id','question_type']
    if set(req_cols) <= set(question_keys):
        with open(answers_file, "r") as f:
            answer_data = [json.loads(n) for n in f]
        ans_data = pd.DataFrame(answer_data)
        q_data = pd.DataFrame(questions).drop(columns=['text'])
        full_data = pd.merge(q_data, ans_data, on='question_id')
        report_data = process_vqa_report(full_data,ans_key='text')
        report_file = os.path.expanduser(answers_file.replace('.jsonl','_fmtreport.jsonl'))
        save_jsonl(report_file, report_data)
    else:
        print('Question keys need:',req_cols)
        print('Question keys:',question_keys)
    


if __name__ == "__main__":

    infer()