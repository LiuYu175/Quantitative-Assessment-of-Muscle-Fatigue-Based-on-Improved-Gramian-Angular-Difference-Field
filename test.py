import os
import time
import pandas as pd
from timm.data import create_dataset, create_loader
from timm.models import create_model
import torch
import pycm
import yaml

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass

def load_model(train_folder):
    '''Load model from the training folder'''
    file=os.listdir(train_folder)
    device = torch.device('cuda')
    
    yaml_path = train_folder+'/'+file[0]
    with open(yaml_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    channel=data['in_chans']
    input_size=data['input_size']
    model_name=data['model']
    num_classes=data['num_classes']
    data_name=data['data_dir'][-6:]
    data_path=data['data_dir']  
    
    for n in range(len(data_name)):
        if data_name[n]=='_':
            data_name=data_name[n+1:]
            break
    
    checkpoint = train_folder + '\\model_best.pth.tar'
    print('Loading model:', model_name, 'Data:', data_name)
   
    model = create_model(
        model_name,
        num_classes=num_classes,
        in_chans=channel,
        pretrained=False,
        checkpoint_path=checkpoint,
        # input_size=input_size
    )
    model = model.to(device)
    model.eval()
    return model , data_name ,model_name,num_classes,input_size,data_path
   
def CMF(root_dir,model,input_size):
    dataset = create_dataset(
            root=root_dir,
            name='',
            input_img_mode=None,
         )
    
    loader = create_loader(
            dataset,
            batch_size=1,
            use_prefetcher=True,
            num_workers=1,
            pin_memory=True,
            is_training=False,
            input_size=input_size,
            use_multi_epochs_loader=True,
            no_aug=False,
            re_prob=0.0,
            re_mode='pixel',
            re_count=1,
            re_split=False,
            train_crop_mode='',
            scale=[0.08,1.0],
            ratio=[0.75,1.3333],
            hflip=0.5,
            vflip=0.0,
            distributed=False,
        )
    
    time1 = time.time()
    index=[]
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            output = model(input)
            output = output.softmax(-1)
            output, indices = output.topk(1)
            np_indices = indices.cpu().numpy()
            index.append(np_indices[0][0])
            
    time2 = time.time()
    real=[]
    file=os.listdir(root_dir)
    for f in file:
        name=os.listdir(root_dir+'\\'+f)
        for i in name:
            real.append(int(f[0]))
    
    cm = pycm.ConfusionMatrix(actual_vector=real, predict_vector=index)
    return cm.Overall_ACC,cm.F1_Macro,(time2-time1)/len(real)*1000

if __name__ == '__main__':
    '''
        Just give the path of the output to get the test results of the model
    '''
    file_path='output/train'
    create_folder_if_not_exists('output/test')

    ACC={}
    F1={}
    Time={}
    for name in os.listdir(file_path):
        model,data_name,model_name,num_classes,input_size,data_path=load_model(file_path+'\\'+name)
        print('Model:',model_name,'Data:',data_name,'Num_classes:',num_classes,'Input_size:',input_size,'Data_path:',data_path)
        root_dir = data_path+'/test'

        set = data_path.split('/')[2]
        method = data_path.split('/')[-1][5:]

        name = set + '_' + method
        test_acc,test_f1,test_time=CMF(root_dir,model,input_size)

        try:
            ACC[name][model_name]=test_acc
            F1[name][model_name]=test_f1
            Time[name][model_name]=test_time
        except:
            ACC[name]={model_name:test_acc}
            F1[name]={model_name:test_f1}
            Time[name]={model_name:test_time}
    
    ACC=pd.DataFrame(ACC)
    ACC.to_csv('output/test/ACC.csv')
    F1=pd.DataFrame(F1)
    F1.to_csv('output/test/F1.csv')
    Time=pd.DataFrame(Time)
    Time.to_csv('output/test/Time.csv')

    print('All test results were successfully exported!')