import yaml
from timm._tools import Run

def change_net(net,dataset_path,in_chans):
    # Reading a YAML file
    batch_size = 64
    num_classes = 10
    epoch_repeats = 1
    epochs = 300
    
    dataset_path = dataset_path
    in_chans = in_chans

    path_i = 'timm/config.yaml'
    with open(path_i, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # Modify data
    data['model'] = net
    data['batch_size'] = batch_size
    data['num_classes'] = num_classes   
    data['data_dir'] = dataset_path
    data['val_split'] = 'validation'
    data['workers'] = 4
    data['opt'] = 'sgd'
    data['drop'] = 0.005
    data['img_size'] = 224
    data['in_chans'] = in_chans
    data['input_size'] = [in_chans, 224, 224]
    data['interpolation'] = ''
    data['fuser'] = 'nvfuser'
    data['epoch_repeats'] = epoch_repeats
    data['model_ema'] = False
    data['amp'] = True
    data['pin_mem'] = True
    data['use_multi_epochs_loader'] = True
    data['pretrained'] = False
    data['epochs'] = epochs
    
    # Writing to a YAML file
    with open(path_i, 'w') as file:
        yaml.dump(data, file)
    print("Data has been modified and written to " + path_i + " successfully!")

if __name__ == '__main__':
    # Model List
    ModelNet = [
        'efficientnet_b0',
        'densenet121',
        'resnet18',
        'vgg11',
        'fastvit_s12',
        'tiny_vit_5m_224',
        ]

    set=[
         'A1',
         'A2',
         'A3',
         'Max',
     ]
    feature=[
        'G-GASF',
        'G-GADF',
        'GASF',
        'GADF',
        'MTF',
        'RP',
    ]
    
    path=[]
    for i in set:
        for j in feature:
            path.append(['dataset/ALL/'+i+'/'+'DATA_'+j,1])
    print(path)
    # Run training for each model
    for net_i in ModelNet:
        for p in path:
            change_net(net_i,p[0],p[1])
            print('Begin Training: ',net_i,p[0])
            Run('timm/config.yaml')
            
        
