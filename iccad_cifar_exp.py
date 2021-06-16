import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pdb
from .binarized_modules import  BinarizeLinear,BinarizeConv2d, BinarizedDenseModule,output_paras_to_w_th_npy, BinarizedConvModule

__all__ = ['iccad_cifar_exp']

class ICCAD_CIFAR_exp(nn.Module):

    def __init__(self, num_classes=1000):
        super(ICCAD_CIFAR_exp, self).__init__()
        num_classes=10
        self.cfg=[(128,1),(128,2),(128,1),(128,2),(128,1),(128,2),(128,1),(128,2)]
        input_wh=32
        prev_channel=3
        final_wh=input_wh
        self.ratioInfl=1
        self.features = nn.ModuleList()
        
        for i,(channel,stride) in enumerate(self.cfg):
            '''
            self.features.append(BinarizeConv2d(prev_channel, channel, kernel_size=3, stride=1, padding=1))
            if (stride!=1):
                self.features.append(nn.MaxPool2d(kernel_size=stride,stride=stride))
            self.features.append(nn.BatchNorm2d(channel))
            self.features.append(nn.Hardtanh(inplace=True))
            '''
            self.features.append(BinarizedConvModule(prev_channel,channel,kernel_size=3,stride=stride,max_pool=True))

            prev_channel=channel
            final_wh=final_wh//stride
        for i in range(len(self.features)-1):
            self.features[i].set_next_module(self.features[i+1])



        self.dfg=[128,128,128,128,num_classes]
        self.classifier = nn.ModuleList()
        dl_num=len(self.dfg)
        self.flattened_num=self.cfg[-1][0]*final_wh*final_wh
        prev_n_num=self.flattened_num

        for i,n_num in enumerate(self.dfg):
            self.classifier.append(BinarizedDenseModule(prev_n_num,n_num,binarize_output=(True if (i!=dl_num-1) else False)))
            prev_n_num=n_num
        for i,layer in enumerate(self.classifier):
            if (i!=dl_num-1):
                layer.set_next_module(self.classifier[i+1])

        self.features[-1].set_next_module(self.classifier[0])
        self.features[-1].wh_multiplier=final_wh*final_wh

        print (self.features)
        print (self.classifier[0])

        '''
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        '''
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-2},
            30: {'lr': 1e-3},
            60: {'lr': 1e-4},
            90: {'lr': 5e-5},
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                #transforms.Scale(256),
                #transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                #transforms.Scale(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):

        for i, layer in enumerate(self.features):
            x=layer(x)
        x = x.view(-1, self.flattened_num)
        for i, layer in enumerate(self.classifier):
            x = layer(x)

        return x

    def output_paras(self, output_name):
        count=1
        #TODO: cifar10 first layer conv threshold still not correct
        for i,layer in enumerate(self.features):
            output_paras_to_w_th_npy(layer.conv,layer.bn,output_name+'_conv{}_w'.format(count),output_name+'_conv{}_th'.format(count),conv=True)
            count+=1

        count=1
        for i,layer in enumerate(self.classifier):
            if (hasattr(layer,'bn')):
                output_paras_to_w_th_npy(layer.fc,layer.bn,output_name+'_fc{}_w'.format(count),output_name+'_fc{}_th'.format(count),conv=False)
            else:
                output_paras_to_w_th_npy(layer.fc,None,output_name+'_fc{}_w'.format(count),output_name+'_fc{}_th'.format(count),conv=False)
            count+=1

    def prune_minabs_gammath_neuron(self,prune_neuron):
        for i,layer in enumerate(self.features):
            layer.prune_minabs_gammath_neuron(prune_neuron)
        for i,layer in enumerate(self.classifier):
            layer.prune_minabs_gammath_neuron(prune_neuron)

    def prune_minabs_neuron(self,prune_neuron):
        for i,layer in enumerate(self.features):
            layer.prune_minabs_neuron(prune_neuron)
        for i,layer in enumerate(self.classifier):
            layer.prune_minabs_neuron(prune_neuron)

    def prune_random_neuron(self, prune_neuron):
        #print (idx)
        for i,layer in enumerate(self.features):
            layer.prune_random_neuron(prune_neuron)
        for i,layer in enumerate(self.classifier):
            layer.prune_random_neuron(prune_neuron)
            
    def prune_random_entry(self, prune_neuron):
        prev_prune_neuron=0
        for i,layer in enumerate(self.features):
            layer.prune_random_entry(prune_neuron,prev_prune_neuron)
            prev_prune_neuron=layer.pruned_neuron

        prev_prune_neuron=0
        for i,layer in enumerate(self.classifier):
            layer.prune_random_entry(prune_neuron,prev_prune_neuron)
            prev_prune_neuron=layer.pruned_neuron

    def prune_minabs_entry(self,prune_neuron):
        prev_prune_neuron=0
        for i,layer in enumerate(self.features):
            layer.prune_minabs_entry(prune_neuron,prev_prune_neuron)
            prev_prune_neuron=layer.pruned_neuron

        prev_prune_neuron=0
        for i,layer in enumerate(self.classifier):
            layer.prune_minabs_entry(prune_neuron,prev_prune_neuron)
            prev_prune_neuron=layer.pruned_neuron

    def return_prune_ratio(self):
        total_pruned=0
        total_para_num=0
        for i,layer in enumerate(self.features):
            total_pruned+=np.sum(layer.conv.w_mask.data.cpu().numpy())
            total_para_num+=layer.conv.w_mask.numel()
        for i,layer in enumerate(self.classifier):
            total_pruned+=np.sum(layer.fc.w_mask.data.cpu().numpy())
            total_para_num+=layer.fc.w_mask.numel()
        #pdb.set_trace()
       
        return total_pruned/total_para_num


        
def iccad_cifar_exp(**kwargs):
    return ICCAD_CIFAR_exp(10)
