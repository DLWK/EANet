from torch import optim
from loss.losses import *
from data.dataloader import ISBI_Loader5, ISBI_Loadertest
import torch.nn as nn
import torch
from .models import  EANet
from torchvision import transforms
from utils.metric import *
from evaluation import *


transform=transforms.Compose([  
            transforms.ToTensor()])
           

def test(testLoader,fold, net, device):
    net.to(device)
    sig = torch.nn.Sigmoid()
    net.eval()
    with torch.no_grad():
         # when in test stage, no grad
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        count = 0
        for image, label, edge in testLoader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred,p1,p2,p3,p4,e= net(image)
          
            sig = torch.nn.Sigmoid()
            pred = sig(pred)
            print(pred.shape)
            acc += get_accuracy(pred,label)
            SE += get_sensitivity(pred,label)
            SP += get_specificity(pred,label)
            PC += get_precision(pred,label)
            F1 += get_F1(pred,label)
            JS += get_JS(pred,label)
            DC += get_DC(pred,label)
            count+=1
        acc = acc/count
        SE = SE/count
        SP = SP/count
        PC = PC/count
        F1 = F1/count
        JS = JS/count
        DC = DC/count
        score = JS + DC
        return  acc, SE, SP, PC, F1, JS, DC, score
               

def train_net(net, device, train_data_path,test_data_path, fold, epochs=40, batch_size=8, lr=0.00001):
    isbi_train_dataset = ISBI_Loader5(train_data_path,transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_dataset = ISBI_Loadertest(test_data_path,transform=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1, 
                                               shuffle=False)
    
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
 

    criterion2 = nn.BCEWithLogitsLoss()

    criterion3 = structure_loss()

    best_loss = float('inf')
    result = 0
    f = open('./finall_loss_unet'+str(fold)+'.csv', 'w')
    f.write('epoch,loss'+'\n')
    for epoch in range(epochs):
        net.train()
        for image, label, edge in train_loader:
            optimizer.zero_grad()

            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32) 
            edge = edge.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred, p1,p2,p3,p4,e= net(image)
            loss2 = criterion3(p1, label)
            loss3 = criterion3(p2, label)
            loss4 = criterion3(p3, label)
            loss5 = criterion3(p4, label)
            loss6 = criterion2(e, edge)
            loss = loss1+(loss2+loss3+loss4+loss5)+loss6 
     
           
            print('Train Epoch:{}'.format(epoch))
            print('Loss/train', loss.item())
            # print('Loss/edge', loss6.item())
           
         
            loss.backward()
            optimizer.step()
            # if epoch==0:
            #     Iou_score, dice_score, sens_score, ppv_score=test(test_loader,fold, net, device) 
            #     with open("./VGG19_unet_metric_"+str(fold)+".csv", "a") as w:
        
           #         w.write("dice="+str(dice_score)+", iou="+str(Iou_score)+",sen="+str(sens_score)+",ppv="+str(ppv_score)+"\n")
        f.write(str(epoch)+","+str(best_loss.item())+"\n")  
        if epoch>0:
            acc, SE, SP, PC, F1, JS, DC, score=test(test_loader,fold, net, device)
            if result < score:
                result = score
                # best_epoch = epoch
                torch.save(net.state_dict(), './EANet/eanet_best_'+str(fold)+'.pth')
                with open("./EANet/eanet_"+str(fold)+".csv", "a") as w:
                    w.write("acc="+str(acc)+", SE="+str(SE)+",SP="+str(SP)+",PC="+str(PC)+",F1="+str(F1)+",JS="+str(JS)+",DC="+str(DC)+",Score="+str(score)+"\n")

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed=1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fold = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    net = EANet(n_channels=1, n_classes=1)
    net.to(device=device)
    data_path = "/home/wangkun/EANet/data_5fold_edge/train_96_"+str(fold)
    test_data_path = "/home/wangkun/EAnet/data_5fold_edge/test_96_"+str(fold)
    train_net(net, device, data_path,test_data_path, fold)

# by kun wang 