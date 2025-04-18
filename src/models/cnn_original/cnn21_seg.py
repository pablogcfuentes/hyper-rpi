#!/usr/bin/env python3
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials (pytorch-tutorial-master.zip)
# Adapted to multi/hyperspectral images: F. Arguello
# CNN21: 2 capas convolucionales, 1 completamente conectada
# oitaven WP (15%, texturas+fv+3kelm, t=3m44s): OA=93.03, OA=87.18
# CNN21 SEG EXP: 5 EPOCHS: 100 SAMPLES: [0.15, 0.0] ADA: 1 AUM: 1
# Class 01: 96.28+0.94
# Class 02: 77.81+2.94
# Class 03: 75.81+2.82
# Class 04: 83.73+3.94
# Class 05: 82.23+4.11
# Class 06: 90.06+1.44
# Class 07: 95.78+1.15
# Class 08: 94.88+0.99
# Class 09: 98.36+1.07
# Class 10: 90.93+1.05
# OA=94.01+0.26, AA=88.59+0.54

import math, random, struct, signal, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn import preprocessing
import torchvision.transforms as transforms

EXP=5      # numero de experimentos
EPOCHS=100 # EPOCHS de entrenamiente del clasificador, default=100
SAMPLES=[0.15,0.0] # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.15,0.05) 
ADA=1  # learning rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR
AUM=1  # aumentado: 0-sin_aumentado, 1-con_aumentado
DET=0  # experimentos: 0-aleatorios, 1-deterministas

DATASET='../../data/imagenes_rios/oitaven_river.raw'
GT='../../data/imagenes_rios/oitaven_river.pgm'
SEG='../../data/imagenes_rios/seg_oitaven_wp.raw'
CENTER='../../data/imagenes_rios/seg_oitaven_wp_centers.raw'

# DATASET='/home/amo/profile.raw'
# GT='/mnt/media/images/salinas_gt.pgm'
# SEG='/home/amo/seg.raw'
# CENTER='/mnt/media/images/seg_salinas_centers.raw'

# DATASET='/mnt/media/images/ermidas_creek.raw'
# GT='/mnt/media/images/ermidas_creek.pgm'
# SEG='/mnt/media/images/seg_ermidas.raw'
# CENTER='/mnt/media/images/seg_ermidas_centers.raw'

#-----------------------------------------------------------------
# FUNCIONES PARA LEER DATASETS Y SELECCIONAR MUESTRAS
#-----------------------------------------------------------------

def read_raw(fichero):
  (B,H,V)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32)
  print('* Read dataset:',fichero)
  print('  B:',B,'H:',H,'V:',V)
  print('  Read:',len(datos))
  # esta red no necesita realmente normalizar
  datos=preprocessing.minmax_scale(datos)
  print('  min:',datos.min(),'max:',datos.max())
  datos=datos.reshape(V,H,B)
  datos=torch.FloatTensor(datos)
  return(datos,H,V,B)

def read_seg(fichero):
  (H,V)=np.fromfile(fichero,count=2,dtype=np.uint32)
  datos=np.fromfile(fichero,count=H*V,offset=2*4,dtype=np.uint32)
  print('* Read segmentation:',fichero)
  print('  H:',H,'V:',V)
  print('  Read:',len(datos))
  return(datos,H,V)

def read_seg_centers(fichero):
  (H,V,nseg)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=H*V,offset=3*4,dtype=np.uint32)
  print('* Read centers:',fichero)
  print('  H:',H,'V:',V,'nseg',nseg)
  print('  Read:',len(datos))
  return(datos,H,V,nseg)

def save_raw(output,H,V,B,filename):
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    f.write(struct.pack('i',B))
    f.write(struct.pack('i',H))
    f.write(struct.pack('i',V))
    output=output.reshape(H*V*B)
    for i in range(H*V*B):
      f.write(struct.pack('i',np.int(output[i])))
    f.close()
    print('* Saved file:',filename)

def read_pgm(fichero):
  try:
    pgmf=open(fichero,"rb")
  except IOError:
    print('No puedo abrir ',fichero)
  else:
    assert pgmf.readline().decode()=='P5\n'
    line=pgmf.readline().decode()
    while(line[0]=='#'):
      line=pgmf.readline().decode()
    (H,V)=line.split()
    H=int(H); V=int(V)
    depth=int(pgmf.readline().decode())
    assert depth<=255
    raster=[]
    for i in range(H*V):
      raster.append(ord(pgmf.read(1)))
    print('* Read GT:',fichero)
    print('  H:',H,'V:',V,'depth:',depth)
    print('  Read:',len(raster))
    return(raster,H,V)

def save_pgm(output,H,V,nclases,filename):
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    # f.write(b'P5\n')
    cadena='P5\n'+str(H)+' '+str(V)+'\n'+str(nclases)+'\n'
    f.write(bytes(cadena,'utf-8'))
    f.write(output)
    f.close()
    print('* Saved file:',filename)

def select_patch(datos,sizex,sizey,x,y):
  x1=x-int(sizex/2); x2=x+int(math.ceil(sizex/2));     
  y1=y-int(sizey/2); y2=y+int(math.ceil(sizey/2));
  patch=datos[:,y1:y2,x1:x2]
  return(patch)

# Esta parte tarda mucho, mejor la preprocesamos en C
def seg_center(seg,H,V):
  print('* Segment centers (tarda mucho)')
  nseg=0
  for i in range(H*V):
    if(seg[i]>nseg): nseg=seg[i]
  nseg=nseg+1
  print('  segments:',nseg)
  xmin=[H*V]*nseg; xmax=[0]*nseg; 
  ymin=[H*V]*nseg; ymax=[0]*nseg; 
  for i in range(H*V):
    x=i%H; y=i//H; s=seg[i]
    if(x<xmin[s]): xmin[s]=x
    if(y<ymin[s]): ymin[s]=y
    if(x>xmax[s]): xmax[s]=x
    if(y>ymax[s]): ymax[s]=y
  center=np.zeros(nseg,dtype=np.uint32)
  for s in range(nseg):
    y=(ymin[s]+ymax[s])//2; x=(xmin[s]+xmax[s])//2; 
    center[s]=y*H+x
  return(center,nseg)

def select_training_samples_seg(truth,center,H,V,sizex,sizey,porcentaje):
  print('* Select training samples')
  # hacemos una lista con las clases, pero puede haber clases vacias
  nclases=0; nclases_no_vacias=0
  N=len(truth)
  for i in truth:
    if(i>nclases): nclases=i
  print('  nclasses:',nclases)
  lista=[0]*nclases;
  for i in range(nclases):
    lista[i]=[]
  xmin=int(sizex/2); xmax=H-int(math.ceil(sizex/2))
  ymin=int(sizey/2); ymax=V-int(math.ceil(sizey/2))
  for ind in center:
    i=ind//H; j=ind%H;
    if(i<ymin or i>ymax or j<xmin or j>xmax): continue
    if(truth[ind]>0): lista[truth[ind]-1].append(ind)
  for i in range(nclases):
    random.shuffle(lista[i])
  # seleccionamos muestras para train, validacion y test
  print('  Class  # :   total | train |   val |    test')
  train=[]; val=[]; test=[]
  for i in range(nclases):
    # tot0: numero muestras entrenamiento, tot1: validacion 
    if(porcentaje[0]>=1): tot0=porcentaje[0]
    else: tot0=int(porcentaje[0]*len(lista[i]))
    if(tot0>=len(lista[i])): tot0=len(lista[i])//2
    if(tot0<0 and len(lista[i])>0): tot0=1
    if(tot0!=0): nclases_no_vacias+=1
    if(porcentaje[1]>=1): tot1=porcentaje[1]
    else: tot1=int(porcentaje[1]*len(lista[i]))
    if(tot1>=len(lista[i])-tot0): tot1=(len(lista[i])-tot0)//2
    if(tot1<1 and len(lista[i])>0): tot1=0
    for j in range(len(lista[i])):
      if(j<tot0): train.append(lista[i][j])
      elif(j<tot0+tot1): val.append(lista[i][j])
      else: test.append(lista[i][j])
    print('  Class',f'{i+1:2d}',':',f'{len(lista[i]):7d}','|',f'{tot0:5d}','|',
      f'{tot1:5d}','|',f'{len(lista[i])-tot0-tot1:7d}')
  return(train,val,test,nclases,nclases_no_vacias)

#-----------------------------------------------------------------
# PYTORCH - SETS
#-----------------------------------------------------------------

# cogemos muestras sin ground-truth (dadas por el indice samples)
class HyperAllDataset(Dataset):
  def __init__(self,datos,samples,H,V,sizex,sizey):
    self.datos=datos; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.transform=transforms.Compose(
      [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    datos=self.datos; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    patch=select_patch(datos,sizex,sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    return(patch)

#----------------

# cogemos muestras con ground-truth (dadas por el indice samples)
class HyperDataset(Dataset):
  def __init__(self,datos,truth,samples,H,V,sizex,sizey):
    self.datos=datos; self.truth=truth; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.transform=transforms.Compose(
      [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    datos=self.datos; truth=self.truth; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    patch=select_patch(datos,sizex,sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    # renumeramos porque la red clasifica tambien la clase 0 
    return(patch,truth[self.samples[idx]]-1)

#-----------------------------------------------------------------
# PYTORCH - UTIL
#-----------------------------------------------------------------

# pulsando CNLT-C acabamos el entrenamiento y pasamos a testear
def signal_handler(sig, frame):
  print('\n* Ctrl+C. Exit training')
  global endTrain
  endTrain=True

# For updating learning rate manual
def update_lr(optimizer,lr):    
  for param_group in optimizer.param_groups:
    param_group['lr']=lr

# calcula los promedios de precisiones
def accuracy_mean_deviation(OA,AA,aa):
  n=len(OA); nclases=len(aa[0])
  print('* Means and deviations (%d exp):'%(n))
  # medias
  OAm=0; AAm=0; aam=[0]*nclases;
  for i in range(n):
     OAm+=OA[i]; AAm+=AA[i]
     for j in range(1,nclases): aam[j]+=aa[i][j]
  OAm/=n; AAm/=n
  for j in range(1,nclases): aam[j]/=n
  # desviaciones, usamos la formula que divide entre (n-1)
  OAd=0; AAd=0; aad=[0]*nclases
  for i in range(n):
     OAd+=(OA[i]-OAm)*(OA[i]-OAm); AAd+=(AA[i]-AAm)*(AA[i]-OAm)
     for j in range(1,nclases): aad[j]+=(aa[i][j]-aam[j])*(aa[i][j]-aam[j])
  OAd=math.sqrt(OAd/(n-1)); AAd=math.sqrt(AAd/(n-1))
  for j in range(1,nclases): aad[j]=math.sqrt(aad[j]/(n-1))
  for j in range(1,nclases): print('  Class %02d: %02.02f+%02.02f'%(j,aam[j],aad[j]))
  print('  OA=%02.02f+%02.02f, AA=%02.02f+%02.02f'%(OAm,OAd,AAm,AAd))

#-----------------------------------------------------------------
# PYTORCH - NETWORK
#-----------------------------------------------------------------

# Convolutional neural network (two convolutional layers)
class CNN21(nn.Module):
  def __init__(self,N1,N2,N3,N4,N5,D1,D2):
    super(CNN21,self).__init__()
    self.layer1=nn.Sequential(
      nn.Conv2d(N1,N2,kernel_size=3,stride=1,padding=2),
      nn.BatchNorm2d(N2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=D1))
    self.layer2=nn.Sequential(
      nn.Conv2d(N2,N3,kernel_size=5,stride=1,padding=2),
      nn.BatchNorm2d(N3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=D2))
    self.fc=nn.Linear(N4,N5)
      
  def forward(self,x):
    out=self.layer1(x)
    out=self.layer2(out)
    out=out.reshape(out.size(0),-1)
    out=self.fc(out)
    return out

#-----------------------------------------------------------------
# PYTORCH - MAIN
#-----------------------------------------------------------------

def main(exp):
  print('* CNN21 exp: '+str(exp))
  time_start=time.time()
  # 1. Device configuration
  cuda=True if torch.cuda.is_available() else False
  device=torch.device('cuda' if cuda else 'cpu')
  if torch.backends.cudnn.is_available():
    print('* Activando CUDNN')
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.beBhmark=True
  # experimentos deterministas o aleatorios
  if(DET==1):
    SEED=0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if(cuda==False):
      torch.use_deterministic_algorithms(True)
      g=torch.Generator(); g.manual_seed(SEED)
    else:
      torch.backends.cudnn.deterministic=True
      torch.backends.cudnn.benchmark=False

  # 2. Load datos
  (datos,H,V,B)=read_raw(DATASET)
  (truth,H1,V1)=read_pgm(GT)
  (seg,H2,V2)=read_seg(SEG)
  # necesitamos los datos en band-vector para hacer convoluciones
  datos=np.transpose(datos,(2,0,1))
  # durante la ejecucion de la red vamos a coger patches de tamano cuadrado
  sizex=32; sizey=32 

  # 3. Selection training,testing sets
  # (center,nseg)=seg_center(seg,H,V) # lento, mejor lo cargamos hecho
  (center,H3,V3,nseg)=read_seg_centers(CENTER)
  (train,val,test,nclases,nclases_no_vacias)=select_training_samples_seg(truth,center,H,V,sizex,sizey,SAMPLES)
  dataset_train=HyperDataset(datos,truth,train,H,V,sizex,sizey)
  print('  - train dataset:',len(dataset_train))
  dataset_test=HyperDataset(datos,truth,test,H,V,sizex,sizey)
  print('  - test dataset:',len(dataset_test))
  # Dataloader
  batch_size=100 # defecto 100
  train_loader=DataLoader(dataset_train,batch_size,shuffle=True)
  test_loader=DataLoader(dataset_test,batch_size,shuffle=False)
  # Si queremos validacion
  if(len(val)>0):
    dataset_val=HyperDataset(datos,truth,val,H,V,sizex,sizey)
    print('  - val dataset:',len(dataset_val))
    val_loader=DataLoader(dataset_val,batch_size,shuffle=False)
 
  # 4. Hyper parameters
  if(ADA==0): lr=0.001
  else: lr=0.001
 
  # 5. Red: CNN con dos capas convolucionales y una lineal
  # 5.1. capa conv.1
  N1=B          # dimension de entrada
  D1=2          # decimacion, por defecto 2
  H1=sizex      # lado patches entrada, por defecto 28 (sizex=sizey)
  N2=16         # dimension de salida (seleccionada), por defecto 16
  H2=int(H1/D1) # lado patches salida (calculada), por defecto 28 (sizex=sizey)

  # 5.2. capa conv.2, parametros de entrada N2,H2 vienen dados por la capa anterior
  N3=32         # dimension de salida (seleccionada), por defecto 32
  D2=2          # decimacion, por defecto 2
  H3=int(H2/D2) # lado patches salida
    
  # 5.3. capa completamente conectada, parametro de entrada N4 viene de la etapa anterior 
  N4=H3*H3*N3   # dimension de entrada
  N5=nclases    # dimension de salida
  
  model=CNN21(N1,N2,N3,N4,N5,D1,D2).to(device)

  # 6. Loss, optimizer, and scheduler
  # 6.1 mean-squared error loss
  criterion=nn.CrossEntropyLoss()
  # 6.2 create an optimizer object: Adam optimizer with learning rate lr
  optimizer=torch.optim.Adam(model.parameters(),lr=lr)

  # 6.3 scheduler (no es estrictamente necesario)
  if(ADA==2): scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[EPOCHS//2,(5*EPOCHS)//6],gamma=0.1)
  elif(ADA==3): scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=0, verbose=True)
  elif(ADA==4): scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, verbose=True)
  else: pass

  # 7. Train the model
  print('* Train CNN21, exp.%d'%(exp))
  global endTrain
  endTrain=False
  # signal.signal(signal.SIGINT,signal_handler)
  total_step=len(train_loader)
  for epoch in range(EPOCHS):
    for i,(inputs,labels) in enumerate(train_loader):
      # 7.1. Cogemos muestras para entrenar
      inputs=inputs.to(device)
      labels=labels.to(device)
      
      # 7.2. Forward pass
      outputs=model(inputs)
      loss=criterion(outputs,labels)
      
      # 7.3. Backward and optimize
      # 7.3.1. reset the gradients (PyTorch accumulates gradients on subsequent backward passes)
      optimizer.zero_grad()
      # 7.3.2. compute accumulated gradients
      loss.backward()
      # 7.3.3. perform parameter update based on current gradients
      optimizer.step()
       
    # si tenemos validacion usamos estas muestras, si no el propio train
    if(len(val)>0):
      if(epoch%10==0 or epoch==EPOCHS-1):
        for i,(inputs,labels) in enumerate(val_loader):
          inputs=inputs.to(device)
          labels=labels.to(device)
          outputs=model(inputs)
          loss_val=criterion(outputs,labels)
        print ('  Epoch: %3d/%d, Loss(train): %.4f, Loss(val): %.4f'
          %(epoch,EPOCHS,loss.item(),loss_val.item()))
    else: 
      if(epoch%10==0 or epoch==EPOCHS-1):
        print ('  Epoch: %3d/%d, Loss: %.4f'%(epoch,EPOCHS,loss.item()))

    # Decay learning rate (lo decrementamos cconforme aumentan las iteraciones)
    if(ADA==1 and (epoch+1)%20==0): lr/=2; update_lr(optimizer,lr)
    elif(ADA>1): scheduler.step()
    if(endTrain): break

  # 8. Test the model
  print('* Test CNN21, exp.%d'%(exp))
  output=np.zeros(H*V,dtype=np.uint8) # mapa de salida de pixels
  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
  model.eval()
  with torch.no_grad():
    correct=0; total=0;
    for(inputs,labels) in test_loader:
      inputs=inputs.to(device)
      labels=labels.to(device)
      outputs=model(inputs)
      (_,predicted)=torch.max(outputs.data,1)
      predicted_cpu=predicted.cpu()
      for i in range(len(predicted_cpu)):
        # queremos que las clases comiencen en 1 en vez de 0
        output[test[total+i]]=np.uint8(predicted_cpu[i]+1)
      total+=labels.size(0)
      if(total%2000==0): print('  Test: %6d/%d'%(total,len(dataset_test)))
  print('* Generating classif.map')
  for i in range(H*V): output[i]=output[center[seg[i]]]
  # eliminamos los centros usados en el entrenamiento
  for i in train: output[i]=0
  
  # 9. precisiones por segmentos (excluyendo los usados en el entrenamiento)
  correct=0; total=0
  for i in range(len(center)):
    if(output[center[i]]==0): continue
    total+=1
    if(output[center[i]]==truth[center[i]]): correct=correct+1
  acc=100*correct/total;
  print('* Accuracy (segments): %.02f'%(acc))

  # 10. precisiones a nivel de pixel
  correct=0; total=0; AA=0; OA=0
  class_correct=[0]*(nclases+1)
  class_total=[0]*(nclases+1)
  class_aa=[0]*(nclases+1)
  for i in range(len(output)):
    if(output[i]==0 or truth[i]==0): continue
    total+=1; class_total[truth[i]]+=1
    if(output[i]==truth[i]):
      correct+=1
      class_correct[truth[i]]+=1
  for i in range(1,nclases+1):
    if(class_total[i]!=0): class_aa[i]=100*class_correct[i]/class_total[i]
    else: class_aa[i]=0
    AA+=class_aa[i]
  OA=100*correct/total; AA=AA/nclases_no_vacias 
  print('* Accuracy (pixels) exp.%d:'%(exp))
  for i in range(1,nclases+1): print('  Class %02d: %02.02f'%(i,class_aa[i]))
  print('* Accuracy (pixels) exp.%d, OA=%02.02f, AA=%02.02f'%(exp,OA,AA))
  print('  total:',total,'correct:',correct)

  # guardamos la salida
  save_pgm(output,H,V,nclases,'/home/amo/output_cnn21-'+str(exp)+'.pgm')
  # Save the model checkpoint
  torch.save(model.state_dict(),'/tmp/model_cnn21-'+str(exp)+'.ckpt')

  time_end=time.time()
  print('* Execution time: %.0f s'%(time_end-time_start))
  print('  lr:',lr,'BATCH:',batch_size)
  return(OA,AA,class_aa)

if __name__=='__main__':
  OA=[0]*EXP; AA=[0]*EXP; aa=[0]*EXP 
  for exp in range(EXP): (OA[exp],AA[exp],aa[exp])=main(exp)
  if(EXP>1): accuracy_mean_deviation(OA,AA,aa) 
  print('* CNN21 SEG EXP:',EXP,'EPOCHS:',EPOCHS,'SAMPLES:',SAMPLES,'ADA:',ADA,'AUM:',AUM)
