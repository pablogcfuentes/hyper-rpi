#-----------------------------------------------------------------
#-----------------------------------------------------------------
# FUNCIONES Y VARIABLES DE LA CNN BASE
#-----------------------------------------------------------------
#-----------------------------------------------------------------

import math,random,struct,os,time,sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn import preprocessing
import torchvision.transforms as transforms

DATASET='../../data/imagenes_rios/oitaven_river.raw'
GT='../../data/imagenes_rios/oitaven_river.pgm'

EXP=5    # numero de experimentos
EPOCHS=100 # EPOCHS de entrenamiente del clasificador, default=100
SAMPLES=[0.02,0.01] # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.02,0.01) 
PAD=1  # hacemos padding en los bordes para aprovechar todas las muestras
ADA=0  # learning rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR
AUM=0  # aumentado: 0-sin_aumentado, 1-con_aumentado
DET=0  # experimentos: 0-aleatorios, 1-deterministas

#-----------------------------------------------------------------
# FUNCIONES PARA LEER DATASETS Y SELECCIONAR MUESTRAS
#-----------------------------------------------------------------

def read_raw(fichero):
  (B,H,V)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32)
  print('* Read dataset:',fichero)
  print('  B:',B,'H:',H,'V:',V)
  print('  Read:',len(datos))
  # para normalizar el dataset en [0:1] (en esta red no hace falta)
  # datos=preprocessing.minmax_scale(datos)
  datos=datos.reshape(V,H,B)
  datos=torch.FloatTensor(datos)
  return(datos,H,V,B)

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

def select_training_samples(truth,H,V,sizex,sizey,porcentaje):
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
  for i in range(int(sizey/2),V-int(sizey/2)-1):
    for j in range(int(sizex/2),H-int(sizex/2)-1):
      ind=i*H+j
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
      
  # Modificamos la funcion para obtener el conjunto para inferencia
  # Verificamos si estamos usando el 100% de las muestras para el conjunto de test
  if len(train) == 0 and len(val) == 0:
    # Verificamos las clases presentes en el conjunto de test y actualizamos nclases_no_vacias
    for i in range(nclases):
      if any(truth[idx] == i+1 for idx in test):
        nclases_no_vacias += 1
        
  return(train,val,test,nclases,nclases_no_vacias)

def select_patch(datos,sizex,sizey,x,y):
  x1=x-int(sizex/2); x2=x+int(math.ceil(sizex/2));     
  y1=y-int(sizey/2); y2=y+int(math.ceil(sizey/2));
  patch=datos[:,y1:y2,x1:x2]
  return(patch)

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
    datos=self.datos; H=self.H; V=self.V; B=self.B
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

# Convolutional neural network (two convolutional layers y una lineal)
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


