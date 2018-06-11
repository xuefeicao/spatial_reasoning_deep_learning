import numpy as np
import xlwt
import matplotlib.pyplot as plt
workbook = xlwt.Workbook() 
sheet = workbook.add_sheet("results_for_vgg16_svrt") 
sheet.write(0, 0, 'problem')
sheet.write(0, 1, 'step')
sheet.write(0, 2, 'accuracy')
sheet.write(0, 3, 'total steps for training')
sheet.write(0, 4, 'batch_size=32')
Accs=[]
for i in range(23):
  i=i+1
  file='/media/data/xuefei/svrt/results/'+str(i)+'/validation_accuracies.npz'
  data=np.load(file)
  data_1=data['ckpt_accs']
  data_2=data['ckpt_names']
  data_1=np.sum(data_1,1)/data_1.shape[1]
  j=np.argmax(data_1)
  acc=data_1[j]
  Accs.append(acc)
  step=int(data_2[j])
  t_step=int(data_2[-1])
  sheet.write(i,0,i)
  sheet.write(i,1,step)
  sheet.write(i,2,acc)
  sheet.write(i,3,t_step)
workbook.save("svrt_vgg.xls") 
fig = plt.figure()
N = len(Accs)
x = np.array(range(1,N+1))
y=[i+0.5 for i in range(1,N+1)]
width = 1/1.5
plt.bar(x, Accs, width, color="blue")
my_xticks=[str(i) for i in range(1,N+1)]
plt.xticks(y, my_xticks)
fig.savefig('vgg_16.png', dpi=fig.dpi)



  
  
  
  
