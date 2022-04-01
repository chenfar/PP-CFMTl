import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str, default='performance-mnist')
parser.add_argument('--filename', type=str, default='fig')

if __name__ == '__main__':
    args = parser.parse_args()

    record = np.load('experiments/{}.npy'.format(args.filename))
    record = record.tolist()
    
    if args.experiment == 'performance-mnist':
        x_0 = x_1 = x_2 = range(len(record[0][0]))
        y1_0 = record[0][0]
        y1_1 = record[0][1]    
        y1_2 = record[0][2]
        y2_0 = record[1][0]    
        y2_1 = record[1][1]    
        y2_2 = record[1][2]
        
        plt.figure(figsize=(6.4,12.8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_0, y1_0, '-', marker = 'o', mec = 'r', mfc = 'w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y1_1, '-', marker = 's', mec = 'g', mfc = 'w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y1_2, '-', marker = '^', mec = 'b', mfc = 'w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(80, 100)
        
        plt.subplot(2, 1, 2)
        plt.plot(x_0, y2_0, '-', marker = 'o', mec = 'r', mfc = 'w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y2_1, '-', marker = 's', mec = 'g', mfc = 'w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y2_2, '-', marker = '^', mec = 'b', mfc = 'w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('proportion', fontsize=20)
        plt.ylim(0, 100)
        plt.xlabel('round', fontsize=20)
        
        plt.legend(fontsize=20)
        print(y1_0[-1])
        print(y1_1[-1])
        print(y1_2[-1])
        print(y2_0[-1])
        print(y2_1[-1])
        print(y2_2[-1])

    if args.experiment == 'performance-cifar':
        x_0 = x_1 = x_2 = range(len(record[0][0]))
        y1_0 = record[0][0]
        y1_1 = record[0][1]    
        y1_2 = record[0][2]
        
        plt.figure(figsize=(6.4,6.4))
        
        plt.subplot(1, 1, 1)
        plt.plot(x_0, y1_0, '-', marker = 'o', mec = 'r', mfc = 'w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y1_1, '-', marker = 's', mec = 'g', mfc = 'w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y1_2, '-', marker = '^', mec = 'b', mfc = 'w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(0, 100)
        
        plt.legend(fontsize=20)
        print(y1_0[-1])
        print(y1_1[-1])
        print(y1_2[-1]) 

    if args.experiment == 'communication':

        y_0 = record[1][1]
        y_1 = record[1][0]
        x_0 = record[0][1]
        x_1 = record[0][0]      
        
        plt.figure(figsize=(6.4,6.4))
        
        plt.subplot(1, 1, 1)
        plt.plot(x_0, y_0, '-', marker = 'o', mec = 'r', mfc = 'w', label='FMTL', color='r', markersize = 10)
        plt.plot(x_1, y_1, '-', marker = 's', mec = 'g', mfc = 'w', label='CFMTL', color='g', markersize = 10)
        plt.xlabel('cost', fontsize=20)
        plt.xlim(0, 700)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(0, 100)

        plt.legend(fontsize=20)
        for i in range(len(x1_0)):
            if x_0[i] >= 100:
                print(y_0[i])
                break
        for i in range(len(x1_1)):
            if x_1[i] >= 100:
                print(y_1[i])
                break

    if args.experiment == 'hyperparameters':
        x_0 = x_1 = x_2 = range(len(record[0][0]))
        y1_0 = record[0][0]
        y1_1 = record[0][1]    
        y1_2 = record[0][2]
        y2_0 = record[1][0]    
        y2_1 = record[1][1]    
        y2_2 = record[1][2]
        
        plt.figure(figsize=(6.4,12.8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_0, y1_0, '-', marker = 'o', mec = 'r', mfc = 'w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y1_1, '-', marker = 's', mec = 'g', mfc = 'w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y1_2, '-', marker = '^', mec = 'b', mfc = 'w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(80, 100)
        
        plt.subplot(2, 1, 2)
        plt.plot(x_0, y2_0, '-', marker = 'o', mec = 'r', mfc = 'w', label='FedAvg', color='r', markersize=10)
        plt.plot(x_1, y2_1, '-', marker = 's', mec = 'g', mfc = 'w', label='CFL', color='g', markersize=10)
        plt.plot(x_2, y2_2, '-', marker = '^', mec = 'b', mfc = 'w', label='CFMTL', color='b', markersize=10)
        plt.ylabel('proportion', fontsize=20)
        plt.ylim(0, 100)
        plt.xlabel('round', fontsize=20)
        
        plt.legend(fontsize=20)
        print(y1_0[-1])
        print(y1_1[-1])
        print(y1_2[-1])
        print(y2_0[-1])
        print(y2_1[-1])
        print(y2_2[-1])     

    if args.experiment == 'metric':
        x_0 = x_1 = x_2 = x_3 = range(len(record[0][0]))
        y1_0 = record[0][0]
        y1_1 = record[0][1]    
        y1_2 = record[0][2]
        y1_3 = record[0][3]
        y2_0 = record[1][0]    
        y2_1 = record[1][1]    
        y2_2 = record[1][2]
        y2_3 = record[1][3]
        
        plt.figure(figsize=(6.4,12.8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x_0, y1_0, '-', marker = 'o', mec = 'r', mfc = 'w', label='L2', color='r', markersize = 10)
        plt.plot(x_1, y1_1, '-', marker = 's', mec = 'g', mfc = 'w', label='Equal', color='g', markersize = 10)
        plt.plot(x_2, y1_2, '-', marker = '^', mec = 'b', mfc = 'w', label='L1', color='b', markersize = 10)
        plt.plot(x_3, y1_3, '-', marker = 'h', mec = 'y', mfc = 'w', label='Cos', color='y', markersize = 10)
        plt.ylabel('accuracy', fontsize=20)
        plt.ylim(80, 100)
        
        plt.subplot(2, 1, 2)
        plt.plot(x_0, y2_0, '-', marker = 'o', mec = 'r', mfc = 'w', label='L2', color='r', markersize = 10)
        plt.plot(x_1, y2_1, '-', marker = 's', mec = 'g', mfc = 'w', label='Equal', color='g', markersize = 10)
        plt.plot(x_2, y2_2, '-', marker = '^', mec = 'b', mfc = 'w', label='L1', color='b', markersize = 10)
        plt.plot(x_3, y2_3, '-', marker = 'h', mec = 'y', mfc = 'w', label='Cos', color='y', markersize = 10)
        plt.ylabel('proportion', fontsize=20)
        plt.ylim(0, 100)
        plt.xlabel('round', fontsize=20)
        
        plt.legend(fontsize=20)
        print(y1_0[-1])
        print(y1_1[-1])
        print(y1_2[-1])
        print(y1_3[-1])
        print(y2_0[-1])
        print(y2_1[-1])
        print(y2_2[-1])
        print(y2_3[-1])          
    
    plt.savefig('experiments/{}.eps'.format(args.filename))

        

