# -*- coding: utf-8 -*-
def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix #�������������
    cm = confusion_matrix(y, yp) #��������
    import matplotlib.pyplot as plt #������ͼ��
    plt.matshow(cm, cmap=plt.cm.Greens) #����������ͼ����ɫ���ʹ��cm.Greens����������ο�������
    plt.colorbar() #��ɫ��ǩ
    for x in range(len(cm)): #���ݱ�ǩ
      for y in range(len(cm)):
        plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label') #�������ǩ
    plt.xlabel('Predicted label') #�������ǩ
    return plt