import numpy as np
import matplotlib.pyplot as plt

def L(y, y_): return -(y * np.log(y_ + 1e-9)).sum()

def softmax(x):
    exp = np.exp(x)
    return  exp / exp.sum()

def example(x, y, w, b):
    fig, axes = plt.subplots(1, 7, figsize=(12,6), gridspec_kw={'width_ratios': [4]+[1]*6})
    ax_w, ax_x, ax_b, ax_y_, ax_sm, ax_y, ax_l = axes
    b = b[:,None]
    y = y[:,None]
    
    def mplot(x, ax, title):
        ax.matshow(x, cmap=plt.cm.Reds, alpha=0.5)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ax.annotate(x[i,j], xy=(j,i), fontsize=15,
                            horizontalalignment='center',
                            verticalalignment='center')
        ax.set_title(title)
        ax.set_axis_off()
    
    mplot(w, ax_w, '$ W $')
    
    mplot(x, ax_x, '$ x $')
    
    mplot(b, ax_b, '$ b $')
    
    y_ = (w@x + b).round(2)
    mplot(y_, ax_y_, '$ Wx+b $')
    
    sm = softmax(y_).round(2)
    mplot(sm, ax_sm, '$ \hat{y} $ \n $ softmax(Wx+b) $')
    
    mplot(y, ax_y, '$ y $')

    loss = np.asarray([[L(y[:,0], sm[:,0])]]).round(2)
    mplot(loss, ax_l, '$ loss $')
    
    plt.tight_layout()
    plt.show()

