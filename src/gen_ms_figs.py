import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEEDS=[42,0,7,13,21]
OUT="results/multi_seed"
CLASS_NAMES=['BRCA','COAD','GBM','KIRC','LUAD','PRAD']
models=["lr","svm","mlp"]; labels=["Logistic Regression","RBF-SVM","MLP [512,256]"]; colors=["#3C5488","#F39B7F","#E64B35"]

ALL_PC={
    42:dict(lr=[0.9945,1.0,0.9615,1.0,1.0,1.0],svm=[0.9973,1.0,1.0,0.9945,1.0,1.0],mlp=[1.0,1.0,1.0,0.9945,0.9942,1.0]),
    0: dict(lr=[0.9973,0.9932,0.9615,1.0,1.0,1.0],svm=[0.9919,0.9863,0.9796,0.9945,0.9942,1.0],mlp=[0.9973,0.9863,1.0,0.9945,0.9884,1.0]),
    7: dict(lr=[0.9973,0.9933,0.9804,1.0,0.9882,0.9940],svm=[1.0,0.9863,1.0,0.9945,0.9942,1.0],mlp=[0.9973,0.9933,1.0,1.0,0.9882,1.0]),
    13:dict(lr=[1.0,1.0,1.0,1.0,1.0,1.0],svm=[1.0,1.0,1.0,1.0,1.0,1.0],mlp=[1.0,0.9933,1.0,1.0,0.9942,1.0]),
    21:dict(lr=[0.9973,1.0,0.9804,1.0,1.0,1.0],svm=[0.9919,1.0,1.0,0.9945,0.9882,1.0],mlp=[0.9973,1.0,1.0,1.0,0.9942,1.0]),
}
CV={42:dict(lr=0.9968,svm=0.9951,mlp=0.9983),0:dict(lr=0.9971,svm=0.9962,mlp=0.9961),7:dict(lr=0.9967,svm=0.9955,mlp=0.9978),13:dict(lr=0.9965,svm=0.9965,mlp=0.9972),21:dict(lr=0.9966,svm=0.9949,mlp=0.9967)}
TEST={42:dict(lr=0.9927,svm=0.9986,mlp=0.9981),0:dict(lr=0.9920,svm=0.9911,mlp=0.9944),7:dict(lr=0.9922,svm=0.9958,mlp=0.9965),13:dict(lr=1.0,svm=1.0,mlp=0.9979),21:dict(lr=0.9963,svm=0.9958,mlp=0.9986)}

# fig5: CV vs test scatter
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for ax,m,label,color in zip(axes,models,labels,colors):
    cv_f1s=[CV[s][m] for s in SEEDS]; test_f1s=[TEST[s][m] for s in SEEDS]
    ax.scatter(cv_f1s,test_f1s,color=color,s=70,edgecolors="black",linewidths=0.7,zorder=3)
    for s,cx,ty in zip(SEEDS,cv_f1s,test_f1s):
        ax.annotate(str(s),(cx,ty),fontsize=8,xytext=(4,4),textcoords="offset points")
    lo=min(cv_f1s+test_f1s)-0.005; hi=max(cv_f1s+test_f1s)+0.005
    ax.plot([lo,hi],[lo,hi],"k--",alpha=0.3,linewidth=1)
    ax.set_xlabel("CV Macro-F1 (mean)"); ax.set_ylabel("Test Macro-F1"); ax.set_title(label); ax.grid(alpha=0.3)
plt.suptitle("CV F1 vs Test F1 per Seed  (diagonal = perfect agreement)",fontsize=12)
plt.tight_layout()
p=os.path.join(OUT,"ms_fig5_cv_vs_test.png")
plt.savefig(p,dpi=300,bbox_inches="tight"); plt.close(); print(f"Saved {p}")

# fig6: per-class all models
fig,axes=plt.subplots(1,3,figsize=(18,4))
for ax,m,label in zip(axes,models,labels):
    pc=np.array([ALL_PC[s][m] for s in SEEDS])
    sns.heatmap(pc,annot=True,fmt=".3f",xticklabels=CLASS_NAMES,yticklabels=[f"seed={s}" for s in SEEDS],cmap="YlOrRd_r",vmin=0.96,vmax=1.0,linewidths=0.4,ax=ax,cbar_kws={"label":"F1"})
    ax.set_title(label); ax.set_xlabel("Cancer Type"); ax.set_ylabel("Seed")
plt.suptitle("Per-Class F1 Across 5 Seeds — All Models",fontsize=13)
plt.tight_layout()
p=os.path.join(OUT,"ms_fig6_perclass_all_models.png")
plt.savefig(p,dpi=300,bbox_inches="tight"); plt.close(); print(f"Saved {p}")
