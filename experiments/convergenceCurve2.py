import matplotlib.pyplot as plt
import numpy as np

def process(s):
    l = len(s)
    for i in range(l):
        s[i] = s[i].strip()
        s[i] = s[i][1:-1]
        s[i] = s[i].split(",")
        l1 = len(s[i])
        sum = 0
        for j in range(l1):
            sum += float(s[i][j].strip())
        s[i] = sum / 3

base = "./experiments/results/jitter/"
s1 = "bdpt-1e5"
s2 = "mis-balance-1e5"
f1 = s1 + '.txt'
f2 = s2 + '.txt'
content1 = ""
content2 = ""

with open(base+f1) as f:
    content1 = f.readlines()
with open(base+f2) as f:
    content2 = f.readlines()

process(content1)
process(content2)
l1 = len(content1)
l2 = len(content2)
l = 0
l = max(l, l1)
l = max(l, l2)
content1 = np.array(content1)
content2 = np.array(content2)
if l > l1:
    content1 = np.append(content1, np.ones(l-l1) * content1[-1])
if l > l2:
    content2 = np.append(content2, np.ones(l-l2) * content2[-1])

x = np.arange(0, l, 1)
y1 = np.array(content1)
y2 = np.array(content2)
# y4 = np.zeros(l1)
y11 = np.ones(l) * y1[-1]
y22 = np.ones(l) * y2[-1]

fig, axs = plt.subplots(1, 1)
    
axs.plot(x, y1,  color='orange', alpha=0.75,  linewidth=0.5, label=s1)
axs.plot(x, y11, color='orange', alpha=0.25,  linewidth=0.5)
axs.plot(x, y2,  color='green',   alpha=0.65, linewidth=0.5, label=s2)
axs.plot(x, y22, color='green',   alpha=0.25, linewidth=0.5)
axs.set_xlabel('iters')
axs.set_ylabel('color')
# axs[0].set_ylim(0, 1)
axs.grid(True)

# axs[1].plot(x, y3, label=s1+'-'+s2, color='green')
# axs[1].plot(x, y4, color='grey')
# axs[1].set_xlabel('iters')
# axs[1].set_ylabel('delta')
# # axs[1].set_ylim(-1, 1)
# axs[1].grid(True)

fig.tight_layout()
fig.legend()

fig.savefig(base+s1+'-'+s2+'.png', dpi=600)

plt.show()

