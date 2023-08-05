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

base = "./experiments/jitter/"

# s1 = "pt_uniform"
# s2 = "pt_power"
# s3 = "pt_balance"

s1 = "bsdf-1e5"
s2 = "nee-1e5"
s3 = "mis-balance-1e5"

f1 = s1 + '.txt'
f2 = s2 + '.txt'
f3 = s3 + '.txt'
content1 = ""
content2 = ""
content3 = ""

with open(base+f1) as f:
    content1 = f.readlines()
with open(base+f2) as f:
    content2 = f.readlines()
with open(base+f3) as f:
    content3 = f.readlines()

process(content1)
process(content2)
process(content3)
l1 = len(content1)
l2 = len(content2)
l3 = len(content3)
l = 0
l = max(l, l1)
l = max(l, l2)
l = max(l, l3)

content1 = np.array(content1)
content2 = np.array(content2)
content3 = np.array(content3)

if l > l1:
    content1 = np.append(content1, np.ones(l-l1) * content1[-1])
if l > l2:
    content2 = np.append(content2, np.ones(l-l2) * content2[-1])
if l > l3:
    content3 = np.append(content3, np.ones(l-l3) * content3[-1])

x1 = np.arange(0, l, 1)
x2 = np.arange(0, l, 1)
x3 = np.arange(0, l, 1)
y1 = np.array(content1)
y2 = np.array(content2)
y3 = np.array(content3)
y11 = np.ones(l) * y1[-1]
y22 = np.ones(l) * y2[-1]
y33 = np.ones(l) * y3[-1]

fig, axs = plt.subplots(1, 1)
    
axs.plot(x1, y1,  color='orange', alpha=0.75, linewidth=1, label=s1)
axs.plot(x1, y11, color='orange', alpha=0.25, linewidth=1)
axs.plot(x2, y2,  color='blue',   alpha=0.65, linewidth=1, label=s2)
axs.plot(x2, y22, color='blue',   alpha=0.25, linewidth=1)
axs.plot(x3, y3,  color='g',      alpha=0.75, linewidth=1, label=s3)
axs.plot(x3, y33, color='g',      alpha=0.25, linewidth=1)
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

fig.savefig(base+s1+'-'+s2+'-'+s3+'.png', dpi=300)

plt.show()

