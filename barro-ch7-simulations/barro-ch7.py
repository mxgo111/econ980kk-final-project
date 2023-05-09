import numpy as np
import matplotlib.pyplot as plt

# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

N = 100 # number of intermediates
A = 1 # TFP
q = 1.01
alpha = 0.5
L = 1
r = 0.02
zeta = 2
pi_bar = A ** (1/(1 - alpha)) * ((1-alpha)/alpha) * (alpha ** (2/(1 - alpha))) * L

def get_Y(X_tilde):
    assert len(X_tilde) == N
    return A * (L ** (1 - alpha)) * np.sum(X_tilde ** alpha)

def get_X_tilde(X, kappa):
    return (q ** kappa) * X

def get_phi(kappa):
    return (1/zeta) * q ** (-100*(kappa + 1) * alpha/(1-alpha))

def get_pi(kappa):
    return pi_bar * q ** (kappa * alpha/(1 - alpha))

def get_Z(kappa):
    return q ** ((kappa + 1) * (alpha/(1-alpha))) * (pi_bar - r * zeta)

def get_p(kappa):
    return get_Z(kappa) * get_phi(kappa)

def get_Q(kappa):
    return np.sum(q ** (kappa * alpha/(1 - alpha)))

def get_delta_Q(kappa):
    return np.dot(get_p(kappa), q ** ((kappa + 1) * (alpha/(1-alpha))) - q ** (kappa * alpha/(1 - alpha)))

def get_growth_Q(kappa):
    return get_delta_Q(kappa) / get_Q(kappa)

total_time = 2000
kappa_init = np.zeros(N)
kappa_curr = kappa_init
kappa_next = kappa_curr
kappa_total = np.zeros((total_time, N))
ps_total = np.zeros((total_time, N))

for t in range(total_time):
    kappa_total[t] = kappa_curr
    ps = get_p(kappa_curr)
    ps_total[t] = ps
    # print(ps)
    # breakpoint()
    for j in range(N):
        increase = np.random.choice(2, p=[1-ps[j], ps[j]])
        kappa_next[j] += increase
    kappa_curr = kappa_next

# Qs = np.array([get_Q(kappa) for kappa in kappa_total])
# plt.plot(Qs)
# plt.show()

duration = 10
fps = 50
fig, ax = plt.subplots()

def make_frame(t):
    ax.clear()
    ax.hist(kappa_total[int(t * fps * 4)])
    ax.set_xlabel("Quality Ladder Level")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Quality Ladder Levels")
    # ax.hist(ps_total[int(t * fps * 4)])
    
    return mplfig_to_npimage(fig)
 
# creating animation
animation = VideoClip(make_frame, duration = duration)
 
# displaying animation with auto play and looping
animation.ipython_display(fps = fps, loop = True, autoplay = True)
