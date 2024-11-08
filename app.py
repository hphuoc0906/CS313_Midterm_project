import streamlit as st
from utils import *
import os
from numpy.random import default_rng
from matplotlib.collections import LineCollection

def read_traj(time, day):
    path = os.path.join('./Data-mining', time, day)

    percent = 1
    tdriveGPS = os.listdir(path)
    numfile = int(percent * len(tdriveGPS))

    rng = default_rng()
    idx = rng.choice([int(gps.split('.')[0]) for gps in tdriveGPS] , size=numfile, replace=False)

    traj_data = []
    for id in idx:
        filepath = os.path.join(path, str(id) + '.txt')
        with open(filepath, 'r') as f:
            traj = []
            for i, line in enumerate(f):
                    if(i == 0): 
                        continue
                    content = line.split(',')
                    x, y = float(content[-2]), float(content[-1])
                    traj.append([x, y])
            traj_data.append(np.array(traj))
    return traj_data

def plot_samples(traj_data, n_plot=6):
    if not traj_data:
        st.warning("No data available to plot.")
        return
    
    n_cols = 3
    n_rows = (min(n_plot, len(traj_data)) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle("Random Sample Trajectories", fontsize=16)
    
    sampled_indices = np.random.choice(len(traj_data), min(n_plot, len(traj_data)), replace=False)

    for i, idx in enumerate(sampled_indices):
        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        traj = traj_data[idx]
        ax.plot(traj[:, 0], traj[:, 1], marker='o', linestyle='-', markersize=3, alpha=0.6)
        ax.set_title(f"Sample {idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

    for j in range(i + 1, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')

    st.pyplot(fig)

def plot(rep, csegs):
    plt.figure(figsize=(8, 6))

    line_segments = LineCollection(csegs, colors='gray', linestyle='--')
    plt.gca().add_collection(line_segments)
    points = np.array([[seg[0], seg[1]] for seg in csegs]).reshape(-1, 2)

    plt.scatter(points[:, 0], points[:, 1], color='gray', edgecolors='black', marker='o')

    plt.plot(rep[:, 0], rep[:, 1], 'r', linewidth=3, marker='o', label="Representative Trajectory")

    plt.title('Representative Trajectory of a Cluster')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)



def process(traj_data):
    preprocessed_data = preprocessing(traj_data)

    sto_traj = []
    for traj in preprocessed_data:
        cp = greedy_characteristic_points(traj)
        sto_traj.append(cp)

    segments = to_segments(sto_traj)
    
    vectors = None
    for par in sto_traj:
        if(len(par) < 2): continue
        if(vectors is None):
            vectors = extract_feature_vector(par)
        else:
            tmp = extract_feature_vector(par)
            vectors = np.concatenate((vectors, tmp), axis=0)

    cls = line_segment_clustering(vectors, C=1, min_samples=5)

    for cluster_id in range(0, 5):
        ids = np.where(cls[0] == cluster_id)[0].tolist()
        csegs = [segments[i] for i in ids]

    ids = np.where(cls[0] == 0)[0].tolist()
    csegs = [segments[i] for i in ids]

    rep = get_representative_trajectory(csegs)
    rep = smooth_trajectory(rep, window_size=21)

    plot(rep, csegs)

st.markdown("<h1 style='text-align: center;'>ĐỒ ÁN CS313</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>BEIJING TDRIVER GPS 2008</h3>", unsafe_allow_html=True)

hour = st.selectbox("Chọn giờ (trong 24 giờ):", [f"{h}h" for h in range(24)])

date = st.selectbox("Chọn ngày:", [
    "03/08/2008",
    "04/08/2008",
    "05/08/2008",
    "06/08/2008",
    "07/08/2008",
    "08/08/2008"
])

day = int(date.split('/')[0])
traj_data = read_traj(hour[:-1], str(day))


if st.button("Process and Plot"):
    process(traj_data)

if st.button("Plot some trajs"):
    plot_samples(traj_data)