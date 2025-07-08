import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

plt.rcParams['font.family'] = 'Times New Roman'

CSV_PATH = "/home/chan/aerial_ws/src/mav_mppi/scripts/data/kinova_log.csv"

# 조인트 리밋
q_lower = np.array([-6.2832, 0.8203, -6.2832, 0.5236, -6.2832, 1.1345, -6.2832])
q_upper = np.array([6.2832, 5.4629, 6.2832, 5.7596, 6.2832, 5.1487, 6.2832])

def plot_joint_tracking_subplot(df):
    time = df['time']
    q_actual = np.stack([df[f'q{i+1}'] for i in range(7)], axis=1)

    fig, axs = plt.subplots(7, 1, sharex=True, figsize=(10, 16))
    for i in range(7):
        axs[i].plot(time, q_actual[:, i], color='r', label='Actual')
        axs[i].axhline(q_lower[i], color='g', linestyle='--', label='Joint Limit' if i == 0 else "")
        axs[i].axhline(q_upper[i], color='g', linestyle='--')
        axs[i].set_ylabel(f'Joint {i+1}\n[rad]')
        axs[i].set_title(f'Joint {i+1} Tracking', fontname="Times New Roman")
        axs[i].grid()
        if i == 0:
            axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('Joint Tracking', fontname="Times New Roman")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

def plot_eef_states_subplot(df):
    time = df['time']
    eepos = df[['ee_pos_x', 'ee_pos_y', 'ee_pos_z']].values
    rpy = df[['ee_rpy_roll', 'ee_rpy_pitch', 'ee_rpy_yaw']].values

    eepos_target = df[['ee_target_pos_x', 'ee_target_pos_y', 'ee_target_pos_z']].values
    quat_target = df[['ee_target_quat_x', 'ee_target_quat_y', 'ee_target_quat_z', 'ee_target_quat_w']].values
    rpy_target = np.array([R.from_quat(q).as_euler('zyx', degrees=False) for q in quat_target])

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 14))

    label_map = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    unit_map = ['[m]', '[m]', '[m]', '[rad]', '[rad]', '[rad]']

    # x, y, z
    for i, axis in enumerate(['x', 'y', 'z']):
        axs[i].plot(time, eepos[:, i], color='r', label=f'EEF {axis.upper()}')
        axs[i].plot(time, eepos_target[:, i], color='b', linestyle='--', label=f'Target {axis.upper()}')
        axs[i].set_ylabel(f'{label_map[i]}\n{unit_map[i]}')
        axs[i].set_title(f'End-Effector {label_map[i]}', fontname="Times New Roman")
        axs[i].legend()
        axs[i].grid()
    # roll, pitch, yaw
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        axs[i+3].plot(time, rpy[:, i], color='r', label=f'EEF {axis.capitalize()}')
        axs[i+3].plot(time, rpy_target[:, i], color='b', linestyle='--', label=f'Target {axis.capitalize()}')
        axs[i+3].set_ylabel(f'{label_map[i+3]}\n{unit_map[i+3]}')
        axs[i+3].set_title(f'End-Effector {label_map[i+3]}', fontname="Times New Roman")
        axs[i+3].legend()
        axs[i+3].grid()
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('End-Effector Position & Orientation (RPY)', fontname="Times New Roman")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

def plot_cost_subplot(df):
    time = df['time']
    cost_keys = ['cost_stage', 'cost_terminal', 'cost_joint_limit']
    titles = ['Stage Cost', 'Terminal Cost', 'Joint Limit Cost']
    colors = ['r', 'g', 'b']

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for i, key in enumerate(cost_keys):
        if key in df.columns:
            axs[i].plot(time, df[key], color=colors[i], label=titles[i])
            axs[i].set_ylabel('Cost')
            axs[i].set_title(titles[i], fontname="Times New Roman")
            axs[i].legend()
            axs[i].grid()
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('MPPI Cost Log', fontname="Times New Roman")
    plt.tight_layout(rect=[0, 0, 1, 0.97])


def main():
    df = pd.read_csv(CSV_PATH)
    print(df.columns)
    plot_joint_tracking_subplot(df)  # 조인트 트래킹 창
    plot_eef_states_subplot(df)      # EEF 창
    plot_cost_subplot(df)            # Cost plot 창
    plt.show()

if __name__ == "__main__":
    main()
