import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "/home/chan/aerial_ws/src/mav_mppi/scripts/data/drone_log.csv"

def plot_drone_states(df):
    time = df['time']

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 15))

    # 1~3: 위치 (x, y, z) + 목표 위치
    for i, axis in enumerate(['x', 'y', 'z']):
        axs[i].plot(time, df[f'pos_{axis}'], label=f'pos_{axis}')
        axs[i].plot(time, df[f'xdes_{axis}'], label=f'xdes_{axis}', linestyle='--')
        axs[i].set_ylabel(f'{axis.upper()} [m]')
        axs[i].set_title(f'Drone {axis.upper()} Position (solid: actual, dashed: target)')
        axs[i].legend()
        axs[i].grid()

    # 4~6: 속도 (x, y, z)
    for i, axis in enumerate(['x', 'y', 'z']):
        axs[i+3].plot(time, df[f'vel_{axis}'], label=f'vel_{axis}')
        axs[i+3].set_ylabel(f'V{axis.upper()} [m/s]')
        axs[i+3].set_title(f'Drone {axis.upper()} Velocity')
        axs[i+3].legend()
        axs[i+3].grid()

    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('Drone Position and Velocity')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_drone_rpy(df):
    time = df['time']
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for i, (col, label) in enumerate(zip(['roll', 'pitch', 'yaw'], ['Roll', 'Pitch', 'Yaw'])):
        axs[i].plot(time, df[col], label=label)
        axs[i].set_ylabel(f'{label} [rad]')
        axs[i].set_title(f'Drone {label}')
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('Drone Orientation (RPY)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def main():
    df = pd.read_csv(CSV_PATH)
    plot_drone_states(df)
    plot_drone_rpy(df)

if __name__ == "__main__":
    main()
