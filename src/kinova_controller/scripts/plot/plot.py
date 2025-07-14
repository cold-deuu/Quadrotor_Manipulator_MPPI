import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'

# CSV 경로 지정
csv_path = '/home/chan/aerial_ws/src/mav_mppi/scripts/data/time_horizon/07-14_18_/test.csv'
df = pd.read_csv(csv_path)
timesteps = df.index
print(f"timesteps : {timesteps}")

target_x = df['target_pos_x'].iloc[0]
target_y = df['target_pos_y'].iloc[0]
target_z = df['target_pos_z'].iloc[0]

# 날짜/시간 폴더 이름 추출
folder_name = os.path.basename(os.path.dirname(csv_path))  # '07-14_12-04'
save_dir = os.path.join("/home/chan/aerial_ws/src/mav_mppi/scripts/plot", folder_name)
os.makedirs(save_dir, exist_ok=True)

# --- EEF POSITION FIGURE ---
fig1, axes1 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig1.suptitle(f'Kinova EEF Position : {folder_name}', fontsize=14)

axes1[0].plot(timesteps, df['eef_pos_x'], label='eef_pos_x', color='b')
axes1[0].axhline(y=target_x, color='r', linestyle='--', label='target_pos_x')
axes1[0].set_ylabel('EEF Pos_X')
axes1[0].legend()
axes1[0].grid(True)

axes1[1].plot(timesteps, df['eef_pos_y'], label='eef_pos_y', color='b')
axes1[1].axhline(y=target_y, color='r', linestyle='--', label='target_pos_y')
axes1[1].set_ylabel('EEF Pos_Y')
axes1[1].legend()
axes1[1].grid(True)

axes1[2].plot(timesteps, df['eef_pos_z'], label='eef_pos_z', color='b')
axes1[2].axhline(y=target_z, color='r', linestyle='--', label='target_pos_z')
axes1[2].set_ylabel('EEF Pos_Z')
axes1[2].set_xlabel('Timestep')
axes1[2].legend()
axes1[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
eef_save_path = os.path.join(save_dir, "eef_pos.png")
plt.savefig(eef_save_path)
print(f"EEF position plot saved at: {eef_save_path}")
plt.show()
# --- DRONE POSITION FIGURE ---
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig2.suptitle(f'Drone Position : {folder_name}', fontsize=14)

axes2[0].plot(timesteps, df['drone_pos_x'], label='drone_pos_x', color='g')
axes2[0].set_ylabel('Drone Pos_X')
axes2[0].legend()
axes2[0].grid(True)

axes2[1].plot(timesteps, df['drone_pos_y'], label='drone_pos_y', color='g')
axes2[1].set_ylabel('Drone Pos_Y')
axes2[1].legend()
axes2[1].grid(True)

axes2[2].plot(timesteps, df['drone_pos_z'], label='drone_pos_z', color='g')
axes2[2].set_ylabel('Drone Pos_Z')
axes2[2].set_xlabel('Timestep')
axes2[2].legend()
axes2[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
drone_save_path = os.path.join(save_dir, "drone_pos.png")
plt.savefig(drone_save_path)
print(f"Drone position plot saved at: {drone_save_path}")
plt.show()