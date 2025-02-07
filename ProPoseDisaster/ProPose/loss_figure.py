import json
import numpy as np
import matplotlib.pyplot as plt

# file_name = 'total_loss.json'
# with open(file_name, "r", encoding="utf-8") as json_file:
#     total_loss = json.load(json_file)
file_name = 'metrics.json'
with open(file_name, "r", encoding="utf-8") as json_file:
    metrics = json.load(json_file)
file_name = 'Loss_detail.json'
with open(file_name, "r", encoding="utf-8") as json_file:
    loss_detail = json.load(json_file)
file_name = 'total_loss.json'
with open(file_name, "r", encoding="utf-8") as json_file:
    total_loss = json.load(json_file)

#----------------------------total loss-----------------------
total_loss = np.array(total_loss)

#----------------------------key point loss-------------------
err_kpts = []
err_posts = []
for i in range(len(metrics)):
    err_kpts.append(metrics[i]['err_kpt'])
    err_posts.append(metrics[i]['err_post'])
err_kpts = np.array(err_kpts)
err_posts = np.array(err_posts)

#-------------------beta theta reproject loss-------------------
detail_dict = []
hms , betas, thetas, reprojects = [], [], [], []
for data in loss_detail:
    data = data.split("|")
    parsed_data = {}
    for item in data:
        key, value = item.split(":")  
        key = key.strip()  
        value = float(value.strip())  
        parsed_data[key] = value
    detail_dict.append(parsed_data)

for i in range(len(detail_dict)):
    hms.append(detail_dict[i]['hm'])
    betas.append(detail_dict[i]['beta'])
    thetas.append(detail_dict[i]['theta'])
    reprojects.append(detail_dict[i]['2d'])
hms = np.array(hms)
betas = np.array(betas)
thetas = np.array(thetas)
reprojects = np.array(reprojects)   


x = np.arange(len(total_loss))

#-----------------------------figure of detailed loss and error
fig, axs = plt.subplots(2, 2, figsize=(12, 8))  
axs[0, 0].plot(x, betas, color="blue")
axs[0, 0].set_title("shape param loss")
axs[0, 0].set_xlabel("epoch")
axs[0, 0].set_ylabel("loss")
axs[0, 0].legend()

axs[0, 1].plot(x, thetas, color="green")
axs[0, 1].set_title("pose param loss")
axs[0, 1].set_xlabel("epoch")
axs[0, 1].set_ylabel("loss")
axs[0, 1].legend()

axs[1, 0].plot(x, reprojects, color="red")
axs[1, 0].set_title("2D Prejoction loss")
axs[1, 0].set_xlabel("epoch")
axs[1, 0].set_ylabel("loss")
axs[1, 0].legend()

axs[1, 1].plot(x, err_kpts, color="purple")
axs[1, 1].set_title("key point error")
axs[1, 1].set_xlabel("epoch")
axs[1, 1].set_ylabel("error")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

#------------------------figure of total loss--------------
plt.figure(figsize=(8, 6))   # 设置画布大小
plt.plot(x, total_loss, color="blue", linewidth=2)  # 折线图
plt.title("Total Loss", fontsize=16)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.ylim((4,30))
plt.yticks(np.arange(4, 30, 0.5))  
plt.legend()
plt.show()